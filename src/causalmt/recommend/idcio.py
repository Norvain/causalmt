"""IDCIO：交互效应分解的组合干预优化（Interaction-Decomposed Combined Intervention Optimization）。

三阶段管线：
    1. 原子干预效应估计：τ̂_k(x) 来自 GACRNet（from_estimator）或外部输入（from_effects）
    2. 交互效应学习：InteractionMLP 拟合 τ̂_{ij}(x)，使用组合干预下的样本
    3. QUBO 求解：穷举或贪心搜索最优 binary action 向量

典型用法::

    from causalmt import GACRNet, IDCIO

    # 路径 A：与 GACRNet 配套
    atomic = GACRNet(num_treatments=3).fit(x, t_atomic, y)
    rec = IDCIO.from_estimator(atomic_estimator=atomic).fit_interaction(
        x_combined, y_combined, treatment_pairs=[(0, 1)]
    )
    actions = rec.recommend(x_new, method="exhaustive")

    # 路径 B：解耦外部 atomic 效应（如 EconML）
    rec = IDCIO.from_effects(atomic_effects=tau_array).fit_interaction(...)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from causalmt.nn import InteractionMLP
from causalmt.recommend.mc_dropout import mc_dropout_predict
from causalmt.recommend.qubo import (
    EXHAUSTIVE_MAX_K,
    build_qubo_matrix,
    solve_qubo_exhaustive,
    solve_qubo_greedy,
)
from causalmt.trainer import TrainConfig, Trainer
from causalmt.utils import get_logger, resolve_device, set_seed, to_numpy, to_tensor

__all__ = ["IDCIO"]


@dataclass
class _IDCIOConfig:
    """IDCIO 配置（不含 atomic_estimator 引用，便于序列化）。"""

    n_atomic_treatments: int  # 真实干预数（不含控制组）
    control_index: int = 0  # GACRNet 中控制组索引
    # InteractionMLP 架构
    interaction_hidden: tuple[int, ...] = (100, 50)
    interaction_dropout: float = 0.1
    # 训练
    epochs: int = 200
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    early_stopping_patience: int | None = 20
    val_split: float = 0.2
    # MC Dropout
    use_mc_dropout: bool = False
    mc_samples: int = 100
    # 系统
    random_state: int | None = 42
    verbose: bool = True
    # 记录已 fit 的交互对
    fitted_pairs: list[tuple[int, int]] = field(default_factory=list)


class IDCIO:
    """组合干预推荐器。

    属性:
        config: IDCIO 配置快照
        atomic_estimator: 原子干预模型（可选，来自 from_estimator）
        atomic_effects_cache: 外部 atomic 效应（可选，来自 from_effects）
        interaction_models: dict[(i, j), InteractionMLP]，每个交互对一个模型
    """

    def __init__(
        self,
        n_atomic_treatments: int,
        *,
        atomic_estimator: Any = None,
        atomic_effects_cache: np.ndarray | None = None,
        interaction_hidden: Sequence[int] = (100, 50),
        interaction_dropout: float = 0.1,
        epochs: int = 200,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        early_stopping_patience: int | None = 20,
        val_split: float = 0.2,
        use_mc_dropout: bool = False,
        mc_samples: int = 100,
        device: str | torch.device = "auto",
        random_state: int | None = 42,
        verbose: bool = True,
        control_index: int = 0,
    ) -> None:
        if n_atomic_treatments < 2:
            raise ValueError(f"n_atomic_treatments 至少为 2 才有组合，得 {n_atomic_treatments}")
        self.config = _IDCIOConfig(
            n_atomic_treatments=n_atomic_treatments,
            control_index=control_index,
            interaction_hidden=tuple(interaction_hidden),
            interaction_dropout=interaction_dropout,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            early_stopping_patience=early_stopping_patience,
            val_split=val_split,
            use_mc_dropout=use_mc_dropout,
            mc_samples=mc_samples,
            random_state=random_state,
            verbose=verbose,
        )
        self.atomic_estimator = atomic_estimator
        self.atomic_effects_cache = atomic_effects_cache
        self.device = resolve_device(device)
        self._logger = get_logger("causalmt.idcio")
        self.interaction_models: dict[tuple[int, int], InteractionMLP] = {}

    # ------------------------------------------------------------------ 构造
    @classmethod
    def from_estimator(cls, atomic_estimator: Any, **kwargs: Any) -> IDCIO:
        """配套 GACRNet 使用。

        Args:
            atomic_estimator: 已训练的 GACRNet 实例
            **kwargs: 透传给 __init__

        Note:
            n_atomic_treatments 自动设为 estimator.config.num_treatments - 1
            （减去控制组）。
        """
        if not hasattr(atomic_estimator, "config") or not hasattr(
            atomic_estimator, "predict_potential_outcomes"
        ):
            raise TypeError(
                "atomic_estimator 需为 GACRNet 实例（含 .config 和 .predict_potential_outcomes）"
            )
        n = atomic_estimator.config.num_treatments - 1
        return cls(
            n_atomic_treatments=n,
            atomic_estimator=atomic_estimator,
            **kwargs,
        )

    @classmethod
    def from_effects(cls, atomic_effects: np.ndarray, **kwargs: Any) -> IDCIO:
        """解耦使用，直接接受外部原子效应数组。

        Args:
            atomic_effects: shape (n, K_atomic) 每个样本在每个原子干预下的预测效应
            **kwargs: 透传给 __init__
        """
        arr = np.asarray(atomic_effects)
        if arr.ndim != 2:
            raise ValueError(f"atomic_effects 应为 2D (n, K)，得 shape {arr.shape}")
        return cls(
            n_atomic_treatments=arr.shape[1],
            atomic_effects_cache=arr,
            **kwargs,
        )

    # ------------------------------------------------------------------ fit
    def fit_interaction(
        self,
        x_combined,  # type: ignore[no-untyped-def]
        y_combined,  # type: ignore[no-untyped-def]
        *,
        treatment_pairs: Sequence[tuple[int, int]] | None = None,
        atomic_effects_combined: np.ndarray | None = None,
    ) -> IDCIO:
        """训练交互效应 MLP。

        Args:
            x_combined: shape (n, d) 接受组合干预的样本协变量
            y_combined: shape (n,) 观察结果（注：这里 y 本身已经是组合下的结果）
            treatment_pairs: 要拟合的交互对列表，例如 [(0, 1), (0, 2)]。
                None 时拟合所有 K*(K-1)/2 对。
            atomic_effects_combined: 可选 (n, K) 显式提供该批样本的原子效应预测；
                未提供时若有 atomic_estimator 则自动调用，否则要求 from_effects 时已缓存

        Note:
            交互项的标签按 Sun et al.(2024) 风格构造：
                y_{ij} = y_obs - μ̂_0 - τ̂_i - τ̂_j
            其中 μ̂_0 是控制组潜在结果预测，τ̂_i / τ̂_j 是原子效应。
        """
        cfg = self.config
        if cfg.random_state is not None:
            set_seed(cfg.random_state)

        x = to_tensor(x_combined, dtype=torch.float32)
        y = to_tensor(y_combined, dtype=torch.float32).reshape(-1)
        if x.dim() != 2 or x.shape[0] != y.shape[0]:
            raise ValueError("x_combined/y_combined 形状不匹配")

        k = cfg.n_atomic_treatments
        pairs = self._resolve_pairs(treatment_pairs, k)

        # 获取原子效应（来自 estimator 或外部缓存）
        atomic = self._get_atomic_effects(x.numpy(), atomic_effects_combined)  # (n, K)
        # 控制组基准 μ̂_0
        mu0 = self._get_mu0(x.numpy())  # (n,) 或 None

        for i, j in pairs:
            target = self._build_interaction_target(y.numpy(), mu0, atomic, i, j)
            target_t = torch.as_tensor(target, dtype=torch.float32)

            model = self._train_one_pair(x, target_t, pair=(i, j))
            self.interaction_models[(i, j)] = model
            if (i, j) not in cfg.fitted_pairs:
                cfg.fitted_pairs.append((i, j))

        return self

    def _resolve_pairs(
        self,
        pairs: Sequence[tuple[int, int]] | None,
        k: int,
    ) -> list[tuple[int, int]]:
        if pairs is None:
            return [(i, j) for i in range(k) for j in range(i + 1, k)]
        out = []
        for i, j in pairs:
            if not 0 <= i < j < k:
                raise ValueError(f"交互对 ({i}, {j}) 不满足 0 ≤ i < j < {k}")
            out.append((int(i), int(j)))
        return out

    def _build_interaction_target(
        self,
        y_obs: np.ndarray,
        mu0: np.ndarray | None,
        atomic: np.ndarray,
        i: int,
        j: int,
    ) -> np.ndarray:
        """构造交互项训练标签：y_{ij} = y_obs - μ̂_0 - τ̂_i - τ̂_j。

        若没有 mu0（外部 effects 模式），退化为 y_{ij} = y_obs - τ̂_i - τ̂_j。
        """
        residual = y_obs.astype(np.float64)
        if mu0 is not None:
            residual -= mu0
        residual -= atomic[:, i]
        residual -= atomic[:, j]
        return residual.astype(np.float32)

    def _get_atomic_effects(
        self,
        x_np: np.ndarray,
        explicit: np.ndarray | None,
    ) -> np.ndarray:
        """获取协变量对应的原子效应 (n, K)。"""
        if explicit is not None:
            arr = np.asarray(explicit)
            if arr.shape != (x_np.shape[0], self.config.n_atomic_treatments):
                raise ValueError(
                    f"atomic_effects_combined 形状应为 ({x_np.shape[0]}, "
                    f"{self.config.n_atomic_treatments})，得 {arr.shape}"
                )
            return arr
        if self.atomic_estimator is not None:
            po = self.atomic_estimator.predict_potential_outcomes(x_np)  # (n, K_total)
            ctrl = self.config.control_index
            non_ctrl = [k for k in range(po.shape[1]) if k != ctrl]
            mu0 = po[:, ctrl]
            return po[:, non_ctrl] - mu0[:, None]
        if self.atomic_effects_cache is not None:
            if self.atomic_effects_cache.shape[0] != x_np.shape[0]:
                raise ValueError(
                    "from_effects 缓存的 atomic_effects 行数和 x_combined 不一致；"
                    "请显式传入 atomic_effects_combined 参数"
                )
            return self.atomic_effects_cache
        raise RuntimeError(
            "无法获取 atomic 效应：未提供 atomic_estimator、atomic_effects_cache、" "或 atomic_effects_combined。"
        )

    def _get_mu0(self, x_np: np.ndarray) -> np.ndarray | None:
        if self.atomic_estimator is None:
            return None
        po = self.atomic_estimator.predict_potential_outcomes(x_np)
        return po[:, self.config.control_index]

    def _train_one_pair(
        self, x: torch.Tensor, target: torch.Tensor, *, pair: tuple[int, int]
    ) -> InteractionMLP:
        cfg = self.config
        n = x.shape[0]

        if cfg.val_split > 0 and n > 4:
            n_val = max(1, int(n * cfg.val_split))
            perm = torch.randperm(n)
            val_idx, tr_idx = perm[:n_val], perm[n_val:]
            x_tr, y_tr = x[tr_idx], target[tr_idx]
            x_val, y_val = x[val_idx], target[val_idx]
        else:
            x_tr, y_tr = x, target
            x_val = y_val = None

        model = InteractionMLP(
            input_dim=x.shape[1],
            hidden_dims=cfg.interaction_hidden,
            dropout=cfg.interaction_dropout,
        )

        def _step(net: torch.nn.Module, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
            xb, yb = batch
            pred = net(xb).squeeze(-1)
            return torch.nn.functional.mse_loss(pred, yb)

        trainer = Trainer(
            model=model,
            loss_fn=_step,
            config=TrainConfig(
                epochs=cfg.epochs,
                batch_size=cfg.batch_size,
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
                early_stopping_patience=cfg.early_stopping_patience,
                verbose=cfg.verbose,
            ),
            device=self.device,
        )
        val = (x_val, y_val) if x_val is not None else None
        trainer.fit((x_tr, y_tr), val)
        if cfg.verbose:
            self._logger.info("trained interaction pair %s", pair)
        return model

    # ------------------------------------------------------------------ 推断
    @torch.no_grad()
    def predict_interactions(
        self, x
    ) -> dict[tuple[int, int], np.ndarray]:  # type: ignore[no-untyped-def]
        """预测每个已 fit 交互对的效应，dict 形如 {(i, j): (n,)}。"""
        if not self.interaction_models:
            raise RuntimeError("尚未训练任何交互对，请先调用 fit_interaction(...)")
        x_t = to_tensor(x, dtype=torch.float32, device=self.device)
        out: dict[tuple[int, int], np.ndarray] = {}
        for pair, model in self.interaction_models.items():
            model.eval()
            pred = model(x_t).squeeze(-1)
            out[pair] = to_numpy(pred)
        return out

    def predict_interactions_with_uncertainty(
        self, x  # type: ignore[no-untyped-def]
    ) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
        """带不确定性的交互项预测，dict 形如 {(i, j): (mean, std)}。"""
        if not self.config.use_mc_dropout:
            raise RuntimeError("需要 use_mc_dropout=True 才能调用 MC Dropout 推断")
        if not self.interaction_models:
            raise RuntimeError("尚未训练任何交互对")
        x_t = to_tensor(x, dtype=torch.float32, device=self.device)
        out: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        for pair, model in self.interaction_models.items():
            mean, std = mc_dropout_predict(model, x_t, n_samples=self.config.mc_samples)
            out[pair] = (mean.reshape(-1), std.reshape(-1))
        return out

    # ------------------------------------------------------------------ 推荐
    def recommend(
        self,
        x,  # type: ignore[no-untyped-def]
        *,
        costs: np.ndarray | None = None,
        method: str = "exhaustive",
        maximize: bool = True,
        return_uncertainty: bool = False,
        atomic_effects: np.ndarray | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """推荐每个样本的最优组合干预。

        Args:
            x: shape (n, d) 待推荐样本
            costs: shape (K,) 每个原子干预的单位成本（不含控制组）
            method: "exhaustive" | "greedy"
            maximize: True 最大化 V(z)，False 最小化
            return_uncertainty: True 时同时返回基于 MC Dropout 的 action 置信度
            atomic_effects: 可选，显式提供 (n, K) 原子效应，跳过 estimator 调用

        Returns:
            actions: shape (n, K) 0/1 binary action 矩阵
            (可选) confidence: shape (n,) 推荐的置信度（每行最大概率方向）
        """
        x_np = np.asarray(to_numpy(x) if not isinstance(x, np.ndarray) else x)
        atomic = self._get_atomic_effects(x_np, atomic_effects)  # (n, K)

        # 收集交互项预测
        interactions: dict[tuple[int, int], np.ndarray] = {}
        for pair, model in self.interaction_models.items():
            x_t = to_tensor(x_np, dtype=torch.float32, device=self.device)
            model.eval()
            with torch.no_grad():
                interactions[pair] = to_numpy(model(x_t).squeeze(-1))

        q = build_qubo_matrix(atomic, interactions)  # (n, K, K)
        actions = self._solve(q, costs=costs, method=method, maximize=maximize)[0]

        if not return_uncertainty:
            return actions

        confidence = self._action_confidence(x_np, q, costs, method, maximize, actions)
        return actions, confidence

    def _solve(
        self,
        q: np.ndarray,
        *,
        costs: np.ndarray | None,
        method: str,
        maximize: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        if method == "exhaustive":
            if q.shape[-1] > EXHAUSTIVE_MAX_K:
                if self.config.verbose:
                    self._logger.warning(
                        "K=%d 超过穷举上限 %d，自动切换 greedy",
                        q.shape[-1],
                        EXHAUSTIVE_MAX_K,
                    )
                return solve_qubo_greedy(q, costs=costs, maximize=maximize)
            return solve_qubo_exhaustive(q, costs=costs, maximize=maximize)
        if method == "greedy":
            return solve_qubo_greedy(q, costs=costs, maximize=maximize)
        raise ValueError(f"未知 method={method!r}，可选: 'exhaustive' | 'greedy'")

    def _action_confidence(
        self,
        x_np: np.ndarray,
        q_mean: np.ndarray,
        costs: np.ndarray | None,
        method: str,
        maximize: bool,
        actions: np.ndarray,
    ) -> np.ndarray:
        """基于交互项 MC Dropout 采样，估计每个样本推荐 action 的稳定性。

        多次重新求解 QUBO，统计推荐与最终 action 一致的比例作为置信度。
        """
        if not self.config.use_mc_dropout:
            raise RuntimeError("需要 use_mc_dropout=True 才能返回 uncertainty")

        n = x_np.shape[0]
        agreements = np.zeros(n, dtype=np.int64)
        x_t = to_tensor(x_np, dtype=torch.float32, device=self.device)

        for _ in range(self.config.mc_samples):
            interactions_sample: dict[tuple[int, int], np.ndarray] = {}
            for pair, model in self.interaction_models.items():
                with torch.no_grad():
                    # 强制开启 dropout 一次前向
                    model.train(True)
                    interactions_sample[pair] = to_numpy(model(x_t).squeeze(-1))
                    model.train(False)
            q_sample = build_qubo_matrix(self._get_atomic_effects(x_np, None), interactions_sample)
            sample_actions, _ = self._solve(q_sample, costs=costs, method=method, maximize=maximize)
            agreements += (sample_actions == actions).all(axis=1).astype(np.int64)

        return agreements / self.config.mc_samples
