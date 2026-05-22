"""GACRNet：sklearn 风格的多值干预因果效应估计器。

高层用户入口。封装 GACRNetModule + LossFunction + Trainer，
提供 fit/predict_potential_outcomes/estimate_ate/estimate_cate/save/load 接口。

典型用法::

    from causalmt import GACRNet
    est = GACRNet(num_treatments=3, use_routing=True, use_mmd=True)
    est.fit(x_train, t_train, y_train)
    ate = est.estimate_ate(x_test, treatment_a=1, treatment_b=0)
    cate = est.estimate_cate(x_test, treatment_a=1, treatment_b=0)
    y_potential = est.predict_potential_outcomes(x_test)  # (n, K)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from causalmt.losses import LossFunction
from causalmt.nn import GACRNetModule
from causalmt.trainer import TrainConfig, Trainer
from causalmt.utils import get_logger, resolve_device, set_seed, to_numpy, to_tensor

__all__ = ["GACRNet"]


@dataclass
class _GACRNetConfig:
    """GACRNet 全部超参数快照（用于 save/load）。"""

    num_treatments: int
    input_dim: int | None = None
    # 架构
    backbone: str = "mlp"
    head: str = "uplift"
    routing: str = "gps"
    hidden_dim: int = 200
    head_hidden: int = 100
    dropout: float = 0.0
    # 正则
    use_mmd: bool = True
    mmd_weight: float = 1.0
    mmd_gammas: tuple[float, ...] = (0.1, 1.0, 10.0)
    use_tarreg: bool = True
    tarreg_ratio: float = 1.0
    gps_weight: float = 1.0
    reg_l2: float = 0.01
    epsilon_clip: float = 0.1
    # 训练
    epochs: int = 300
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    early_stopping_patience: int | None = 30
    val_split: float = 0.2
    # 系统
    random_state: int | None = 42
    verbose: bool = True
    # Transformer 专用
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 256


class GACRNet:
    """GPS-Attention Causal Routing Network。

    多值干预因果效应估计模型，扩展 DragonNet 到 K 类干预，集成 GPS 路由注意力、
    多尺度 MMD 表示正则、目标正则化。

    Args:
        num_treatments: 干预类别数 K（含控制组，索引 0）
        backbone: "mlp" | "transformer"
        head: "uplift"（μ₀+Δ_k 分解）| "separate"（每干预独立 head）
        routing: "gps"（默认）| "self" | "concat" | "none"
        hidden_dim: 共享表示维度
        head_hidden: 结果预测 head 隐藏维度
        dropout: dropout 概率
        use_mmd / mmd_weight / mmd_gammas: 多尺度 MMD 表示正则开关与超参
        use_tarreg / tarreg_ratio: 目标正则化开关与权重
        gps_weight: GPS 分类损失权重
        reg_l2: outcome head + routing 的 L2 正则系数
        epochs / batch_size / lr / weight_decay: 训练超参
        early_stopping_patience: 早停耐心轮数，None 关闭
        val_split: fit 时如未提供 val_data，从训练集切分的比例
        device: "auto" | "cuda" | "mps" | "cpu"
        random_state: 全局随机种子，None 不固定
        verbose: 是否打印训练日志
    """

    def __init__(
        self,
        num_treatments: int,
        *,
        backbone: str = "mlp",
        head: str = "uplift",
        routing: str = "gps",
        hidden_dim: int = 200,
        head_hidden: int = 100,
        dropout: float = 0.0,
        use_mmd: bool = True,
        mmd_weight: float = 1.0,
        mmd_gammas: Sequence[float] = (0.1, 1.0, 10.0),
        use_tarreg: bool = True,
        tarreg_ratio: float = 1.0,
        gps_weight: float = 1.0,
        reg_l2: float = 0.01,
        epsilon_clip: float = 0.1,
        epochs: int = 300,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        early_stopping_patience: int | None = 30,
        val_split: float = 0.2,
        device: str | torch.device = "auto",
        random_state: int | None = 42,
        verbose: bool = True,
        # Transformer 专用
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
    ) -> None:
        if num_treatments < 2:
            raise ValueError(f"num_treatments 至少为 2，得 {num_treatments}")
        if not 0.0 <= val_split < 1.0:
            raise ValueError(f"val_split 应在 [0, 1)，得 {val_split}")

        self.config = _GACRNetConfig(
            num_treatments=num_treatments,
            backbone=backbone,
            head=head,
            routing=routing,
            hidden_dim=hidden_dim,
            head_hidden=head_hidden,
            dropout=dropout,
            use_mmd=use_mmd,
            mmd_weight=mmd_weight,
            mmd_gammas=tuple(mmd_gammas),
            use_tarreg=use_tarreg,
            tarreg_ratio=tarreg_ratio,
            gps_weight=gps_weight,
            reg_l2=reg_l2,
            epsilon_clip=epsilon_clip,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            early_stopping_patience=early_stopping_patience,
            val_split=val_split,
            random_state=random_state,
            verbose=verbose,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
        )
        self.device = resolve_device(device)
        self._logger = get_logger("causalmt.gacrnet")
        self.module: GACRNetModule | None = None
        self.loss_fn: LossFunction | None = None
        self.history_: dict[str, Any] = {}

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        x,  # type: ignore[no-untyped-def]
        t,  # type: ignore[no-untyped-def]
        y,  # type: ignore[no-untyped-def]
        *,
        val_data: tuple | None = None,
    ) -> GACRNet:
        """训练 GACRNet。

        Args:
            x: shape (n, d) 协变量
            t: shape (n,) 干预索引，0..K-1
            y: shape (n,) 观测结果
            val_data: (x_val, t_val, y_val)，None 时按 val_split 从训练集切分
        """
        cfg = self.config
        if cfg.random_state is not None:
            set_seed(cfg.random_state)

        x_t = to_tensor(x, dtype=torch.float32)
        t_t = to_tensor(t, dtype=torch.long)
        y_t = to_tensor(y, dtype=torch.float32).reshape(-1)

        if x_t.dim() != 2:
            raise ValueError(f"x 应为 2D，得 shape {tuple(x_t.shape)}")
        if t_t.shape[0] != x_t.shape[0] or y_t.shape[0] != x_t.shape[0]:
            raise ValueError("x/t/y 样本数不一致")

        input_dim = x_t.shape[1]
        self.config.input_dim = input_dim

        # 切分验证集
        if val_data is not None:
            xv, tv, yv = val_data
            x_val = to_tensor(xv, dtype=torch.float32)
            t_val = to_tensor(tv, dtype=torch.long)
            y_val = to_tensor(yv, dtype=torch.float32).reshape(-1)
        elif cfg.val_split > 0:
            n = x_t.shape[0]
            n_val = max(1, int(n * cfg.val_split))
            perm = torch.randperm(n)
            val_idx, tr_idx = perm[:n_val], perm[n_val:]
            x_val, t_val, y_val = x_t[val_idx], t_t[val_idx], y_t[val_idx]
            x_t, t_t, y_t = x_t[tr_idx], t_t[tr_idx], y_t[tr_idx]
        else:
            x_val = t_val = y_val = None  # type: ignore[assignment]

        self._build_module(input_dim)
        assert self.module is not None and self.loss_fn is not None

        def _step(model: GACRNetModule, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
            xb, tb, yb = batch
            concat_true = torch.stack([yb, tb.float()], dim=1)
            assert self.loss_fn is not None
            pred, shared = model.forward_with_features(xb)
            return self.loss_fn(concat_true, pred, shared) + model.get_l2_regularization()

        train_cfg = TrainConfig(
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            early_stopping_patience=cfg.early_stopping_patience,
            verbose=cfg.verbose,
        )
        trainer = Trainer(self.module, _step, train_cfg, self.device)
        val_tensors = (x_val, t_val, y_val) if x_val is not None else None
        history = trainer.fit((x_t, t_t, y_t), val_tensors)

        self.history_ = {
            "train_loss": history.train_loss,
            "val_loss": history.val_loss,
            "best_epoch": history.best_epoch,
            "best_val_loss": history.best_val_loss,
        }
        return self

    # ------------------------------------------------------------------ 推断
    @torch.no_grad()
    def predict_potential_outcomes(self, x) -> np.ndarray:  # type: ignore[no-untyped-def]
        """预测每个样本在 K 种干预下的潜在结果，shape (n, K)。"""
        self._check_fitted()
        assert self.module is not None
        self.module.eval()
        x_t = to_tensor(x, dtype=torch.float32, device=self.device)
        outcomes = self.module.predict_outcomes(x_t)
        return to_numpy(outcomes)

    @torch.no_grad()
    def predict_gps(self, x) -> np.ndarray:  # type: ignore[no-untyped-def]
        """预测广义倾向得分（softmax 概率），shape (n, K)。"""
        self._check_fitted()
        assert self.module is not None
        self.module.eval()
        x_t = to_tensor(x, dtype=torch.float32, device=self.device)
        return to_numpy(self.module.predict_gps(x_t))

    def estimate_cate(
        self, x, *, treatment_a: int = 1, treatment_b: int = 0  # type: ignore[no-untyped-def]
    ) -> np.ndarray:
        """估计个体处理效应 τ(x) = E[Y(a) - Y(b) | X=x]。

        Args:
            x: 协变量
            treatment_a: 处理组干预索引（默认 1）
            treatment_b: 对照组干预索引（默认 0）

        Returns:
            shape (n,) CATE 估计
        """
        self._check_treatment_idx(treatment_a)
        self._check_treatment_idx(treatment_b)
        po = self.predict_potential_outcomes(x)
        return po[:, treatment_a] - po[:, treatment_b]

    def estimate_ate(
        self, x, *, treatment_a: int = 1, treatment_b: int = 0  # type: ignore[no-untyped-def]
    ) -> float:
        """估计平均处理效应 ATE = E[Y(a) - Y(b)]。"""
        return float(
            np.mean(self.estimate_cate(x, treatment_a=treatment_a, treatment_b=treatment_b))
        )

    # ------------------------------------------------------------------ I/O
    def save(self, path: str | Path) -> None:
        """保存模型权重和超参到文件。"""
        self._check_fitted()
        assert self.module is not None
        path = Path(path)
        payload = {
            "config": asdict(self.config),
            "state_dict": self.module.state_dict(),
            "history": self.history_,
        }
        torch.save(payload, path)
        if self.config.verbose:
            self._logger.info("saved GACRNet to %s", path)

    @classmethod
    def load(cls, path: str | Path, *, device: str | torch.device = "auto") -> GACRNet:
        """从文件加载。"""
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        cfg_dict = payload["config"]
        input_dim = cfg_dict.pop("input_dim")
        if input_dim is None:
            raise ValueError("加载的模型未训练（input_dim 缺失）")
        est = cls(device=device, **_drop_internal_keys(cfg_dict))
        est.config.input_dim = input_dim
        est._build_module(input_dim)
        assert est.module is not None
        est.module.load_state_dict(payload["state_dict"])
        est.module.to(est.device)
        est.history_ = payload.get("history", {})
        return est

    # ------------------------------------------------------------------ 内部
    def _build_module(self, input_dim: int) -> None:
        cfg = self.config
        self.module = GACRNetModule(
            input_dim=input_dim,
            num_treatments=cfg.num_treatments,
            backbone_type=cfg.backbone,
            head_type=cfg.head,
            routing_type=cfg.routing if cfg.routing != "none" else "none",
            hidden_dim=cfg.hidden_dim,
            head_hidden=cfg.head_hidden,
            dropout=cfg.dropout,
            epsilon_clip=cfg.epsilon_clip,
            reg_l2=cfg.reg_l2,
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=cfg.num_layers,
            dim_feedforward=cfg.dim_feedforward,
        )
        self.loss_fn = LossFunction(
            num_treatments=cfg.num_treatments,
            feature_dim=cfg.hidden_dim,
            gps_weight=cfg.gps_weight,
            mmd_weight=cfg.mmd_weight if cfg.use_mmd else 0.0,
            mmd_gammas=cfg.mmd_gammas,
            use_tarreg=cfg.use_tarreg,
            tarreg_ratio=cfg.tarreg_ratio,
        )
        self.module.to(self.device)
        self.loss_fn.to(self.device)

    def _check_fitted(self) -> None:
        if self.module is None:
            raise RuntimeError("GACRNet 尚未训练，请先调用 fit(x, t, y)")

    def _check_treatment_idx(self, k: int) -> None:
        if not 0 <= k < self.config.num_treatments:
            raise ValueError(f"treatment 索引 {k} 越界，应在 [0, {self.config.num_treatments - 1}]")


def _drop_internal_keys(cfg: dict[str, Any]) -> dict[str, Any]:
    """过滤掉 _GACRNetConfig 中不属于 __init__ 参数的字段。"""
    return {k: v for k, v in cfg.items() if k != "input_dim"}
