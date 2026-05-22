"""通用训练循环，含 early stopping + best ckpt + 历史记录。

用于 GACRNet 和 IDCIO 的 InteractionMLP 训练复用。
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from causalmt.utils import get_logger

__all__ = ["TrainConfig", "TrainHistory", "Trainer"]


@dataclass(frozen=True)
class TrainConfig:
    """训练超参数（不可变）。"""

    epochs: int = 300
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    early_stopping_patience: int | None = 30
    grad_clip: float | None = None
    verbose: bool = True
    log_every: int = 10


@dataclass
class TrainHistory:
    """训练历史记录。"""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_epoch: int = -1
    best_val_loss: float = float("inf")


# LossCallable: 接收一个 batch tuple，返回标量 loss
LossCallable = Callable[[nn.Module, tuple[torch.Tensor, ...]], torch.Tensor]


class Trainer:
    """通用训练器。

    Loss 计算逻辑由调用方通过 `loss_fn` 注入：
        loss_fn(model, batch) -> Tensor

    `batch` 是 DataLoader 产出的 tuple，结构由 `build_dataset` 决定。
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: LossCallable,
        config: TrainConfig,
        device: torch.device,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.config = config
        self.device = device
        self._logger = get_logger("causalmt.trainer")
        self.history = TrainHistory()

    @staticmethod
    def build_loader(
        tensors: tuple[torch.Tensor, ...],
        *,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        return DataLoader(
            TensorDataset(*tensors),
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,
        )

    def fit(
        self,
        train_tensors: tuple[torch.Tensor, ...],
        val_tensors: tuple[torch.Tensor, ...] | None = None,
    ) -> TrainHistory:
        """执行训练循环。

        Args:
            train_tensors: (x, t, y) 或类似元组，传入 TensorDataset
            val_tensors: 同 train_tensors，用于 early stopping；None 时禁用 ES

        Returns:
            TrainHistory（含 train/val loss 序列和 best epoch）
        """
        cfg = self.config
        self.model.to(self.device)
        optim = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        train_loader = self.build_loader(train_tensors, batch_size=cfg.batch_size, shuffle=True)

        best_state: dict[str, torch.Tensor] | None = None
        patience_counter = 0

        for epoch in range(1, cfg.epochs + 1):
            train_loss = self._train_one_epoch(train_loader, optim)
            self.history.train_loss.append(train_loss)

            if val_tensors is not None:
                val_loss = self._evaluate(val_tensors)
                self.history.val_loss.append(val_loss)

                improved = val_loss < self.history.best_val_loss
                if improved:
                    self.history.best_val_loss = val_loss
                    self.history.best_epoch = epoch
                    best_state = {
                        k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                    }
                    patience_counter = 0
                else:
                    patience_counter += 1

                if cfg.verbose and (epoch == 1 or epoch % cfg.log_every == 0):
                    self._logger.info(
                        "epoch %3d | train %.4f | val %.4f | best %.4f @ %d",
                        epoch,
                        train_loss,
                        val_loss,
                        self.history.best_val_loss,
                        self.history.best_epoch,
                    )

                if (
                    cfg.early_stopping_patience is not None
                    and patience_counter >= cfg.early_stopping_patience
                ):
                    if cfg.verbose:
                        self._logger.info(
                            "early stop @ epoch %d (best @ %d)", epoch, self.history.best_epoch
                        )
                    break
            else:
                if cfg.verbose and (epoch == 1 or epoch % cfg.log_every == 0):
                    self._logger.info("epoch %3d | train %.4f", epoch, train_loss)

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self.history

    def _train_one_epoch(self, loader: DataLoader, optim: torch.optim.Optimizer) -> float:
        self.model.train()
        total, n_batches = 0.0, 0
        for batch in loader:
            batch = tuple(t.to(self.device) for t in batch)
            optim.zero_grad()
            loss = self.loss_fn(self.model, batch)
            loss.backward()
            if self.config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            optim.step()
            total += float(loss.detach())
            n_batches += 1
        return total / max(n_batches, 1)

    @torch.no_grad()
    def _evaluate(self, tensors: tuple[torch.Tensor, ...]) -> float:
        self.model.eval()
        loader = self.build_loader(tensors, batch_size=self.config.batch_size, shuffle=False)
        total, n_batches = 0.0, 0
        for batch in loader:
            batch = tuple(t.to(self.device) for t in batch)
            loss = self.loss_fn(self.model, batch)
            total += float(loss.detach())
            n_batches += 1
        return total / max(n_batches, 1)


def clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """复制模型 state_dict 到 CPU（深拷贝）。"""
    return copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
