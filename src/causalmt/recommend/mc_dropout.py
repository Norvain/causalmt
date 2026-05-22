"""Monte Carlo Dropout 推断：用 dropout 估计预测不确定性。

在推断时保持 dropout 为开启状态（与 eval 模式不同），多次前向取均值和方差，
作为预测分布的近似。需要模型在结构里包含 nn.Dropout 层。
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import numpy as np
import torch
import torch.nn as nn

__all__ = ["mc_dropout_predict", "enable_dropout"]


@contextlib.contextmanager
def enable_dropout(model: nn.Module) -> Iterator[nn.Module]:
    """暂时启用所有 Dropout 层（即使在 eval 模式）。退出时恢复原状态。"""
    was_training: dict[int, bool] = {}
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            was_training[id(module)] = module.training
            module.train(True)
    try:
        yield model
    finally:
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.train(was_training.get(id(module), False))


@torch.no_grad()
def mc_dropout_predict(
    model: nn.Module,
    x: torch.Tensor,
    *,
    n_samples: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """启用 dropout 的 MC 多次前向，返回均值和标准差。

    Args:
        model: 含 nn.Dropout 层的 torch 模型（dropout > 0）
        x: 输入张量
        n_samples: MC 采样次数

    Returns:
        mean: (n, ...) 预测均值（numpy）
        std: (n, ...) 预测标准差（numpy）
    """
    model.eval()
    samples: list[torch.Tensor] = []
    with enable_dropout(model):
        for _ in range(n_samples):
            out = model(x)
            samples.append(out.detach().cpu())
    stacked = torch.stack(samples, dim=0)  # (n_samples, batch, ...)
    return stacked.mean(dim=0).numpy(), stacked.std(dim=0).numpy()
