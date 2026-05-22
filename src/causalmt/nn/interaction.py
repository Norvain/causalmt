"""组合干预交互效应 MLP（IDCIO 第二阶段使用）。

输入为原子干预效应预测值 [τ̂_1, τ̂_2, ..., τ̂_K] 与协变量特征拼接，
输出该组合下的交互项 τ_{ij...}。
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

__all__ = ["InteractionMLP"]


class InteractionMLP(nn.Module):
    """轻量 MLP 拟合组合干预交互效应。

    Args:
        input_dim: 输入维度（通常为协变量维度 + 原子效应数）
        hidden_dims: 隐藏层宽度序列
        dropout: dropout 概率（>0 时启用 MC Dropout 推断）
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (100, 50),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
