"""共享特征提取骨干网络。

提供两种 backbone：
    - MLPBackbone：3 层 ELU MLP（DragonNet 风格，默认）
    - TransformerBackbone：单 token Transformer 编码器（适合高维稀疏特征）
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["MLPBackbone", "TransformerBackbone"]


class MLPBackbone(nn.Module):
    """3 层 ELU MLP 共享表示。

    Args:
        input_dim: 输入特征维度
        hidden_dim: 隐藏层/输出维度
        dropout: dropout 概率，0 时禁用
    """

    def __init__(self, input_dim: int, hidden_dim: int = 200, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )
        self.output_dim: int = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBackbone(nn.Module):
    """单 token Transformer 编码器共享表示。

    将每个样本编码为一个 token，过 N 层 TransformerEncoder。

    Args:
        input_dim: 输入特征维度
        hidden_dim: 输出维度
        d_model: Transformer 内部维度
        nhead: 注意力头数
        num_layers: 编码器层数
        dim_feedforward: FFN 隐藏维度
        dropout: dropout 概率
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 200,
        d_model: int = 200,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 200,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, hidden_dim)
        self.output_dim: int = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x).unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)
        return self.output_layer(x)
