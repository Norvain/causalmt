"""结果预测头与 epsilon 层。

两种 head 设计：
    - SeparateHeads：每个干预独立一个 MLP，直接预测潜在结果 y_k
    - UpliftHeads：先预测 μ₀（控制组），其余干预预测相对 μ₀ 的 uplift Δ_k，
      最终 y_k = μ₀ + Δ_k（论文默认）

EpsilonLayer 为目标正则化提供可学习扰动参数 ε_k。
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["SeparateHeads", "UpliftHeads", "EpsilonLayer", "build_outcome_heads"]


def _mlp_head(input_dim: int, head_hidden: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, head_hidden),
        nn.ELU(),
        nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        nn.Linear(head_hidden, head_hidden),
        nn.ELU(),
        nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        nn.Linear(head_hidden, 1),
    )


class SeparateHeads(nn.Module):
    """每个干预独立 MLP，直接输出 y_k。"""

    def __init__(
        self,
        input_dim: int,
        num_treatments: int,
        head_hidden: int = 100,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.predictors = nn.ModuleList(
            [_mlp_head(input_dim, head_hidden, dropout) for _ in range(num_treatments)]
        )

    def forward(self, features_list: list[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        outputs = []
        for i, predictor in enumerate(self.predictors):
            feat = features_list[i] if isinstance(features_list, list) else features_list
            outputs.append(predictor(feat))
        return torch.cat(outputs, dim=1)


class UpliftHeads(nn.Module):
    """μ₀ + uplift 分解的预测头（论文默认）。

    第 0 个 head 预测控制组 μ₀，其余 K-1 个 head 预测相对 μ₀ 的 uplift。
    潜在结果 y_k = μ₀ + Δ_k。
    """

    def __init__(
        self,
        input_dim: int,
        num_treatments: int,
        head_hidden: int = 100,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.mu0_head = _mlp_head(input_dim, head_hidden, dropout)
        self.uplift_heads = nn.ModuleList(
            [_mlp_head(input_dim, head_hidden, dropout) for _ in range(num_treatments - 1)]
        )

    def forward(self, features_list: list[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        base_feat = features_list[0] if isinstance(features_list, list) else features_list
        mu0 = self.mu0_head(base_feat)
        outputs = [mu0]
        for i, head in enumerate(self.uplift_heads):
            feat = features_list[i + 1] if isinstance(features_list, list) else features_list
            uplift = head(feat)
            outputs.append(mu0 + uplift)
        return torch.cat(outputs, dim=1)


class EpsilonLayer(nn.Module):
    """可学习的每干预扰动参数 ε_k，用于目标正则化。"""

    def __init__(self, num_treatments: int) -> None:
        super().__init__()
        self.epsilon = nn.Parameter(torch.randn(num_treatments, 1))

    def forward(self, batch_size: int) -> torch.Tensor:
        return self.epsilon.T.expand(batch_size, -1)


def build_outcome_heads(
    head_type: str,
    input_dim: int,
    num_treatments: int,
    head_hidden: int = 100,
    dropout: float = 0.0,
) -> nn.Module:
    """根据字符串名构造结果预测头。

    Args:
        head_type: "uplift"（默认）| "separate"
        input_dim: 输入特征维度
        num_treatments: 干预类别数 K
        head_hidden: head 隐藏层维度
        dropout: dropout 概率
    """
    if head_type == "uplift":
        return UpliftHeads(input_dim, num_treatments, head_hidden, dropout)
    if head_type == "separate":
        return SeparateHeads(input_dim, num_treatments, head_hidden, dropout)
    raise ValueError(f"未知 head_type={head_type!r}，可选: 'uplift' | 'separate'")
