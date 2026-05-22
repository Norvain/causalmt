"""Treatment Routing Attention：根据 GPS / 自注意力为各干预 head 生成路由表示。

三种 routing 机制（论文消融研究）：
    - GPSQueryRouting：以 GPS 概率作为 query（论文默认）
    - FeatureSelfQueryRouting：以共享特征自身作为 query
    - ConcatQueryRouting：特征与 GPS 概率拼接后作为 query

每种 routing 接收共享特征 (B, D) 和 treatment 概率 (B, K)，
输出 K-1 个干预特定的特征张量（控制组 0 仍使用原始共享特征）。
"""

from __future__ import annotations

import math
from typing import Protocol

import torch
import torch.nn as nn

__all__ = [
    "Routing",
    "GPSQueryRouting",
    "FeatureSelfQueryRouting",
    "ConcatQueryRouting",
    "build_routing",
]


class Routing(Protocol):
    """所有 routing 模块共享的接口。"""

    num_eff_treatments: int

    def __call__(
        self, shared_features: torch.Tensor, treatment_probs: torch.Tensor
    ) -> list[torch.Tensor]:
        ...


class _BaseQKVRouting(nn.Module):
    """三种 routing 共享的核心逻辑：sigmoid 加权的 (Q, K, V) 残差融合。"""

    def __init__(self, feature_dim: int, num_treatments: int) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.num_eff_treatments = num_treatments - 1

    def _attend(
        self,
        shared_features: torch.Tensor,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> list[torch.Tensor]:
        attention_scores = torch.sum(query * keys, dim=2, keepdim=True) / math.sqrt(
            self.feature_dim
        )
        attention_weights = torch.sigmoid(attention_scores)
        phi_k_tensor = attention_weights * values + (
            1 - attention_weights
        ) * shared_features.unsqueeze(1)
        return list(torch.unbind(phi_k_tensor, dim=1))


class GPSQueryRouting(_BaseQKVRouting):
    """以 GPS 概率作为 query 的 routing（论文默认）。"""

    def __init__(self, feature_dim: int, num_treatments: int) -> None:
        super().__init__(feature_dim, num_treatments)
        if self.num_eff_treatments > 0:
            self.key_proj = nn.Linear(feature_dim, self.num_eff_treatments * feature_dim)
            self.value_proj = nn.Linear(feature_dim, self.num_eff_treatments * feature_dim)
        self.query_proj = nn.Linear(1, feature_dim)

    def forward(
        self, shared_features: torch.Tensor, treatment_probs: torch.Tensor
    ) -> list[torch.Tensor]:
        if self.num_eff_treatments == 0:
            return []
        batch_size = shared_features.shape[0]
        q_k = treatment_probs[:, 1:].unsqueeze(-1)
        query = self.query_proj(q_k)
        keys = self.key_proj(shared_features).view(
            batch_size, self.num_eff_treatments, self.feature_dim
        )
        values = self.value_proj(shared_features).view(
            batch_size, self.num_eff_treatments, self.feature_dim
        )
        return self._attend(shared_features, query, keys, values)


class FeatureSelfQueryRouting(_BaseQKVRouting):
    """以共享特征自身作为 query 的 routing。"""

    def __init__(self, feature_dim: int, num_treatments: int) -> None:
        super().__init__(feature_dim, num_treatments)
        if self.num_eff_treatments > 0:
            self.query_proj = nn.Linear(feature_dim, self.num_eff_treatments * feature_dim)
            self.key_proj = nn.Linear(feature_dim, self.num_eff_treatments * feature_dim)
            self.value_proj = nn.Linear(feature_dim, self.num_eff_treatments * feature_dim)

    def forward(
        self, shared_features: torch.Tensor, treatment_probs: torch.Tensor | None = None
    ) -> list[torch.Tensor]:
        if self.num_eff_treatments == 0:
            return []
        batch_size = shared_features.shape[0]
        query = self.query_proj(shared_features).view(
            batch_size, self.num_eff_treatments, self.feature_dim
        )
        keys = self.key_proj(shared_features).view(
            batch_size, self.num_eff_treatments, self.feature_dim
        )
        values = self.value_proj(shared_features).view(
            batch_size, self.num_eff_treatments, self.feature_dim
        )
        return self._attend(shared_features, query, keys, values)


class ConcatQueryRouting(_BaseQKVRouting):
    """特征与 GPS 概率拼接后作为 query 的 routing。"""

    def __init__(self, feature_dim: int, num_treatments: int) -> None:
        super().__init__(feature_dim, num_treatments)
        if self.num_eff_treatments > 0:
            self.query_proj = nn.Linear(feature_dim + 1, feature_dim)
            self.key_proj = nn.Linear(feature_dim, self.num_eff_treatments * feature_dim)
            self.value_proj = nn.Linear(feature_dim, self.num_eff_treatments * feature_dim)

    def forward(
        self, shared_features: torch.Tensor, treatment_probs: torch.Tensor
    ) -> list[torch.Tensor]:
        if self.num_eff_treatments == 0:
            return []
        batch_size = shared_features.shape[0]
        probs_k = treatment_probs[:, 1:].unsqueeze(-1)
        feat_expanded = shared_features.unsqueeze(1).expand(-1, self.num_eff_treatments, -1)
        concat_input = torch.cat([feat_expanded, probs_k], dim=-1)
        query = self.query_proj(concat_input)
        keys = self.key_proj(shared_features).view(
            batch_size, self.num_eff_treatments, self.feature_dim
        )
        values = self.value_proj(shared_features).view(
            batch_size, self.num_eff_treatments, self.feature_dim
        )
        return self._attend(shared_features, query, keys, values)


_ROUTING_REGISTRY: dict[str, type[nn.Module]] = {
    "gps": GPSQueryRouting,
    "self": FeatureSelfQueryRouting,
    "concat": ConcatQueryRouting,
}


def build_routing(routing_type: str, feature_dim: int, num_treatments: int) -> nn.Module | None:
    """根据字符串名构造 routing 模块。

    Args:
        routing_type: "gps" | "self" | "concat" | "none"
        feature_dim: 共享特征维度
        num_treatments: 干预类别数 K（含控制组）

    Returns:
        Routing 模块，或 None（routing_type="none" 或 K<2 时）
    """
    if routing_type == "none" or num_treatments < 2:
        return None
    if routing_type not in _ROUTING_REGISTRY:
        raise ValueError(f"未知 routing_type={routing_type!r}，可选: {list(_ROUTING_REGISTRY)} 或 'none'")
    return _ROUTING_REGISTRY[routing_type](feature_dim, num_treatments)
