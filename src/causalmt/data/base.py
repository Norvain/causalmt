"""统一的因果推断数据容器。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["CausalDataset"]


@dataclass(frozen=True)
class CausalDataset:
    """因果推断数据容器（不可变）。

    Attributes:
        x: shape (n, d) 协变量
        t: shape (n,) 干预索引（0..K-1）。组合干预场景下可能是 (n, K_atomic) 的多热矩阵
        y: shape (n,) 观察到的（factual）结果
        y_potential: shape (n, K) 真实潜在结果（仅合成数据有）
        cate_true: shape (n,) 真实 CATE τ(x)，用于评估
        ate_true: 真实 ATE 标量，用于评估
        feature_names: 长度 d 的特征名列表
        name: 数据集名称（用于日志/标识）
    """

    x: np.ndarray
    t: np.ndarray
    y: np.ndarray
    y_potential: np.ndarray | None = None
    cate_true: np.ndarray | None = None
    ate_true: float | None = None
    feature_names: tuple[str, ...] | None = None
    name: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.x.shape[0])

    @property
    def d(self) -> int:
        return int(self.x.shape[1])

    @property
    def num_treatments(self) -> int:
        """从 t 推断 K（仅适用于一维 t）。"""
        if self.t.ndim == 1:
            return int(self.t.max()) + 1
        return int(self.t.shape[1])

    def __repr__(self) -> str:
        parts = [f"CausalDataset(name={self.name!r}, n={self.n}, d={self.d}"]
        parts.append(f"K={self.num_treatments}")
        if self.y_potential is not None:
            parts.append("has_potential=True")
        if self.cate_true is not None:
            parts.append("has_cate_true=True")
        return ", ".join(parts) + ")"
