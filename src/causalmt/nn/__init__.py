"""神经网络积木模块。

底层 nn.Module 组件，通常用户不直接使用，而是通过 `causalmt.GACRNet` 高层接口调用。
"""

from causalmt.nn.backbones import MLPBackbone, TransformerBackbone
from causalmt.nn.heads import EpsilonLayer, SeparateHeads, UpliftHeads, build_outcome_heads
from causalmt.nn.interaction import InteractionMLP
from causalmt.nn.network import GACRNetModule
from causalmt.nn.routing import (
    ConcatQueryRouting,
    FeatureSelfQueryRouting,
    GPSQueryRouting,
    build_routing,
)

__all__ = [
    "MLPBackbone",
    "TransformerBackbone",
    "GPSQueryRouting",
    "FeatureSelfQueryRouting",
    "ConcatQueryRouting",
    "build_routing",
    "SeparateHeads",
    "UpliftHeads",
    "EpsilonLayer",
    "build_outcome_heads",
    "InteractionMLP",
    "GACRNetModule",
]
