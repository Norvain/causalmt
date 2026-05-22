"""广义倾向得分（GPS）分类损失。"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["gps_classification_loss"]


def gps_classification_loss(gps_logits: torch.Tensor, t_true: torch.Tensor) -> torch.Tensor:
    """K 类交叉熵监督 GPS 头。

    Args:
        gps_logits: shape (B, K) 未归一化 logits
        t_true: shape (B,) 真实干预索引（long）
    """
    return F.cross_entropy(gps_logits, t_true)
