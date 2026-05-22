"""组合损失函数：outcome + GPS + (可选) MMD + (可选) tarreg。

`LossFunction.forward(concat_true, concat_pred, shared_features=None)` 是训练循环
唯一调用入口。`concat_true` 形如 (B, 2)，列为 [y, t]；`concat_pred` 形如 (B, 3K)，
列分为 [outcomes(K), gps_logits(K), epsilons(K)]。
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from causalmt.losses.gps import gps_classification_loss
from causalmt.losses.mmd import MMDLoss
from causalmt.losses.outcome import multi_treatment_outcome_loss
from causalmt.losses.tarreg import tarreg_loss

__all__ = ["LossFunction"]


class LossFunction(nn.Module):
    """组合损失函数。

    所有正则项都通过权重开关控制：mmd_weight=0 跳过 MMD，use_tarreg=False 跳过 tarreg。
    """

    def __init__(
        self,
        num_treatments: int,
        feature_dim: int,
        *,
        gps_weight: float = 1.0,
        mmd_weight: float = 0.0,
        mmd_gammas: Sequence[float] = (0.1, 1.0, 10.0),
        use_tarreg: bool = False,
        tarreg_ratio: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_treatments = num_treatments
        self.gps_weight = gps_weight
        self.mmd_weight = mmd_weight
        self.use_tarreg = use_tarreg
        self.tarreg_ratio = tarreg_ratio

        self.mmd_module: MMDLoss | None
        if mmd_weight > 0.0:
            self.mmd_module = MMDLoss(num_treatments, feature_dim, mmd_gammas)
        else:
            self.mmd_module = None

    def forward(
        self,
        concat_true: torch.Tensor,
        concat_pred: torch.Tensor,
        shared_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        y_true = concat_true[:, 0]
        t_true = concat_true[:, 1].long()
        k = self.num_treatments

        outcome_preds = concat_pred[:, :k]
        gps_logits = concat_pred[:, k : 2 * k]
        epsilons = concat_pred[:, 2 * k : 3 * k]

        # 主损失
        loss = multi_treatment_outcome_loss(y_true, outcome_preds, t_true)
        loss = loss + self.gps_weight * gps_classification_loss(gps_logits, t_true)

        # MMD 正则
        if self.mmd_module is not None and shared_features is not None:
            loss = loss + self.mmd_weight * self.mmd_module(shared_features, t_true)

        # 目标正则化
        if self.use_tarreg:
            y_pred = torch.gather(outcome_preds, 1, t_true.unsqueeze(1)).squeeze(1)
            loss = loss + self.tarreg_ratio * tarreg_loss(
                y_true, y_pred, gps_logits, epsilons, t_true
            )

        return loss
