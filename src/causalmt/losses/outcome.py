"""结果预测损失。

对每个样本，仅在其实际接受的干预对应的 head 上计算 MSE。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["multi_treatment_outcome_loss"]


def multi_treatment_outcome_loss(
    y_true: torch.Tensor,
    outcome_preds: torch.Tensor,
    t_true: torch.Tensor,
) -> torch.Tensor:
    """仅在观察到的干预 head 上计算 MSE。

    Args:
        y_true: shape (B,) 观察到的结果
        outcome_preds: shape (B, K) 所有 head 的潜在结果预测
        t_true: shape (B,) 观察到的干预索引（long）

    Returns:
        标量 MSE 损失
    """
    y_pred = torch.gather(outcome_preds, 1, t_true.unsqueeze(1)).squeeze(1)
    return F.mse_loss(y_true, y_pred)
