"""目标正则化（Targeted Regularization）。

基于 Shi et al. (2019) DragonNet 的目标正则化思想，扩展到多值干预：
    y_pert = y_pred + ε_t * h(t, x),  h(t, x) = 1 / GPS(t | x)

通过最小化 MSE(y_true, y_pert)，鼓励模型在 GPS 加权扰动下仍保持预测精度，
从而获得对 ATE 的双重稳健（doubly robust）估计。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["tarreg_loss"]


def tarreg_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    gps_logits: torch.Tensor,
    epsilons: torch.Tensor,
    t_true: torch.Tensor,
) -> torch.Tensor:
    """多值干预的目标正则化损失。

    Args:
        y_true: shape (B,) 观察结果
        y_pred: shape (B,) 已 gather 出的真实干预下的预测结果
        gps_logits: shape (B, K) GPS logits
        epsilons: shape (B, K) 已 clip 的 epsilon 扰动
        t_true: shape (B,) 真实干预索引（long）

    Returns:
        标量 MSE(y_true, y_pert) 损失
    """
    gps_probs = (F.softmax(gps_logits, dim=1) + 0.01) / 1.02
    gps_t = torch.gather(gps_probs, 1, t_true.unsqueeze(1)).squeeze(1)
    h_terms = 1.0 / gps_t
    h_terms = h_terms * h_terms.shape[0] / (h_terms.sum() + 1e-8)
    eps_t = torch.gather(epsilons, 1, t_true.unsqueeze(1)).squeeze(1)
    y_pert = y_pred + eps_t * h_terms
    return F.mse_loss(y_true, y_pert)
