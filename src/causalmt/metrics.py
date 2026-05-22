"""因果推断常用评估指标。

PEHE：Precision in Estimation of Heterogeneous Effect，sqrt(MSE(CATE_pred, CATE_true))。
ATE error：|mean(CATE_pred) - mean(CATE_true)| 或对真值 ATE 的绝对误差。
Policy risk：策略 π(x) = argmax_k τ̂_k(x) 下的反事实期望损失（Jobs 风格）。
"""

from __future__ import annotations

import numpy as np

from causalmt.utils import to_numpy

__all__ = ["pehe", "ate_error", "policy_risk"]


def pehe(cate_pred, cate_true) -> float:  # type: ignore[no-untyped-def]
    """sqrt(MSE(CATE_pred, CATE_true))。"""
    p = to_numpy(cate_pred).reshape(-1)
    t = to_numpy(cate_true).reshape(-1)
    return float(np.sqrt(np.mean((p - t) ** 2)))


def ate_error(cate_pred, cate_true=None, ate_true: float | None = None) -> float:  # type: ignore[no-untyped-def]
    """ATE 绝对误差。

    `cate_true` 与 `ate_true` 至少提供一个：
        - 给 cate_true：用其均值作为 ATE_true
        - 给 ate_true：直接用标量
    """
    if cate_true is None and ate_true is None:
        raise ValueError("必须提供 cate_true 或 ate_true 之一")
    p = to_numpy(cate_pred).reshape(-1)
    true = float(ate_true) if ate_true is not None else float(np.mean(to_numpy(cate_true)))
    return float(abs(np.mean(p) - true))


def policy_risk(
    potential_outcomes_pred,  # type: ignore[no-untyped-def]
    potential_outcomes_true,  # type: ignore[no-untyped-def]
    *,
    objective: str = "max",
) -> float:
    """策略风险：相对最优策略的期望结果损失。

    π(x) = argmax_k ŷ_k(x)（objective="max"，最大化好结果）
    或 argmin_k（objective="min"，最小化损失）。

    Args:
        potential_outcomes_pred: shape (n, K) 预测潜在结果
        potential_outcomes_true: shape (n, K) 真实潜在结果
        objective: "max" | "min"

    Returns:
        E[y(π_pred)] 与 E[y(π_opt)] 的差值（绝对值）
    """
    pred = to_numpy(potential_outcomes_pred)
    true = to_numpy(potential_outcomes_true)
    if pred.ndim != 2 or true.ndim != 2 or pred.shape != true.shape:
        raise ValueError(f"输入形状应为 (n, K) 且一致，得 {pred.shape} vs {true.shape}")

    if objective == "max":
        actions_pred = pred.argmax(axis=1)
        actions_opt = true.argmax(axis=1)
    elif objective == "min":
        actions_pred = pred.argmin(axis=1)
        actions_opt = true.argmin(axis=1)
    else:
        raise ValueError(f"objective 应为 'max' 或 'min'，得 {objective!r}")

    n = len(true)
    y_pred_policy = true[np.arange(n), actions_pred].mean()
    y_opt_policy = true[np.arange(n), actions_opt].mean()
    return float(abs(y_opt_policy - y_pred_policy))
