"""多尺度 MMD（Maximum Mean Discrepancy）正则。

通过最小化不同干预组共享特征分布间的 MMD，促进表示均衡，
缓解多值干预下的选择偏差。

实现要点：
    - 使用 one-hot 分配矩阵将组间核和向量化为 (K, K) 矩阵运算，避免 K^2 次循环
    - 多尺度高斯核 (gamma 列表) 对不同尺度的分布差异都敏感
    - MPS 设备 fallback：cdist 反向传播在 MPS 上不可用时切换到 CPU
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["gaussian_kernel", "mmd_loss", "MMDLoss"]


def gaussian_kernel(x: torch.Tensor, y: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """高斯 RBF 核矩阵 K(x, y) = exp(-gamma * ||x - y||²)。

    在 MPS 设备上 `torch.cdist` 反向传播不支持，自动切换到 CPU 计算后转回。
    """
    device = x.device
    if device.type == "mps":
        x_cpu = x.cpu()
        y_cpu = y.cpu()
        dist_sq = torch.cdist(x_cpu, y_cpu, p=2) ** 2
        dist_sq = dist_sq.to(device)
    else:
        dist_sq = torch.cdist(x, y, p=2) ** 2
    return torch.exp(-gamma * dist_sq)


def mmd_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    gammas: Sequence[float] = (0.1, 1.0, 10.0),
) -> torch.Tensor:
    """计算两组样本的多尺度 MMD²（无偏估计的简化版）。"""
    total = torch.zeros((), device=x.device)
    for gamma in gammas:
        k_xx = gaussian_kernel(x, x, gamma).mean()
        k_yy = gaussian_kernel(y, y, gamma).mean()
        k_xy = gaussian_kernel(x, y, gamma).mean()
        total = total + (k_xx + k_yy - 2 * k_xy)
    return total / len(gammas)


class MMDLoss(nn.Module):
    """多组 MMD 正则模块（向量化实现）。

    构建 (K, K) 组间核和矩阵，一次性计算所有组对的 MMD²。

    Args:
        num_treatments: 干预类别数 K
        feature_dim: 共享特征维度（用于 LayerNorm）
        gammas: 多尺度高斯核 gamma 列表
    """

    def __init__(
        self,
        num_treatments: int,
        feature_dim: int,
        gammas: Sequence[float] = (0.1, 1.0, 10.0),
    ) -> None:
        super().__init__()
        self.num_treatments = num_treatments
        self.gammas = tuple(gammas)
        self.layer_norm = nn.LayerNorm(feature_dim)

    def forward(self, shared_features: torch.Tensor, treatments: torch.Tensor) -> torch.Tensor:
        features = self.layer_norm(shared_features)
        device = features.device
        k = self.num_treatments

        # one-hot 分配矩阵 A ∈ R^{B×K}
        a = F.one_hot(treatments, num_classes=k).float()
        n_per_group = a.sum(dim=0)

        # 平方距离矩阵 (B, B)
        x_norm = (features**2).sum(dim=1, keepdim=True)
        dist_sq = x_norm + x_norm.T - 2 * torch.matmul(features, features.T)
        dist_sq = torch.clamp(dist_sq, min=0.0)

        # 有效组对：每组 n_i > 1 才能计算无偏 within
        valid_mask = (n_per_group > 1).unsqueeze(1) & (n_per_group > 1).unsqueeze(0)
        triu_mask = torch.triu(torch.ones(k, k, device=device, dtype=torch.bool), diagonal=1)
        pair_mask = valid_mask & triu_mask

        if not pair_mask.any():
            return torch.zeros((), device=device)

        total_mmd = torch.zeros((), device=device)
        for gamma in self.gammas:
            kmat = torch.exp(-gamma * dist_sq)
            # 组对核和 M[i,j] = Σ_{n∈i, m∈j} K[n,m]
            m = a.T @ kmat @ a
            # 组内核和（去对角线 K[n,n]=1）
            within_sum = m.diagonal() - n_per_group

            within_denom = n_per_group * (n_per_group - 1) + 1e-8
            between_denom = n_per_group.unsqueeze(1) * n_per_group.unsqueeze(0) + 1e-8

            within_mean = within_sum / within_denom
            between_mean = m / between_denom

            mmd_matrix = within_mean.unsqueeze(1) + within_mean.unsqueeze(0) - 2 * between_mean
            mmd_matrix = torch.clamp(mmd_matrix, min=0.0)

            mmd_sum = (mmd_matrix * pair_mask).sum()
            pair_count = pair_mask.sum()
            total_mmd = total_mmd + mmd_sum / pair_count

        return total_mmd / len(self.gammas)
