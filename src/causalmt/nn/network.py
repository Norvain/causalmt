"""GACR 主网络模块。

将 backbone + GPS head + routing + outcome heads + epsilon layer 组装为
端到端因果推断网络。前向输出 (outcomes, gps_logits, epsilons) 拼接张量
和共享特征（供 MMD 正则使用）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from causalmt.nn.backbones import MLPBackbone, TransformerBackbone
from causalmt.nn.heads import EpsilonLayer, build_outcome_heads
from causalmt.nn.routing import build_routing

__all__ = ["GACRNetModule"]


class GACRNetModule(nn.Module):
    """GACR 主网络（底层 nn.Module，供 GACRNet Estimator 使用）。

    用户通常不直接实例化此类，而是通过顶层 `causalmt.GACRNet` 接口。

    前向输出 `concat_pred` 形状为 `(B, 3*K)`，三段：
        - `[:, 0:K]`：K 个潜在结果预测
        - `[:, K:2K]`：GPS logits
        - `[:, 2K:3K]`：epsilon 扰动值（已 clip）
    """

    def __init__(
        self,
        input_dim: int,
        num_treatments: int,
        *,
        backbone_type: str = "mlp",
        head_type: str = "uplift",
        routing_type: str = "gps",
        hidden_dim: int = 200,
        head_hidden: int = 100,
        dropout: float = 0.0,
        epsilon_clip: float = 0.1,
        reg_l2: float = 0.01,
        # Transformer-only 参数
        d_model: int = 200,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 200,
    ) -> None:
        super().__init__()
        self.num_treatments = num_treatments
        self.reg_l2 = reg_l2
        self.epsilon_clip = epsilon_clip
        self.head_type = head_type

        if backbone_type == "mlp":
            self.backbone: nn.Module = MLPBackbone(input_dim, hidden_dim, dropout)
        elif backbone_type == "transformer":
            self.backbone = TransformerBackbone(
                input_dim, hidden_dim, d_model, nhead, num_layers, dim_feedforward, dropout
            )
        else:
            raise ValueError(f"未知 backbone_type={backbone_type!r}，可选: 'mlp' | 'transformer'")

        feat_dim = getattr(self.backbone, "output_dim", hidden_dim)
        self.gps_head = nn.Linear(feat_dim, num_treatments)
        self.routing = build_routing(routing_type, feat_dim, num_treatments)
        self.outcome_heads = build_outcome_heads(
            head_type, feat_dim, num_treatments, head_hidden, dropout
        )
        self.epsilon_layer = EpsilonLayer(num_treatments)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """仅返回拼接预测，便于推断时使用。"""
        concat_pred, _ = self.forward_with_features(x)
        return concat_pred

    def forward_with_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """同时返回拼接预测和共享特征（训练时计算 MMD 用）。"""
        shared_features = self.backbone(x)
        gps_logits = self.gps_head(shared_features)

        features_list: list[torch.Tensor] | torch.Tensor
        if self.routing is not None:
            gps_probs = F.softmax(gps_logits, dim=1)
            routed_features = self.routing(shared_features, gps_probs)
            features_list = [shared_features] + routed_features
        else:
            features_list = shared_features

        outcomes = self.outcome_heads(features_list)
        epsilons = self.epsilon_layer(x.shape[0])
        epsilons = torch.clamp(epsilons, -self.epsilon_clip, self.epsilon_clip)
        concat_pred = torch.cat([outcomes, gps_logits, epsilons], dim=1)
        return concat_pred, shared_features

    def get_l2_regularization(self) -> torch.Tensor:
        """outcome heads + routing 的 L2 正则项。"""
        l2_reg: torch.Tensor = torch.zeros((), device=next(self.parameters()).device)
        for name, param in self.outcome_heads.named_parameters():
            if "weight" in name:
                l2_reg = l2_reg + torch.norm(param, 2)
        if self.routing is not None:
            for name, param in self.routing.named_parameters():
                if "weight" in name:
                    l2_reg = l2_reg + torch.norm(param, 2)
        return self.reg_l2 * l2_reg

    @torch.no_grad()
    def predict_outcomes(self, x: torch.Tensor) -> torch.Tensor:
        """预测 K 个潜在结果，shape (B, K)。"""
        concat_pred, _ = self.forward_with_features(x)
        return concat_pred[:, : self.num_treatments]

    @torch.no_grad()
    def predict_gps(self, x: torch.Tensor) -> torch.Tensor:
        """预测广义倾向得分（softmax 概率），shape (B, K)。"""
        shared_features = self.backbone(x)
        gps_logits = self.gps_head(shared_features)
        return F.softmax(gps_logits, dim=1)
