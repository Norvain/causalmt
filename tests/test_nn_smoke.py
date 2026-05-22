"""nn/ + losses/ 端到端冒烟测试。

只验证：
    - 各 backbone × routing × head 组合能构造 + 前向
    - 输出形状正确
    - LossFunction 在不同正则开关下能跑通反向传播
"""

from __future__ import annotations

import pytest
import torch

from causalmt.losses import LossFunction
from causalmt.nn import GACRNetModule, InteractionMLP

BATCH = 32
INPUT_DIM = 8
HIDDEN_DIM = 64
K_VALUES = [2, 3, 4]


@pytest.fixture
def fake_batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {
        "x": torch.randn(BATCH, INPUT_DIM),
        "y": torch.randn(BATCH),
    }


@pytest.mark.unit
@pytest.mark.parametrize("num_treatments", K_VALUES)
@pytest.mark.parametrize("backbone", ["mlp", "transformer"])
@pytest.mark.parametrize("routing", ["none", "gps", "self", "concat"])
@pytest.mark.parametrize("head", ["uplift", "separate"])
def test_gacrnet_forward_shapes(
    fake_batch: dict[str, torch.Tensor],
    num_treatments: int,
    backbone: str,
    routing: str,
    head: str,
) -> None:
    """所有 backbone × routing × head × K 组合的前向形状都正确。"""
    model = GACRNetModule(
        input_dim=INPUT_DIM,
        num_treatments=num_treatments,
        backbone_type=backbone,
        head_type=head,
        routing_type=routing,
        hidden_dim=HIDDEN_DIM,
        head_hidden=32,
    )
    concat_pred, shared = model.forward_with_features(fake_batch["x"])
    assert concat_pred.shape == (BATCH, 3 * num_treatments)
    assert shared.shape == (BATCH, HIDDEN_DIM)

    outcomes = model.predict_outcomes(fake_batch["x"])
    assert outcomes.shape == (BATCH, num_treatments)

    gps = model.predict_gps(fake_batch["x"])
    assert gps.shape == (BATCH, num_treatments)
    assert torch.allclose(gps.sum(dim=1), torch.ones(BATCH), atol=1e-5)


@pytest.mark.unit
@pytest.mark.parametrize(
    "mmd_weight,use_tarreg", [(0.0, False), (1.0, False), (0.0, True), (1.0, True)]
)
def test_loss_function_backward(
    fake_batch: dict[str, torch.Tensor], mmd_weight: float, use_tarreg: bool
) -> None:
    """4 种正则组合下损失都能反向传播。"""
    k = 3
    model = GACRNetModule(
        input_dim=INPUT_DIM,
        num_treatments=k,
        hidden_dim=HIDDEN_DIM,
        head_hidden=32,
    )
    loss_fn = LossFunction(
        num_treatments=k,
        feature_dim=HIDDEN_DIM,
        mmd_weight=mmd_weight,
        use_tarreg=use_tarreg,
    )

    t = torch.randint(0, k, (BATCH,))
    concat_true = torch.stack([fake_batch["y"], t.float()], dim=1)

    concat_pred, shared = model.forward_with_features(fake_batch["x"])
    loss = loss_fn(concat_true, concat_pred, shared) + model.get_l2_regularization()

    assert loss.dim() == 0
    assert torch.isfinite(loss)
    loss.backward()
    # 至少 backbone 第一层应有梯度
    assert model.backbone.net[0].weight.grad is not None


@pytest.mark.unit
def test_interaction_mlp() -> None:
    """InteractionMLP 前向 + 反向。"""
    net = InteractionMLP(input_dim=10, hidden_dims=(32, 16), dropout=0.1)
    x = torch.randn(BATCH, 10)
    y = torch.randn(BATCH, 1)
    pred = net(x)
    assert pred.shape == (BATCH, 1)
    loss = torch.nn.functional.mse_loss(pred, y)
    loss.backward()
    assert net.net[0].weight.grad is not None
