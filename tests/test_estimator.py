"""GACRNet end-to-end 集成测试。

使用合成数据：协变量 → treatment（多类）→ 异质效应 → 观测结果。
验证 fit / predict / estimate / save / load 全链路。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from causalmt import GACRNet
from causalmt.metrics import ate_error, pehe


def _make_toy_data(n: int = 400, d: int = 6, k: int = 3, seed: int = 0):
    """生成一个简单的多值干预合成数据集。

    返回:
        x, t, y_obs, potential_outcomes (n, K), cate_true (n,) for τ=Y1-Y0
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d)).astype(np.float32)
    # 真实潜在结果：每个干预下加不同的非线性函数
    coefs = rng.normal(scale=0.5, size=(k, d)).astype(np.float32)
    potential = np.stack(
        [x @ coefs[i] + 0.5 * np.tanh(x[:, 0]) * (i + 1) for i in range(k)], axis=1
    )
    # GPS：和第一个特征相关
    logits = np.stack([x[:, 0] * (i - 1) * 0.5 for i in range(k)], axis=1)
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    t = np.array([rng.choice(k, p=probs[i]) for i in range(n)], dtype=np.int64)
    y_obs = potential[np.arange(n), t] + rng.normal(scale=0.1, size=n).astype(np.float32)
    cate_true = potential[:, 1] - potential[:, 0]
    return x, t, y_obs.astype(np.float32), potential.astype(np.float32), cate_true


@pytest.fixture(scope="module")
def toy() -> dict:
    x, t, y, po, cate = _make_toy_data(n=400, d=6, k=3, seed=0)
    # 切分 train/test
    n_test = 100
    return {
        "x_train": x[:-n_test],
        "t_train": t[:-n_test],
        "y_train": y[:-n_test],
        "x_test": x[-n_test:],
        "t_test": t[-n_test:],
        "y_test": y[-n_test:],
        "po_test": po[-n_test:],
        "cate_test": cate[-n_test:],
    }


@pytest.mark.integration
def test_fit_predict_estimate(toy: dict) -> None:
    """fit + 各推断接口都返回正确形状，PEHE 应优于随机猜测。"""
    est = GACRNet(
        num_treatments=3,
        hidden_dim=32,
        head_hidden=16,
        epochs=80,
        batch_size=64,
        lr=5e-3,
        use_mmd=True,
        use_tarreg=True,
        early_stopping_patience=20,
        verbose=False,
        random_state=42,
    )
    est.fit(toy["x_train"], toy["t_train"], toy["y_train"])

    po = est.predict_potential_outcomes(toy["x_test"])
    assert po.shape == (100, 3)

    gps = est.predict_gps(toy["x_test"])
    assert gps.shape == (100, 3)
    np.testing.assert_allclose(gps.sum(axis=1), 1.0, atol=1e-4)

    cate = est.estimate_cate(toy["x_test"], treatment_a=1, treatment_b=0)
    assert cate.shape == (100,)

    ate = est.estimate_ate(toy["x_test"], treatment_a=1, treatment_b=0)
    assert isinstance(ate, float)

    # 训练后 PEHE 应该明显小于零模型（全 0 预测）
    pehe_model = pehe(cate, toy["cate_test"])
    pehe_zero = pehe(np.zeros_like(toy["cate_test"]), toy["cate_test"])
    assert pehe_model < pehe_zero, f"训练后 PEHE={pehe_model:.3f} 应 < 零模型 {pehe_zero:.3f}"

    # ATE error 也应该有意义
    assert ate_error(cate, toy["cate_test"]) < pehe_zero


@pytest.mark.integration
def test_save_load_roundtrip(toy: dict, tmp_path: Path) -> None:
    """save → load 后预测一致。"""
    est = GACRNet(
        num_treatments=3,
        hidden_dim=32,
        head_hidden=16,
        epochs=20,
        batch_size=64,
        lr=5e-3,
        verbose=False,
        random_state=42,
    )
    est.fit(toy["x_train"], toy["t_train"], toy["y_train"])
    po_before = est.predict_potential_outcomes(toy["x_test"])

    save_path = tmp_path / "gacrnet.pt"
    est.save(save_path)

    loaded = GACRNet.load(save_path, device="cpu")
    po_after = loaded.predict_potential_outcomes(toy["x_test"])

    np.testing.assert_allclose(po_before, po_after, atol=1e-5)
    assert loaded.config.num_treatments == 3
    assert loaded.config.input_dim == 6


@pytest.mark.unit
def test_validation_errors() -> None:
    """非法参数应抛 ValueError。"""
    with pytest.raises(ValueError, match="num_treatments"):
        GACRNet(num_treatments=1)
    with pytest.raises(ValueError, match="val_split"):
        GACRNet(num_treatments=2, val_split=1.5)


@pytest.mark.unit
def test_not_fitted_error() -> None:
    """未 fit 调用 predict 抛 RuntimeError。"""
    est = GACRNet(num_treatments=2)
    with pytest.raises(RuntimeError, match="尚未训练"):
        est.predict_potential_outcomes(np.zeros((3, 5)))


@pytest.mark.unit
def test_treatment_index_check() -> None:
    """越界 treatment 索引抛错。"""
    x, t, y, *_ = _make_toy_data(n=50, d=4, k=2, seed=1)
    est = GACRNet(
        num_treatments=2,
        hidden_dim=16,
        head_hidden=8,
        epochs=2,
        verbose=False,
        random_state=0,
    )
    est.fit(x, t, y)
    with pytest.raises(ValueError, match="treatment 索引"):
        est.estimate_cate(x, treatment_a=2, treatment_b=0)
