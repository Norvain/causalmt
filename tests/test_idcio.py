"""IDCIO 端到端集成测试。

合成数据生成：
    控制组 y_0 = f(x)
    单干预 y_k = y_0 + τ_k(x)
    组合 (i, j) y_{ij} = y_0 + τ_i + τ_j + τ_{ij}(x)
"""

from __future__ import annotations

import numpy as np
import pytest

from causalmt import IDCIO, GACRNet


def _make_combined_data(n: int = 600, d: int = 5, seed: int = 0):
    """生成含 2 个干预的合成数据：原子样本 (T1 or T2 or 控制) + 组合样本 (T1+T2)。"""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d)).astype(np.float32)

    # 真效应函数（同符号方便测试 recommend 命中 (1,1)）
    tau1 = (0.5 + 0.2 * x[:, 0]).astype(np.float32)
    tau2 = (0.7 + 0.1 * x[:, 1]).astype(np.float32)
    tau12 = (0.2 + 0.05 * x[:, 2]).astype(np.float32)  # 正交互
    mu0 = (0.0 + 0.1 * x[:, 0]).astype(np.float32)

    # 随机分配 4 个干预组：0=control, 1=T1, 2=T2, 3=T1+T2
    t_combo = rng.integers(0, 4, size=n)
    y = mu0.copy()
    y[t_combo == 1] += tau1[t_combo == 1]
    y[t_combo == 2] += tau2[t_combo == 2]
    y[t_combo == 3] += tau1[t_combo == 3] + tau2[t_combo == 3] + tau12[t_combo == 3]
    y += rng.normal(scale=0.05, size=n).astype(np.float32)

    # 原子集：t ∈ {0, 1, 2}，组合集：t == 3
    atomic_mask = t_combo != 3
    combined_mask = t_combo == 3
    t_atomic_idx = t_combo[atomic_mask]  # 0/1/2 对应控制/T1/T2

    return {
        "x_atomic": x[atomic_mask],
        "t_atomic": t_atomic_idx,
        "y_atomic": y[atomic_mask],
        "x_combined": x[combined_mask],
        "y_combined": y[combined_mask],
        "x_all": x,
        "tau1_true": tau1,
        "tau2_true": tau2,
        "tau12_true": tau12,
    }


@pytest.fixture(scope="module")
def data() -> dict:
    return _make_combined_data(n=800, d=5, seed=0)


@pytest.fixture(scope="module")
def trained_gacrnet(data: dict) -> GACRNet:
    est = GACRNet(
        num_treatments=3,
        hidden_dim=32,
        head_hidden=16,
        epochs=80,
        batch_size=64,
        lr=5e-3,
        use_mmd=True,
        use_tarreg=True,
        verbose=False,
        random_state=42,
    )
    est.fit(data["x_atomic"], data["t_atomic"], data["y_atomic"])
    return est


@pytest.mark.integration
def test_idcio_from_estimator_full_pipeline(data: dict, trained_gacrnet: GACRNet) -> None:
    """from_estimator 路径：fit_interaction + recommend 全流程。"""
    rec = IDCIO.from_estimator(
        atomic_estimator=trained_gacrnet,
        interaction_hidden=(32, 16),
        epochs=60,
        batch_size=64,
        lr=5e-3,
        early_stopping_patience=15,
        verbose=False,
        random_state=42,
    )
    rec.fit_interaction(
        data["x_combined"],
        data["y_combined"],
        treatment_pairs=[(0, 1)],
    )

    # 预测交互项
    inter = rec.predict_interactions(data["x_combined"][:20])
    assert (0, 1) in inter
    assert inter[(0, 1)].shape == (20,)

    # 推荐
    actions = rec.recommend(data["x_combined"][:30], method="exhaustive", maximize=True)
    assert actions.shape == (30, 2)
    assert actions.dtype == np.int64
    # 由于真效应全正，最优应大多为 (1, 1)
    combined_rate = (actions.sum(axis=1) == 2).mean()
    assert combined_rate >= 0.6, f"组合推荐率 {combined_rate:.2f} 过低（期望 ≥ 0.6）"


@pytest.mark.integration
def test_idcio_from_effects_path(data: dict) -> None:
    """from_effects 路径：用外部 atomic 效应做推荐。"""
    n_combined = len(data["x_combined"])
    # 提供"近似真值"的 atomic 效应
    atomic_external = np.stack(
        [data["tau1_true"][:n_combined], data["tau2_true"][:n_combined]], axis=1
    )

    rec = IDCIO.from_effects(
        atomic_effects=atomic_external,
        interaction_hidden=(32, 16),
        epochs=40,
        batch_size=64,
        lr=5e-3,
        early_stopping_patience=15,
        verbose=False,
        random_state=42,
    )
    rec.fit_interaction(data["x_combined"], data["y_combined"])

    # 推荐时显式提供 atomic_effects
    actions = rec.recommend(
        data["x_combined"][:20],
        atomic_effects=atomic_external[:20],
        method="exhaustive",
        maximize=True,
    )
    assert actions.shape == (20, 2)


@pytest.mark.integration
def test_idcio_recommend_with_costs(data: dict, trained_gacrnet: GACRNet) -> None:
    """加大成本时，组合推荐应减少（部分样本应不再推荐两个都用）。"""
    rec = IDCIO.from_estimator(
        atomic_estimator=trained_gacrnet,
        interaction_hidden=(32, 16),
        epochs=40,
        lr=5e-3,
        verbose=False,
        random_state=42,
    )
    rec.fit_interaction(data["x_combined"], data["y_combined"])

    x_subset = data["x_combined"][:50]
    no_cost = rec.recommend(x_subset, costs=None, maximize=True)
    high_cost = rec.recommend(x_subset, costs=np.array([10.0, 10.0]), maximize=True)

    # 极高成本应迫使大多数样本变为 (0, 0)
    assert high_cost.sum() < no_cost.sum()


@pytest.mark.integration
def test_idcio_greedy_method(data: dict, trained_gacrnet: GACRNet) -> None:
    """method='greedy' 在小 K 上应和 exhaustive 一致。"""
    rec = IDCIO.from_estimator(
        atomic_estimator=trained_gacrnet,
        interaction_hidden=(32, 16),
        epochs=40,
        lr=5e-3,
        verbose=False,
        random_state=42,
    )
    rec.fit_interaction(data["x_combined"], data["y_combined"])

    x_subset = data["x_combined"][:30]
    a_ex = rec.recommend(x_subset, method="exhaustive", maximize=True)
    a_gr = rec.recommend(x_subset, method="greedy", maximize=True)
    # K=2 时两者高度一致（贪心 ≥ 90% 命中即可）
    agreement = (a_ex == a_gr).all(axis=1).mean()
    assert agreement >= 0.9


@pytest.mark.integration
def test_idcio_mc_dropout_confidence(data: dict, trained_gacrnet: GACRNet) -> None:
    """MC Dropout 模式下返回 (actions, confidence) 二元组。"""
    rec = IDCIO.from_estimator(
        atomic_estimator=trained_gacrnet,
        interaction_hidden=(32, 16),
        interaction_dropout=0.3,
        use_mc_dropout=True,
        mc_samples=8,  # 测试加速
        epochs=40,
        lr=5e-3,
        verbose=False,
        random_state=42,
    )
    rec.fit_interaction(data["x_combined"], data["y_combined"])

    result = rec.recommend(
        data["x_combined"][:15],
        method="exhaustive",
        maximize=True,
        return_uncertainty=True,
    )
    assert isinstance(result, tuple) and len(result) == 2
    actions, confidence = result
    assert actions.shape == (15, 2)
    assert confidence.shape == (15,)
    assert np.all((confidence >= 0) & (confidence <= 1))


@pytest.mark.unit
def test_idcio_invalid_inputs() -> None:
    """非法输入抛错。"""
    with pytest.raises(ValueError, match="n_atomic_treatments"):
        IDCIO(n_atomic_treatments=1)

    arr_1d = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="2D"):
        IDCIO.from_effects(atomic_effects=arr_1d)

    class FakeEst:
        pass

    with pytest.raises(TypeError, match="GACRNet"):
        IDCIO.from_estimator(atomic_estimator=FakeEst())
