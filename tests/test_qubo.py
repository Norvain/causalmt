"""QUBO 求解器单元测试。"""

from __future__ import annotations

import numpy as np
import pytest

from causalmt.recommend.qubo import (
    build_qubo_matrix,
    solve_qubo_exhaustive,
    solve_qubo_greedy,
)


@pytest.mark.unit
def test_build_qubo_single() -> None:
    """单样本 Q 矩阵：对角线 atomic，上三角 interaction。"""
    atomic = np.array([1.0, 2.0, 3.0])
    interactions = {(0, 1): 0.5, (0, 2): -0.3, (1, 2): 0.1}
    q = build_qubo_matrix(atomic, interactions)
    expected = np.array([[1.0, 0.5, -0.3], [0.0, 2.0, 0.1], [0.0, 0.0, 3.0]])
    np.testing.assert_allclose(q, expected)


@pytest.mark.unit
def test_build_qubo_batched_dict() -> None:
    """批量 Q 矩阵，dict 形式的 interaction。"""
    atomic = np.array([[1.0, 2.0], [0.5, 1.5]])  # (2, 2)
    interactions = {(0, 1): np.array([0.1, 0.2])}
    q = build_qubo_matrix(atomic, interactions)
    assert q.shape == (2, 2, 2)
    assert q[0, 0, 0] == 1.0 and q[0, 0, 1] == 0.1
    assert q[1, 0, 0] == 0.5 and q[1, 0, 1] == 0.2


@pytest.mark.unit
def test_solve_qubo_exhaustive_simple() -> None:
    """两个正 atomic、正 interaction：最优应为 (1, 1)。"""
    q = np.array([[1.0, 0.5], [0.0, 1.0]])
    action, value = solve_qubo_exhaustive(q, maximize=True)
    np.testing.assert_array_equal(action, [1, 1])
    assert value == pytest.approx(1.0 + 1.0 + 0.5)


@pytest.mark.unit
def test_solve_qubo_with_costs() -> None:
    """加上成本后 (1, 1) 不再最优。"""
    q = np.array([[1.0, 0.5], [0.0, 1.0]])
    costs = np.array([0.6, 0.6])  # 单干预成本 0.6
    action, value = solve_qubo_exhaustive(q, costs=costs, maximize=True)
    # 各组合的净值：
    # (0,0): 0, (1,0): 1-0.6=0.4, (0,1): 1-0.6=0.4, (1,1): 2+0.5-1.2=1.3
    np.testing.assert_array_equal(action, [1, 1])
    assert value == pytest.approx(1.3)


@pytest.mark.unit
def test_solve_qubo_minimize() -> None:
    """minimize 模式（如血糖降低场景）：负 atomic 应被选中。"""
    q = np.array([[-1.0, 0.0], [0.0, -2.0]])
    action, value = solve_qubo_exhaustive(q, maximize=False)
    np.testing.assert_array_equal(action, [1, 1])
    assert value == pytest.approx(-3.0)


@pytest.mark.unit
def test_solve_qubo_batched() -> None:
    """批量求解：每个样本独立最优。"""
    q = np.stack(
        [
            np.array([[1.0, 0.0], [0.0, 1.0]]),  # → (1, 1)
            np.array([[-1.0, 0.0], [0.0, -2.0]]),  # → (0, 0)
        ]
    )
    actions, values = solve_qubo_exhaustive(q, maximize=True)
    assert actions.shape == (2, 2)
    np.testing.assert_array_equal(actions[0], [1, 1])
    np.testing.assert_array_equal(actions[1], [0, 0])
    np.testing.assert_allclose(values, [2.0, 0.0])


@pytest.mark.unit
def test_solve_qubo_batched_sample_costs() -> None:
    """批量求解支持每个样本不同的干预成本。"""
    q = np.stack(
        [
            np.array([[2.0, 0.0], [0.0, 1.0]]),
            np.array([[2.0, 0.0], [0.0, 1.0]]),
        ]
    )
    costs = np.array(
        [
            [0.0, 0.0],
            [3.0, 0.0],
        ]
    )
    actions, values = solve_qubo_exhaustive(q, costs=costs, maximize=True)
    np.testing.assert_array_equal(actions[0], [1, 1])
    np.testing.assert_array_equal(actions[1], [0, 1])
    np.testing.assert_allclose(values, [3.0, 1.0])


@pytest.mark.unit
def test_solve_qubo_greedy_matches_exhaustive_on_easy() -> None:
    """简单可分场景下 greedy 和 exhaustive 一致。"""
    rng = np.random.default_rng(0)
    for _ in range(20):
        atomic = rng.normal(size=4)
        q = build_qubo_matrix(atomic, None)  # 仅对角线，无交互
        a_ex, v_ex = solve_qubo_exhaustive(q, maximize=True)
        a_gr, v_gr = solve_qubo_greedy(q, maximize=True)
        np.testing.assert_array_equal(a_ex, a_gr)
        assert v_ex == pytest.approx(v_gr)


@pytest.mark.unit
def test_solve_qubo_invalid_pair() -> None:
    """build_qubo_matrix 对非法交互对应报错。"""
    atomic = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="0 ≤ i < j"):
        build_qubo_matrix(atomic, {(1, 0): 0.5})
