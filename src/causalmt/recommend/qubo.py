"""QUBO 求解器：组合干预最优 action 搜索。

K 个干预（不含控制组），每个 action z_k ∈ {0, 1}。目标函数::

    V(z) = Σ_k z_k * Q[k, k] + Σ_{i<j} z_i * z_j * Q[i, j] - cost(z)

其中 Q 矩阵的对角线放原子效应 τ_k，上三角放交互效应 τ_{ij}。
支持两种求解器：
    - exhaustive：穷举 2^K 个 binary 组合（K ≤ 12 时秒级）
    - greedy：贪心翻转启发式（K 较大时使用）
"""

from __future__ import annotations

import itertools

import numpy as np

__all__ = [
    "build_qubo_matrix",
    "solve_qubo_exhaustive",
    "solve_qubo_greedy",
    "EXHAUSTIVE_MAX_K",
]

EXHAUSTIVE_MAX_K: int = 12  # 2^12 = 4096，仍可秒级穷举


def build_qubo_matrix(
    atomic_effects: np.ndarray,
    interaction_effects: dict[tuple[int, int], np.ndarray] | np.ndarray | None = None,
) -> np.ndarray:
    """构造 QUBO 矩阵。

    Args:
        atomic_effects: shape (n, K) 或 (K,)，K 个原子干预效应
        interaction_effects:
            - dict[(i, j), array(n,) 或 scalar]：稀疏指定交互项（i < j）
            - ndarray shape (n, K, K) 或 (K, K)：稠密上三角矩阵
            - None：无交互项

    Returns:
        Q: 形状对齐 atomic_effects 的 batched 矩阵，
           (n, K, K) 或 (K, K)，对角线为原子效应，上三角为交互效应。
    """
    atomic = np.asarray(atomic_effects)
    if atomic.ndim == 1:
        return _build_single(atomic, interaction_effects)
    if atomic.ndim != 2:
        raise ValueError(f"atomic_effects 维度应为 1 或 2，得 {atomic.ndim}")

    n, k = atomic.shape
    q = np.zeros((n, k, k), dtype=np.float64)
    diag_idx = np.arange(k)
    q[:, diag_idx, diag_idx] = atomic

    if interaction_effects is None:
        return q
    if isinstance(interaction_effects, dict):
        for (i, j), val in interaction_effects.items():
            _check_pair(i, j, k)
            q[:, i, j] = np.asarray(val).reshape(-1) if hasattr(val, "__len__") else val
    else:
        inter = np.asarray(interaction_effects)
        if inter.shape != (n, k, k):
            raise ValueError(f"interaction_effects 形状应为 ({n}, {k}, {k})，得 {inter.shape}")
        # 只取严格上三角
        triu = np.triu(np.ones((k, k), dtype=bool), k=1)
        q[:, triu] = inter[:, triu]
    return q


def _build_single(
    atomic: np.ndarray,
    interaction_effects: dict[tuple[int, int], float] | np.ndarray | None,
) -> np.ndarray:
    k = atomic.shape[0]
    q = np.zeros((k, k), dtype=np.float64)
    np.fill_diagonal(q, atomic)
    if interaction_effects is None:
        return q
    if isinstance(interaction_effects, dict):
        for (i, j), val in interaction_effects.items():
            _check_pair(i, j, k)
            q[i, j] = float(val)
    else:
        inter = np.asarray(interaction_effects)
        if inter.shape != (k, k):
            raise ValueError(f"interaction_effects 形状应为 ({k}, {k})，得 {inter.shape}")
        triu = np.triu(np.ones((k, k), dtype=bool), k=1)
        q[triu] = inter[triu]
    return q


def _check_pair(i: int, j: int, k: int) -> None:
    if not (0 <= i < j < k):
        raise ValueError(f"交互对应满足 0 ≤ i < j < K={k}，得 ({i}, {j})")


def _enumerate_actions(k: int) -> np.ndarray:
    """生成所有 2^K 个 binary action 向量，shape (2^K, K)。"""
    if k > EXHAUSTIVE_MAX_K:
        raise ValueError(f"穷举仅支持 K ≤ {EXHAUSTIVE_MAX_K}，得 K={k}。请改用 solve_qubo_greedy。")
    return np.array(list(itertools.product([0, 1], repeat=k)), dtype=np.float64)


def _objective(actions: np.ndarray, q: np.ndarray, costs: np.ndarray | None) -> np.ndarray:
    """计算 (M, K) actions 在单/批 Q 矩阵下的目标值 V(z) - cost(z)。

    Args:
        actions: (M, K) 候选 action
        q: (K, K) 或 (n, K, K)
        costs: (K,) 或 None，每个干预的单位成本

    Returns:
        (M,) 或 (n, M) 的目标值矩阵
    """
    # 一次性计算 z^T Q z（含对角线 + 上三角交互）
    if q.ndim == 2:
        # actions @ Q @ actions.T 取对角线，等价于按行计算 z_m Q z_m
        # 对每个 m 行：sum_{i,j} z_i Q_ij z_j（这里 Q 已经只上三角有交互值）
        vals = np.einsum("mi,ij,mj->m", actions, q, actions)
        if costs is not None:
            vals = vals - actions @ costs
        return vals
    # batched
    vals = np.einsum("mi,nij,mj->nm", actions, q, actions)
    if costs is not None:
        vals = vals - actions @ costs
    return vals


def solve_qubo_exhaustive(
    q: np.ndarray,
    *,
    costs: np.ndarray | None = None,
    maximize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """穷举求解 QUBO，返回最优 action 和目标值。

    Args:
        q: (K, K) 单样本 或 (n, K, K) 批量
        costs: (K,) 每干预单位成本，None 时无成本
        maximize: True 找 V(z) 最大；False 找最小

    Returns:
        actions: (K,) 或 (n, K) 最优 binary action
        values: () 或 (n,) 对应目标值
    """
    q = np.asarray(q, dtype=np.float64)
    k = q.shape[-1]
    candidates = _enumerate_actions(k)  # (2^K, K)
    if costs is not None:
        costs = np.asarray(costs, dtype=np.float64).reshape(k)

    vals = _objective(candidates, q, costs)  # (M,) 或 (n, M)

    if q.ndim == 2:
        idx = int(vals.argmax() if maximize else vals.argmin())
        return candidates[idx].astype(np.int64), vals[idx]

    idx = vals.argmax(axis=1) if maximize else vals.argmin(axis=1)
    best_actions = candidates[idx].astype(np.int64)  # (n, K)
    best_values = vals[np.arange(q.shape[0]), idx]
    return best_actions, best_values


def solve_qubo_greedy(
    q: np.ndarray,
    *,
    costs: np.ndarray | None = None,
    maximize: bool = True,
    max_iter: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """贪心翻转启发式：从全零开始，每步翻转使目标改进最大的位。

    适用于 K 较大、穷举不可行的场景。不保证全局最优。

    Args:
        q: (K, K) 或 (n, K, K)
        costs: (K,) 或 None
        maximize: True 最大化，False 最小化
        max_iter: 最大翻转轮数（每轮至多翻转一位）

    Returns:
        actions: (K,) 或 (n, K)
        values: () 或 (n,)
    """
    q = np.asarray(q, dtype=np.float64)
    if q.ndim == 2:
        return _greedy_single(q, costs, maximize, max_iter)

    n, k, _ = q.shape
    actions = np.zeros((n, k), dtype=np.int64)
    values = np.zeros(n, dtype=np.float64)
    for i in range(n):
        actions[i], values[i] = _greedy_single(q[i], costs, maximize, max_iter)
    return actions, values


def _greedy_single(
    q: np.ndarray,
    costs: np.ndarray | None,
    maximize: bool,
    max_iter: int,
) -> tuple[np.ndarray, float]:
    k = q.shape[0]
    z = np.zeros(k, dtype=np.float64)
    cost_arr = np.asarray(costs, dtype=np.float64).reshape(k) if costs is not None else None
    current = _eval_one(z, q, cost_arr)

    for _ in range(max_iter):
        best_gain = 0.0
        best_bit = -1
        for bit in range(k):
            z_new = z.copy()
            z_new[bit] = 1 - z_new[bit]
            new_val = _eval_one(z_new, q, cost_arr)
            gain = new_val - current if maximize else current - new_val
            if gain > best_gain:
                best_gain = gain
                best_bit = bit
        if best_bit < 0:
            break
        z[best_bit] = 1 - z[best_bit]
        current = current + best_gain if maximize else current - best_gain

    return z.astype(np.int64), float(current)


def _eval_one(z: np.ndarray, q: np.ndarray, costs: np.ndarray | None) -> float:
    val = float(z @ q @ z)
    if costs is not None:
        val -= float(z @ costs)
    return val
