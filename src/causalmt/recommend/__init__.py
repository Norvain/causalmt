"""组合干预推荐模块。

主要导出：
    - IDCIO：组合干预推荐主类
    - build_qubo_matrix / solve_qubo_exhaustive / solve_qubo_greedy：QUBO 工具
    - mc_dropout_predict：MC Dropout 不确定性估计
"""

from causalmt.recommend.idcio import IDCIO
from causalmt.recommend.mc_dropout import enable_dropout, mc_dropout_predict
from causalmt.recommend.qubo import (
    EXHAUSTIVE_MAX_K,
    build_qubo_matrix,
    solve_qubo_exhaustive,
    solve_qubo_greedy,
)

__all__ = [
    "IDCIO",
    "build_qubo_matrix",
    "solve_qubo_exhaustive",
    "solve_qubo_greedy",
    "EXHAUSTIVE_MAX_K",
    "mc_dropout_predict",
    "enable_dropout",
]
