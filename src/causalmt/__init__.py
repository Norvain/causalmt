"""causalmt - 多值干预因果表示学习与组合干预推荐工具包。

主要导出：
    - GACRNet：多值干预效应估计模型（论文方法）
    - IDCIO：组合干预推荐管线（依赖 GACRNet 或外部 atomic 效应输入）
"""

from causalmt._version import __version__
from causalmt.estimator import GACRNet
from causalmt.recommend import IDCIO

__all__ = ["__version__", "GACRNet", "IDCIO"]
