"""损失函数模块。

`LossFunction` 是训练循环的统一入口，组合 outcome、GPS、MMD、tarreg 四个项。
单独的子损失也对外暴露，便于自定义训练逻辑。
"""

from causalmt.losses.combined import LossFunction
from causalmt.losses.gps import gps_classification_loss
from causalmt.losses.mmd import MMDLoss, gaussian_kernel, mmd_loss
from causalmt.losses.outcome import multi_treatment_outcome_loss
from causalmt.losses.tarreg import tarreg_loss

__all__ = [
    "LossFunction",
    "multi_treatment_outcome_loss",
    "gps_classification_loss",
    "MMDLoss",
    "mmd_loss",
    "gaussian_kernel",
    "tarreg_loss",
]
