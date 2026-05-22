"""数据加载模块。

本包不附带数据集本身，loader 接受用户提供的本地路径或 URL，URL 会自动下载到
缓存目录（默认 ~/.causalmt_cache/，可通过环境变量 CAUSALMT_CACHE_DIR 覆盖）。

主要导出：
    - CausalDataset：统一数据容器
    - load_ihdp：IHDP 100 切片基准
    - load_multi_attribution：EconML 多归因合成数据
"""

from causalmt.data.base import CausalDataset
from causalmt.data.ihdp import load_ihdp
from causalmt.data.multi_attribution import MULTI_ATTRIBUTION_URL, load_multi_attribution

__all__ = [
    "CausalDataset",
    "load_ihdp",
    "load_multi_attribution",
    "MULTI_ATTRIBUTION_URL",
]
