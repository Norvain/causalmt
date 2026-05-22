"""通用工具：随机种子、设备解析、numpy↔tensor 转换。"""

from __future__ import annotations

import logging
import random
from typing import Any

import numpy as np
import torch

__all__ = [
    "set_seed",
    "resolve_device",
    "to_tensor",
    "to_numpy",
    "get_logger",
]


def set_seed(seed: int) -> None:
    """固定所有 RNG 种子（python random / numpy / torch / cuda / mps）。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device: str | torch.device) -> torch.device:
    """解析 device 字符串。

    "auto" 优先级：cuda > mps > cpu。
    """
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def to_tensor(
    x: Any,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    """numpy / list / pandas → torch.Tensor。已为 Tensor 时仅做 dtype/device 转换。"""
    if isinstance(x, torch.Tensor):
        out = x.to(dtype=dtype)
    elif hasattr(x, "values"):  # pandas Series/DataFrame
        out = torch.as_tensor(x.values, dtype=dtype)
    else:
        out = torch.as_tensor(np.asarray(x), dtype=dtype)
    if device is not None:
        out = out.to(device)
    return out


def to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    """torch.Tensor → numpy.ndarray（detach 并搬到 CPU）。"""
    if isinstance(x, np.ndarray):
        return x
    return x.detach().cpu().numpy()


def get_logger(name: str = "causalmt", level: int = logging.INFO) -> logging.Logger:
    """获取格式化的 causalmt 日志器（仅首次添加 handler）。"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger
