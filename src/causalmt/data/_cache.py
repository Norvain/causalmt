"""HTTP 下载缓存工具。

默认缓存目录 `~/.causalmt_cache/`，可通过 `CAUSALMT_CACHE_DIR` 环境变量覆盖。
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

from causalmt.utils import get_logger

__all__ = ["default_cache_dir", "ensure_local_path"]


def default_cache_dir() -> Path:
    """返回缓存目录路径（必要时创建）。"""
    env = os.environ.get("CAUSALMT_CACHE_DIR")
    base = Path(env).expanduser() if env else Path.home() / ".causalmt_cache"
    base.mkdir(parents=True, exist_ok=True)
    return base


def ensure_local_path(
    path_or_url: str | Path,
    *,
    cache_dir: str | Path | None = None,
    filename: str | None = None,
) -> Path:
    """将路径或 URL 解析为本地文件路径，URL 自动下载并缓存。

    Args:
        path_or_url: 本地路径（直接返回）或 http(s) URL
        cache_dir: 缓存目录，None 时用 default_cache_dir()
        filename: 缓存文件名，None 时从 URL 推断或用哈希

    Returns:
        本地文件路径
    """
    s = str(path_or_url)
    if not (s.startswith("http://") or s.startswith("https://")):
        p = Path(path_or_url).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"文件不存在: {p}")
        return p

    cache = Path(cache_dir).expanduser() if cache_dir else default_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = _filename_from_url(s)
    target = cache / filename
    if target.exists():
        return target

    logger = get_logger("causalmt.data")
    logger.info("下载 %s → %s", s, target)
    urllib.request.urlretrieve(s, target)  # noqa: S310 (受控来源)
    return target


def _filename_from_url(url: str) -> str:
    """从 URL 末段推断文件名，无扩展则使用 URL 哈希。"""
    tail = url.rstrip("/").rsplit("/", 1)[-1].split("?", 1)[0]
    if "." in tail:
        return tail
    digest = hashlib.sha1(url.encode()).hexdigest()[:12]  # noqa: S324 (用于命名，非加密)
    return f"download-{digest}"
