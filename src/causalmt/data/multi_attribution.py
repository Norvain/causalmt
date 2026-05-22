"""EconML multi_attribution_sample.csv 数据加载。

来源：https://github.com/py-why/EconML 的 ROI 多归因数据，含两个干预
（Tech Support / Discount），4 个二元特征 + 4 个连续特征。

CSV 两种形态自动兼容：
    - **原始**（11 列）：含 `IT Spend / Employee Count / PC Count / Size` 等原始连续列，
      loader 自动做 log(x+1) 变换；不含真值列
    - **扩展**（含 `logIT / logEmp / ...` 与 `tau_1_true / tau_2_true / tau_12_true`）：
      直接使用 log 列与真值

适用场景：
    split="atomic"   → 排除两干预都有的样本，t ∈ {0=ctrl, 1=Tech Support, 2=Discount}
    split="combined" → 仅两干预都有的样本，y 是组合下的观察结果
    split="all"      → 全部样本 + 多热 (Tech Support, Discount) 矩阵
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from causalmt.data._cache import ensure_local_path
from causalmt.data.base import CausalDataset

__all__ = ["load_multi_attribution", "MULTI_ATTRIBUTION_URL"]

MULTI_ATTRIBUTION_URL: str = (
    "https://raw.githubusercontent.com/py-why/EconML/data/datasets/ROI/"
    "multi_attribution_sample.csv"
)

_BINARY_COLS: tuple[str, ...] = ("Global Flag", "Major Flag", "SMC Flag", "Commercial Flag")
_RAW_CONT_COLS: tuple[str, ...] = ("IT Spend", "Employee Count", "PC Count", "Size")
_LOG_CONT_COLS: tuple[str, ...] = ("logIT", "logEmp", "logPC", "logSize")
_TREATMENT_COLS: tuple[str, str] = ("Tech Support", "Discount")
_OUTCOME_COL: str = "Revenue"


def load_multi_attribution(
    path_or_url: str | Path | None = None,
    *,
    split: str = "atomic",
    cache_dir: str | Path | None = None,
) -> CausalDataset:
    """加载 EconML multi_attribution CSV。

    Args:
        path_or_url: 本地路径或 HTTP URL。None 时用 EconML 官方 URL 自动下载至缓存
        split: "atomic" | "combined" | "all"
        cache_dir: URL 下载缓存目录，None 时用 ~/.causalmt_cache/

    Returns:
        CausalDataset；含真值列时附 cate_true / ate_true。
        `atomic` / `combined` 时 t 为 (n,) 索引；`all` 时 t 为 (n, 2) 多热矩阵。
    """
    if split not in {"atomic", "combined", "all"}:
        raise ValueError(f"split 应为 'atomic' | 'combined' | 'all'，得 {split!r}")

    url_or_path = path_or_url if path_or_url is not None else MULTI_ATTRIBUTION_URL
    local = ensure_local_path(url_or_path, cache_dir=cache_dir)
    df = pd.read_csv(local)
    _check_required(df)

    if split == "atomic":
        return _build_atomic(df)
    if split == "combined":
        return _build_combined(df)
    return _build_all(df)


def _check_required(df: pd.DataFrame) -> None:
    """必备列：二元特征 + 连续特征（log 或 raw）+ 两干预列 + Revenue。"""
    required = list(_BINARY_COLS) + list(_TREATMENT_COLS) + [_OUTCOME_COL]
    has_log = all(c in df.columns for c in _LOG_CONT_COLS)
    has_raw = all(c in df.columns for c in _RAW_CONT_COLS)
    if not (has_log or has_raw):
        raise ValueError(
            "multi_attribution CSV 既缺少 log 连续列 "
            f"{_LOG_CONT_COLS} 也缺少原始连续列 {_RAW_CONT_COLS}"
        )
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"multi_attribution CSV 缺少列: {missing}")


def _build_features(df: pd.DataFrame) -> tuple[np.ndarray, tuple[str, ...]]:
    """构造特征矩阵，优先使用 log 列；否则对 raw 列做 log(x+1) 变换。"""
    binary = df[list(_BINARY_COLS)].to_numpy(dtype=np.float32)
    if all(c in df.columns for c in _LOG_CONT_COLS):
        cont = df[list(_LOG_CONT_COLS)].to_numpy(dtype=np.float32)
        cont_names = _LOG_CONT_COLS
    else:
        cont_raw = df[list(_RAW_CONT_COLS)].to_numpy(dtype=np.float64)
        cont = np.log1p(cont_raw).astype(np.float32)
        cont_names = tuple(f"log_{c.replace(' ', '')}" for c in _RAW_CONT_COLS)
    x = np.hstack([binary, cont]).astype(np.float32)
    return x, _BINARY_COLS + cont_names


def _maybe_taus(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """采集存在的真值列。"""
    extra: dict[str, np.ndarray] = {}
    for col in ("tau_1_true", "tau_2_true", "tau_12_true"):
        if col in df.columns:
            extra[col] = df[col].to_numpy(dtype=np.float32)
    return extra


def _build_atomic(df: pd.DataFrame) -> CausalDataset:
    """排除两干预都有的样本，t ∈ {0=ctrl, 1=Tech Support, 2=Discount}。"""
    ts = df["Tech Support"] == 1
    disc = df["Discount"] == 1
    mask = ~(ts & disc)
    sub = df[mask].reset_index(drop=True)

    t = np.zeros(len(sub), dtype=np.int64)
    t[(sub["Tech Support"] == 1).to_numpy()] = 1
    t[(sub["Discount"] == 1).to_numpy()] = 2

    x, feat_names = _build_features(sub)
    y = sub[_OUTCOME_COL].to_numpy(dtype=np.float32)
    taus = _maybe_taus(sub)

    cate_true: np.ndarray | None = None
    ate_true: float | None = None
    y_potential: np.ndarray | None = None
    if "tau_1_true" in taus and "tau_2_true" in taus:
        tau1 = taus["tau_1_true"]
        tau2 = taus["tau_2_true"]
        cate_true = tau1.astype(np.float32)  # 默认 CATE = τ_(1 vs 0)
        ate_true = float(cate_true.mean())
        # 构造潜在结果（τ_0 = 0）
        tau_per_t = np.stack([np.zeros_like(tau1), tau1, tau2], axis=1)
        y_baseline = y - tau_per_t[np.arange(len(sub)), t]
        y_potential = (y_baseline[:, None] + tau_per_t).astype(np.float32)

    return CausalDataset(
        x=x, t=t, y=y,
        y_potential=y_potential,
        cate_true=cate_true,
        ate_true=ate_true,
        feature_names=feat_names,
        name="multi_attribution_atomic",
        extra=taus,
    )


def _build_combined(df: pd.DataFrame) -> CausalDataset:
    """仅两干预都=1 的样本。t 全为 1（标记"已组合干预"）。"""
    sub = df[(df["Tech Support"] == 1) & (df["Discount"] == 1)].reset_index(drop=True)
    if len(sub) == 0:
        raise ValueError("数据中没有 Tech Support=1 且 Discount=1 的样本")

    x, feat_names = _build_features(sub)
    y = sub[_OUTCOME_COL].to_numpy(dtype=np.float32)
    t = np.ones(len(sub), dtype=np.int64)
    taus = _maybe_taus(sub)

    cate_true: np.ndarray | None = None
    if "tau_1_true" in taus and "tau_2_true" in taus:
        cate_true = (taus["tau_1_true"] + taus["tau_2_true"]).astype(np.float32)
        if "tau_12_true" in taus:
            cate_true = (cate_true + taus["tau_12_true"]).astype(np.float32)

    return CausalDataset(
        x=x, t=t, y=y,
        cate_true=cate_true,
        feature_names=feat_names,
        name="multi_attribution_combined",
        extra=taus,
    )


def _build_all(df: pd.DataFrame) -> CausalDataset:
    """全部样本，t 为 (n, 2) 多热矩阵 [Tech Support, Discount]。"""
    x, feat_names = _build_features(df)
    y = df[_OUTCOME_COL].to_numpy(dtype=np.float32)
    t_multihot = df[list(_TREATMENT_COLS)].to_numpy(dtype=np.int64)
    taus = _maybe_taus(df)

    cate_true: np.ndarray | None = None
    ate_true: float | None = None
    if "tau_1_true" in taus:
        cate_true = taus["tau_1_true"].astype(np.float32)
        ate_true = float(cate_true.mean())

    return CausalDataset(
        x=x, t=t_multihot, y=y,
        cate_true=cate_true,
        ate_true=ate_true,
        feature_names=feat_names,
        name="multi_attribution_all",
        extra=taus,
    )
