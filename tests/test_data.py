"""数据加载模块测试。

不依赖真实数据集：测试都用 tmp_path 生成 dummy npz/CSV。
对真实 IHDP / multi_attribution 数据的兼容性通过 examples notebook 验证。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from causalmt.data import CausalDataset, load_ihdp, load_multi_attribution
from causalmt.data._cache import default_cache_dir, ensure_local_path


# ---------------- CausalDataset ----------------
@pytest.mark.unit
def test_causal_dataset_properties() -> None:
    x = np.zeros((10, 4))
    t = np.array([0, 1, 1, 0, 2, 2, 1, 0, 1, 2])
    y = np.arange(10).astype(np.float32)
    ds = CausalDataset(x=x, t=t, y=y, name="toy")
    assert ds.n == 10 and ds.d == 4 and ds.num_treatments == 3
    assert "toy" in repr(ds)


@pytest.mark.unit
def test_causal_dataset_multihot_t() -> None:
    """t 为 (n, K) 多热时 num_treatments 取列数。"""
    x = np.zeros((5, 3))
    t = np.array([[1, 0], [0, 1], [1, 1], [0, 0], [1, 0]])
    y = np.zeros(5)
    ds = CausalDataset(x=x, t=t, y=y)
    assert ds.num_treatments == 2


# ---------------- _cache ----------------
@pytest.mark.unit
def test_default_cache_dir_creates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CAUSALMT_CACHE_DIR", str(tmp_path / "cache"))
    d = default_cache_dir()
    assert d.exists() and d.is_dir()


@pytest.mark.unit
def test_ensure_local_path_existing_file(tmp_path: Path) -> None:
    p = tmp_path / "data.csv"
    p.write_text("a,b\n1,2\n")
    assert ensure_local_path(p) == p


@pytest.mark.unit
def test_ensure_local_path_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ensure_local_path(tmp_path / "missing.csv")


# ---------------- IHDP ----------------
def _make_dummy_ihdp(dir_: Path, n: int = 30, d: int = 5, n_slices: int = 3) -> None:
    rng = np.random.default_rng(0)
    for fname in [
        "ihdp_npci_1-100.train.npz",
        "ihdp_npci_1-100.test.npz",
    ]:
        x = rng.normal(size=(n, d, n_slices)).astype(np.float32)
        t = rng.integers(0, 2, size=(n, n_slices)).astype(np.int64)
        mu0 = rng.normal(size=(n, n_slices)).astype(np.float32)
        mu1 = mu0 + 0.5 + rng.normal(scale=0.1, size=(n, n_slices)).astype(np.float32)
        yf = np.where(t == 1, mu1, mu0) + rng.normal(scale=0.05, size=(n, n_slices)).astype(
            np.float32
        )
        ycf = np.where(t == 1, mu0, mu1)
        np.savez(dir_ / fname, x=x, t=t, yf=yf, ycf=ycf, mu0=mu0, mu1=mu1)


@pytest.mark.unit
def test_load_ihdp_smoke(tmp_path: Path) -> None:
    _make_dummy_ihdp(tmp_path, n=20, d=5, n_slices=3)
    train, test = load_ihdp(tmp_path, slice_idx=0)
    assert isinstance(train, CausalDataset) and isinstance(test, CausalDataset)
    assert train.x.shape == (20, 5)
    assert train.t.shape == (20,) and train.t.dtype == np.int64
    assert train.y.shape == (20,) and train.y.dtype == np.float32
    assert train.y_potential is not None and train.y_potential.shape == (20, 2)
    assert train.cate_true is not None and train.cate_true.shape == (20,)
    assert train.ate_true is not None
    assert "slice0" in train.name


@pytest.mark.unit
def test_load_ihdp_slice_bounds(tmp_path: Path) -> None:
    _make_dummy_ihdp(tmp_path, n_slices=3)
    with pytest.raises(ValueError, match="slice_idx"):
        load_ihdp(tmp_path, slice_idx=100)


@pytest.mark.unit
def test_load_ihdp_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="训练文件"):
        load_ihdp(tmp_path)


# ---------------- multi_attribution ----------------
def _make_dummy_attribution_csv(
    path: Path, n: int = 100, seed: int = 0, *, with_taus: bool = True, with_logs: bool = True
) -> None:
    """生成 dummy CSV。

    with_logs=True 时含 log 连续列；False 时只含原始 IT Spend 等列。
    with_taus=True 时含 tau_1_true / tau_2_true / tau_12_true。
    """
    rng = np.random.default_rng(seed)
    data: dict = {
        "Global Flag": rng.integers(0, 2, n),
        "Major Flag": rng.integers(0, 2, n),
        "SMC Flag": rng.integers(0, 2, n),
        "Commercial Flag": rng.integers(0, 2, n),
        "Tech Support": rng.integers(0, 2, n),
        "Discount": rng.integers(0, 2, n),
        "Revenue": rng.normal(size=n),
    }
    if with_logs:
        data.update(
            {
                "logIT": rng.normal(size=n),
                "logEmp": rng.normal(size=n),
                "logPC": rng.normal(size=n),
                "logSize": rng.normal(size=n),
            }
        )
    else:
        data.update(
            {
                "IT Spend": rng.uniform(100, 1000, n),
                "Employee Count": rng.uniform(10, 500, n),
                "PC Count": rng.uniform(5, 200, n),
                "Size": rng.uniform(50, 5000, n),
            }
        )
    if with_taus:
        data.update(
            {
                "tau_1_true": rng.normal(size=n),
                "tau_2_true": rng.normal(size=n),
            }
        )
    pd.DataFrame(data).to_csv(path, index=False)


@pytest.mark.unit
def test_load_multi_attribution_atomic(tmp_path: Path) -> None:
    csv = tmp_path / "attr.csv"
    _make_dummy_attribution_csv(csv, n=200)
    ds = load_multi_attribution(csv, split="atomic")
    assert ds.x.shape[1] == 8  # 4 binary + 4 cont
    assert ds.t.ndim == 1 and ds.t.max() <= 2
    # atomic 排除了两个都=1 的样本
    assert ds.n < 200
    assert ds.y_potential is not None and ds.y_potential.shape == (ds.n, 3)
    assert ds.cate_true is not None
    assert "tau_2_true" in ds.extra


@pytest.mark.unit
def test_load_multi_attribution_raw_csv_no_taus(tmp_path: Path) -> None:
    """原始 EconML 格式（无 log 列、无真值）：loader 自动 log 变换，跳过真值。"""
    csv = tmp_path / "raw.csv"
    _make_dummy_attribution_csv(csv, n=150, with_taus=False, with_logs=False)
    ds = load_multi_attribution(csv, split="atomic")
    assert ds.x.shape[1] == 8
    assert ds.cate_true is None  # 没有真值
    assert ds.ate_true is None
    # 特征名应反映 log 变换
    assert ds.feature_names is not None
    assert any(name.startswith("log_") for name in ds.feature_names)


@pytest.mark.unit
def test_load_multi_attribution_combined(tmp_path: Path) -> None:
    csv = tmp_path / "attr.csv"
    _make_dummy_attribution_csv(csv, n=200)
    ds = load_multi_attribution(csv, split="combined")
    # combined 仅含两干预都=1 的样本
    assert ds.n > 0
    assert (ds.t == 1).all()


@pytest.mark.unit
def test_load_multi_attribution_all(tmp_path: Path) -> None:
    csv = tmp_path / "attr.csv"
    _make_dummy_attribution_csv(csv, n=100)
    ds = load_multi_attribution(csv, split="all")
    assert ds.n == 100
    assert ds.t.shape == (100, 2)
    assert ds.num_treatments == 2


@pytest.mark.unit
def test_load_multi_attribution_invalid_split(tmp_path: Path) -> None:
    csv = tmp_path / "attr.csv"
    _make_dummy_attribution_csv(csv, n=20)
    with pytest.raises(ValueError, match="split"):
        load_multi_attribution(csv, split="invalid")


@pytest.mark.unit
def test_load_multi_attribution_missing_columns(tmp_path: Path) -> None:
    csv = tmp_path / "bad.csv"
    pd.DataFrame({"foo": [1, 2]}).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="缺少|连续列"):
        load_multi_attribution(csv)
