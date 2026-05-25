"""IHDP（Infant Health and Development Program）数据加载。

标准基准（Hill 2011），747 个样本 × 25 个特征 × 100 个模拟切片。每个切片包含：
    x  (n, d, 100)   协变量
    t  (n, 100)      二元干预
    yf (n, 100)      factual 结果
    mu0, mu1 (n, 100) 真实潜在结果均值

仓库示例使用项目目录下的 data/ihdp/：
    ihdp_npci_1-100.train.npz
    ihdp_npci_1-100.test.npz
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from causalmt.data.base import CausalDataset

__all__ = ["load_ihdp"]


def load_ihdp(
    data_dir: str | Path,
    *,
    slice_idx: int = 0,
    train_filename: str = "ihdp_npci_1-100.train.npz",
    test_filename: str = "ihdp_npci_1-100.test.npz",
) -> tuple[CausalDataset, CausalDataset]:
    """加载 IHDP 100 切片基准数据集的指定切片。

    Args:
        data_dir: 包含 train/test npz 文件的目录
        slice_idx: 0..99 选择切片索引
        train_filename / test_filename: 自定义文件名

    Returns:
        (train_dataset, test_dataset)，每个含 x/t/y、真实潜在结果 (mu0, mu1) 与 CATE
    """
    if not 0 <= slice_idx < 100:
        raise ValueError(f"slice_idx 应在 [0, 99]，得 {slice_idx}")

    data_dir = Path(data_dir).expanduser()
    train_path = data_dir / train_filename
    test_path = data_dir / test_filename
    if not train_path.exists():
        raise FileNotFoundError(f"IHDP 训练文件不存在: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"IHDP 测试文件不存在: {test_path}")

    train = _build_split(np.load(train_path), slice_idx, name=f"ihdp_train_slice{slice_idx}")
    test = _build_split(np.load(test_path), slice_idx, name=f"ihdp_test_slice{slice_idx}")
    return train, test


def _build_split(npz: np.lib.npyio.NpzFile, slice_idx: int, *, name: str) -> CausalDataset:
    """从 (n, d, 100) / (n, 100) 张量中切出第 slice_idx 个数据集。"""
    x = npz["x"][..., slice_idx].astype(np.float32)
    t = npz["t"][:, slice_idx].astype(np.int64)
    y = npz["yf"][:, slice_idx].astype(np.float32)
    mu0 = npz["mu0"][:, slice_idx].astype(np.float32)
    mu1 = npz["mu1"][:, slice_idx].astype(np.float32)

    y_potential = np.stack([mu0, mu1], axis=1)  # (n, 2)
    cate_true = (mu1 - mu0).astype(np.float32)
    ate_true = float(cate_true.mean())

    return CausalDataset(
        x=x,
        t=t,
        y=y,
        y_potential=y_potential,
        cate_true=cate_true,
        ate_true=ate_true,
        name=name,
    )
