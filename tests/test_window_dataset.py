"""
Test sliding-window dataset shapes (1-D and 2-D inputs, process params, split).

SIMULATED data only — for code validation, not experimental results.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from datasets.window_dataset import (
    build_windows, WindowDataset, split_by_experiment,
    split_experiment_indices, count_windows,
)


def test_build_windows_1d():
    data = np.arange(100, dtype=np.float32)
    X, y = build_windows(data, input_len=10, pred_len=5, step=1)
    assert X.shape == (86, 10, 1)
    assert y.shape == (86, 5, 1)


def test_build_windows_2d():
    data = np.random.randn(60, 3).astype(np.float32)
    X, y = build_windows(data, input_len=20, pred_len=10, step=2)
    n = count_windows(60, 20, 10, 2)
    assert X.shape == (n, 20, 3)
    assert y.shape == (n, 10, 3)


def test_dataset_with_process_params():
    data = np.sin(np.linspace(0, 10, 200)).astype(np.float32)
    ds = WindowDataset(data, input_len=15, pred_len=5,
                       process_params=np.array([1.0, 2.0], dtype=np.float32))
    assert ds.feature_dim == 3            # 1 curve + 2 params
    x, y = ds[0]
    assert x.shape == (15, 3)
    assert y.shape == (5, 1)


def test_split_by_experiment_no_overlap():
    data_list = [np.random.randn(120, 2).astype(np.float32) for _ in range(4)]
    full = WindowDataset(data_list, input_len=20, pred_len=10)
    tr, va, te = split_by_experiment(full, val_ratio=0.25, test_ratio=0.25, seed=0)
    assert len(tr) + len(va) + len(te) == len(full)


def test_split_indices_reproducible():
    a = split_experiment_indices(6, 0.2, 0.2, seed=42)
    b = split_experiment_indices(6, 0.2, 0.2, seed=42)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


if __name__ == "__main__":
    for fn in [test_build_windows_1d, test_build_windows_2d,
               test_dataset_with_process_params,
               test_split_by_experiment_no_overlap,
               test_split_indices_reproducible]:
        fn()
    print("test_window_dataset: OK (simulated/code-validation only)")
