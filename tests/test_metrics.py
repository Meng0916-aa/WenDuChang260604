"""
Test RMSE, MAE, waveform_similarity, and grouped metrics.

SIMULATED data only — for code validation, not experimental results.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils.metrics import (
    rmse, mae, waveform_similarity, compute_all_metrics, compute_grouped_metrics,
)


def test_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    assert np.isclose(rmse(y, y), 0.0)
    assert np.isclose(mae(y, y), 0.0)
    assert np.isclose(waveform_similarity(y, y), 1.0)


def test_rmse_mae_values():
    yt = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    yp = np.array([3.0, 0.0, 0.0], dtype=np.float32)
    assert np.isclose(rmse(yt, yp), np.sqrt(3.0))   # sqrt(9/3)
    assert np.isclose(mae(yt, yp), 1.0)             # 3/3


def test_waveform_similarity_in_range():
    yt = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float32)
    yp = -yt                                         # anti-correlated
    s = waveform_similarity(yt, yp)
    assert 0.0 <= s <= 1.0
    assert s < 0.5                                   # negative correlation -> low


def test_compute_all_keys():
    yt = np.random.randn(10).astype(np.float32)
    yp = yt + 0.1
    m = compute_all_metrics(yt, yp)
    assert set(m.keys()) == {"rmse", "mae", "waveform_similarity"}


def test_grouped_metrics():
    yt = np.random.randn(6, 2).astype(np.float32)
    yp = yt + 0.5
    labels = ["a", "a", "a", "b", "b", "b"]
    g = compute_grouped_metrics(yt, yp, labels)
    assert set(g.keys()) == {"a", "b"}
    assert g["a"]["sample_count"] == 3
    assert g["b"]["sample_count"] == 3


if __name__ == "__main__":
    for fn in [test_perfect_prediction, test_rmse_mae_values,
               test_waveform_similarity_in_range, test_compute_all_keys,
               test_grouped_metrics]:
        fn()
    print("test_metrics: OK (simulated/code-validation only)")
