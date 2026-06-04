"""
Test StandardNormalizer: fit (train only), transform, inverse_transform,
target helpers, and save/load round-trip.

SIMULATED data only — for code validation, not experimental results.
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from preprocess.normalize import StandardNormalizer


def test_fit_transform_inverse_2d():
    X = np.random.randn(100, 30, 3).astype(np.float32) * 50 + 800
    norm = StandardNormalizer().fit(X)
    z = norm.transform(X)
    # per-channel mean ~0, std ~1
    flat = z.reshape(-1, 3)
    assert np.allclose(flat.mean(axis=0), 0.0, atol=1e-3)
    assert np.allclose(flat.std(axis=0), 1.0, atol=1e-2)
    # inverse recovers original
    assert np.allclose(norm.inverse_transform(z), X, atol=1e-2)


def test_1d_sequence():
    x = np.linspace(200, 1500, 50).astype(np.float32)
    norm = StandardNormalizer().fit(x)
    z = norm.transform(x)
    assert z.shape == x.shape
    assert np.allclose(norm.inverse_transform(z), x, atol=1e-2)


def test_target_roundtrip():
    X = np.random.randn(50, 10, 2).astype(np.float32) * 10 + 500
    norm = StandardNormalizer().fit(X)
    y = np.random.randn(8, 10).astype(np.float32) * 10 + 500   # target ~ channel 0
    yz = norm.transform_target(y)
    assert np.allclose(norm.inverse_transform_target(yz), y, atol=1e-2)


def test_save_load_npz_and_json():
    X = np.random.randn(40, 5, 2).astype(np.float32)
    norm = StandardNormalizer().fit(X)
    with tempfile.TemporaryDirectory() as d:
        for ext in (".npz", ".json"):
            p = os.path.join(d, "norm" + ext)
            norm.save(p)
            loaded = StandardNormalizer.load(p)
            assert np.allclose(loaded.mean_, norm.mean_, atol=1e-5)
            assert np.allclose(loaded.std_, norm.std_, atol=1e-5)


def test_fit_uses_train_only():
    # Sanity: stats come from the array passed to fit (train), independent of
    # any later transform inputs (e.g. test set with a different distribution).
    train = np.ones((10, 4, 1), dtype=np.float32) * 100.0
    norm = StandardNormalizer(eps=1e-6).fit(train)
    assert np.isclose(norm.mean_[0], 100.0)
    # transforming a different "test" distribution does not change the stats
    _ = norm.transform(np.zeros((3, 4, 1), dtype=np.float32))
    assert np.isclose(norm.mean_[0], 100.0)


if __name__ == "__main__":
    for fn in [test_fit_transform_inverse_2d, test_1d_sequence,
               test_target_roundtrip, test_save_load_npz_and_json,
               test_fit_uses_train_only]:
        fn()
    print("test_normalize: OK (simulated/code-validation only)")
