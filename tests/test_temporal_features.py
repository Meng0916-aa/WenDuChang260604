"""
Tests for temporal feature analysis.

SIMULATED data only — for code validation, not experimental results.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from features.temporal_features import (
    compute_time_axis, smooth_curve, compute_derivative,
    extract_temporal_features,
)


def _triangle_curve():
    """
    Build a simple heat-up / cool-down triangle at 10 fps:
      ramp 0->100 C over frames 0..10, then 100->0 C over frames 10..20.
    Peak = 100 C at frame 10 -> t = 1.0 s (fps=10).
    """
    up = np.linspace(0.0, 100.0, 11, dtype=np.float32)        # 11 pts incl. peak
    down = np.linspace(100.0, 0.0, 11, dtype=np.float32)[1:]  # drop dup peak
    return np.concatenate([up, down]), 10.0                   # curve, fps


def test_compute_time_axis():
    t = compute_time_axis(5, frame_rate_fps=1000.0)
    assert t.shape == (5,)
    assert np.allclose(t, [0.0, 0.001, 0.002, 0.003, 0.004])
    # empty / invalid
    assert compute_time_axis(0, 100.0).shape == (0,)
    try:
        compute_time_axis(5, 0.0)
        raise AssertionError("expected ValueError for fps=0")
    except ValueError:
        pass


def test_peak_temperature_and_time():
    curve, fps = _triangle_curve()
    feats = extract_temporal_features(curve, frame_rate_fps=fps,
                                      threshold=50.0, smooth_window=1)
    assert np.isclose(feats["peak_temperature"], 100.0)
    assert np.isclose(feats["peak_time"], 1.0)               # frame 10 / 10 fps
    assert np.isclose(feats["min_temperature"], 0.0)


def test_heating_and_cooling_rates():
    curve, fps = _triangle_curve()
    feats = extract_temporal_features(curve, frame_rate_fps=fps, smooth_window=1)
    # Ramp slope = 100 C over 10 frames = 1 s -> +/-100 C/s.
    assert np.isclose(feats["max_heating_rate"], 100.0, atol=1.0)
    assert np.isclose(feats["max_cooling_rate"], -100.0, atol=1.0)
    assert feats["max_heating_rate"] > 0.0
    assert feats["max_cooling_rate"] < 0.0
    assert feats["mean_cooling_rate"] < 0.0


def test_dwell_time_above_threshold():
    curve, fps = _triangle_curve()
    # threshold 95 C: frames with value >= 95. dt = 0.1 s.
    feats = extract_temporal_features(curve, frame_rate_fps=fps,
                                      threshold=95.0, smooth_window=1)
    n_above = int(np.count_nonzero(curve >= 95.0))
    assert np.isclose(feats["dwell_time_above_threshold"], n_above * 0.1)
    # No threshold -> dwell time 0.
    feats0 = extract_temporal_features(curve, frame_rate_fps=fps, threshold=None)
    assert feats0["dwell_time_above_threshold"] == 0.0


def test_fluctuation_index_nonnegative():
    rng = np.random.RandomState(0)
    noisy = (np.linspace(0, 100, 200) + rng.normal(0, 3, 200)).astype(np.float32)
    feats = extract_temporal_features(noisy, frame_rate_fps=100.0,
                                      smooth_window=5)
    assert feats["fluctuation_index"] >= 0.0
    # A perfectly smooth ramp should have ~0 fluctuation.
    smooth_ramp = np.linspace(0, 100, 200).astype(np.float32)
    f2 = extract_temporal_features(smooth_ramp, frame_rate_fps=100.0,
                                   smooth_window=5)
    assert f2["fluctuation_index"] >= 0.0
    assert f2["fluctuation_index"] <= feats["fluctuation_index"] + 1e-6


def test_smooth_curve_passthrough():
    c = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert np.allclose(smooth_curve(c, 1), c)        # window <=1 -> unchanged
    assert np.allclose(smooth_curve(c, 0), c)
    assert smooth_curve(c, 3).shape == c.shape       # length preserved


def test_derivative_length_and_zero_for_short():
    t = compute_time_axis(4, 10.0)
    c = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    d = compute_derivative(c, t)
    assert d.shape == c.shape
    # constant slope 1 C / 0.1 s = 10 C/s
    assert np.allclose(d, 10.0, atol=1e-3)
    # single sample -> zeros
    assert np.allclose(compute_derivative(np.array([5.0]),
                                          compute_time_axis(1, 10.0)), [0.0])


if __name__ == "__main__":
    for fn in [test_compute_time_axis, test_peak_temperature_and_time,
               test_heating_and_cooling_rates, test_dwell_time_above_threshold,
               test_fluctuation_index_nonnegative, test_smooth_curve_passthrough,
               test_derivative_length_and_zero_for_short]:
        fn()
    print("test_temporal_features: OK (simulated/code-validation only)")
