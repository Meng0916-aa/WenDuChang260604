"""
Tests for src/features/local_section_features.py.

Synthetic ROI matrices only — for code validation, not experimental results.
"""

import os
import sys
import warnings

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from features.local_section_features import (
    compute_frame_window, extract_local_section_features, LOCAL_FEATURE_KEYS,
)

# Minimal config exercising the thermal_field_features thresholds.
_CONFIG = {
    "thermal_field_features": {
        "high_temp_threshold": 800.0,
        "haz_threshold": 500.0,
        "center_threshold": 800.0,
        "pixel_size_mm": None,
        "smooth_window": 3,
        "frame_rate_fps": 1000,
    }
}


def _fake_roi(n=10, h=4, w=5):
    """N x H x W ramp so frames differ (peak rises with frame index)."""
    rng = np.random.RandomState(0)
    base = rng.uniform(0, 100, size=(n, h, w)).astype(np.float32)
    # add a hot spot whose temperature grows with frame index
    for i in range(n):
        base[i, h // 2, w // 2] += 200.0 + 90.0 * i
    return base


def test_frame_center_mid_track():
    # section at 15 mm of a 30 mm track over N=10 -> center near 4 or 5
    fc, fs, fe = compute_frame_window(15.0, 30.0, 10, window_half_frames=2)
    assert fc in (4, 5)
    assert fs == max(0, fc - 2)
    assert fe == min(9, fc + 2)


def test_window_bounds_clamped_at_edges():
    # start of track -> center 0, window clamps to [0, 2]
    fc, fs, fe = compute_frame_window(0.0, 30.0, 10, window_half_frames=2)
    assert (fc, fs, fe) == (0, 0, 2)
    # end of track -> center 9, window clamps to [7, 9]
    fc, fs, fe = compute_frame_window(30.0, 30.0, 10, window_half_frames=2)
    assert (fc, fs, fe) == (9, 7, 9)


def test_position_out_of_range_raises():
    with pytest.raises(ValueError):
        compute_frame_window(31.0, 30.0, 10, window_half_frames=2)
    with pytest.raises(ValueError):
        compute_frame_window(-1.0, 30.0, 10, window_half_frames=2)


def test_bad_travel_distance_raises():
    with pytest.raises(ValueError):
        compute_frame_window(5.0, 0.0, 10, window_half_frames=2)


def test_local_features_present_and_finite():
    roi = _fake_roi()
    feats = extract_local_section_features(
        roi, section_position_mm=15.0, travel_distance_mm=30.0,
        window_half_frames=2, frame_rate_fps=1000, config=_CONFIG)
    for key in LOCAL_FEATURE_KEYS:
        assert key in feats, f"missing {key}"
        assert np.isfinite(feats[key])
    # bookkeeping fields
    assert feats["frame_start"] <= feats["frame_center"] <= feats["frame_end"]
    assert feats["window_frame_count"] == feats["frame_end"] - feats["frame_start"] + 1
    # peak temperature should reflect the injected hot spot in the window
    assert feats["local_peak_temperature"] > 200.0
    assert feats["local_cooling_rate_max_abs"] >= 0.0


def test_small_window_warns_but_returns():
    roi = _fake_roi()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        feats = extract_local_section_features(
            roi, section_position_mm=15.0, travel_distance_mm=30.0,
            window_half_frames=0, frame_rate_fps=1000, config=_CONFIG)
    assert feats["window_frame_count"] == 1
    assert any("window" in str(wi.message).lower() for wi in w)


def test_non_3d_roi_raises():
    with pytest.raises(ValueError):
        extract_local_section_features(
            np.zeros((4, 5), dtype=np.float32), 15.0, 30.0, 2, 1000, _CONFIG)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("test_local_section_features: OK (synthetic/code-validation only)")
