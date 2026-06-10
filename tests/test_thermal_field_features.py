"""
Tests for thermal-field feature extraction.

SIMULATED/synthetic data only — for code validation, not experimental results.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from features.thermal_field_features import (
    compute_high_temperature_mask, compute_high_temperature_area,
    compute_temperature_gradient, compute_thermal_center,
    compute_center_offset, compute_haz_width,
    extract_frame_features, extract_experiment_features,
)

_CONFIG = {
    "thermal_field_features": {
        "high_temp_threshold": 800.0,
        "haz_threshold": 500.0,
        "pixel_size_mm": None,
        "frame_rate_fps": 100.0,
        "center_threshold": 800.0,
        "smooth_window": 3,
    }
}


def _hot_blob_frame(h=20, w=20, hot=1000.0, cold=100.0, r=3):
    """A frame with a hot circular blob centered in the image."""
    frame = np.full((h, w), cold, dtype=np.float32)
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
    frame[mask] = hot
    return frame


def test_high_temperature_mask_and_area():
    frame = _hot_blob_frame()
    mask = compute_high_temperature_mask(frame, 800.0)
    assert mask.dtype == bool
    area_px = compute_high_temperature_area(frame, 800.0)
    assert area_px == float(np.count_nonzero(mask))
    # With pixel size, area scales by pixel_size**2.
    area_mm = compute_high_temperature_area(frame, 800.0, pixel_size_mm=0.5)
    assert np.isclose(area_mm, area_px * 0.25)


def test_thermal_center_is_central_for_centered_blob():
    h = w = 21
    frame = _hot_blob_frame(h, w, r=2)
    cx, cy = compute_thermal_center(frame, threshold=800.0)
    # Centered blob -> center near geometric middle (10, 10).
    assert abs(cx - 10.0) < 1.0
    assert abs(cy - 10.0) < 1.0


def test_center_offset_zero_when_centered():
    h = w = 21
    offset = compute_center_offset(10.0, 10.0, w, h)
    assert np.isclose(offset, 0.0, atol=1e-6)
    # Off-center thermal center -> positive offset.
    off2 = compute_center_offset(15.0, 10.0, w, h)
    assert off2 > 0.0


def test_gradient_stats_keys():
    frame = _hot_blob_frame()
    g = compute_temperature_gradient(frame)
    assert set(g) == {"mean_gradient", "max_gradient", "std_gradient"}
    assert g["max_gradient"] >= g["mean_gradient"] >= 0.0


def test_haz_width_positive_with_hot_region():
    frame = _hot_blob_frame(r=4)
    w_haz = compute_haz_width(frame, 500.0, axis="x")
    assert w_haz > 0.0
    # No pixels above a very high threshold -> width 0.
    assert compute_haz_width(frame, 5000.0, axis="x") == 0.0


def test_extract_frame_features_keys():
    frame = _hot_blob_frame()
    feats = extract_frame_features(frame, _CONFIG)
    for k in ("tmax", "tavg", "high_temp_area", "mean_gradient", "max_gradient",
              "haz_width", "thermal_center_x", "thermal_center_y", "center_offset"):
        assert k in feats


def test_extract_experiment_features_single_row():
    # N x H x W stack of hot-blob frames -> one feature row.
    frames = np.stack([_hot_blob_frame() for _ in range(10)], axis=0)
    feats = extract_experiment_features(frames, frame_rate_fps=100.0, config=_CONFIG)
    for k in ("peak_temperature", "mean_temperature", "max_high_temp_area",
              "mean_high_temp_area", "max_gradient", "mean_gradient",
              "max_cooling_rate", "mean_cooling_rate", "haz_width_max",
              "haz_width_mean", "temperature_fluctuation", "center_offset_mean",
              "center_offset_max", "center_offset_std",
              "dwell_time_above_threshold", "temperature_auc"):
        assert k in feats, f"missing {k}"
    assert np.isclose(feats["peak_temperature"], 1000.0)
    assert feats["dwell_time_above_threshold"] > 0.0     # tmax (1000) >= 800
    # all values are plain floats (one row)
    assert all(isinstance(v, float) for v in feats.values())


def test_extract_experiment_features_rejects_non_3d():
    bad = np.zeros((20, 20), dtype=np.float32)            # 2-D, not N x H x W
    try:
        extract_experiment_features(bad, frame_rate_fps=100.0, config=_CONFIG)
        raise AssertionError("expected ValueError for non-3D input")
    except ValueError as e:
        assert "N x H x W" in str(e)


def test_cooling_rate_sign_on_decaying_stack():
    # Frames whose peak temperature decreases over time -> negative cooling rate.
    h = w = 10
    frames = []
    for i in range(8):
        f = np.full((h, w), 100.0, dtype=np.float32)
        f[4:6, 4:6] = 1000.0 - i * 50.0      # cooling hot spot
        frames.append(f)
    stack = np.stack(frames, axis=0)
    feats = extract_experiment_features(stack, frame_rate_fps=100.0, config=_CONFIG)
    assert feats["max_cooling_rate"] <= 0.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("test_thermal_field_features: OK (synthetic/code-validation only)")
