"""
Tests for the section ML dataset merge (script 14 build_section_dataset).

Synthetic data only — for code validation, not experimental results.
"""

import os
import sys
import importlib.util

import pandas as pd
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))


def _load_script_14():
    path = os.path.join(_ROOT, "scripts", "14_build_section_ml_dataset.py")
    spec = importlib.util.spec_from_file_location("script_14b", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_m = _load_script_14()
_RULE = dict(_m._DEFAULT_RULE)


def _features():
    return pd.DataFrame({
        "sample_id": ["R01_T1_S1", "R01_T1_S2", "R02_T1_S1"],
        "experiment_id": ["R01", "R01", "R02"],
        "track_id": ["T1", "T1", "T1"],
        "section_id": ["S1", "S2", "S1"],
        "section_position_mm": [6.0, 12.0, 6.0],
        "laser_power_W": [300, 300, 320],
        "local_peak_temperature": [1500.0, 1490.0, 1600.0],
    })


def _labels(sample_ids):
    n = len(sample_ids)
    return pd.DataFrame({
        "sample_id": sample_ids,
        "H_mm": [1.0] * n, "W_mm": [5.0] * n, "D_mm": [1.0] * n,
        "theta_left_deg": [40.0] * n, "theta_right_deg": [44.0] * n,
        "defect_presence": [0] * n,
    })


def test_merge_happy_path_adds_labels():
    feats = _features()
    labels = _labels(list(feats["sample_id"]))
    merged, missing_feats = _m.build_section_dataset(feats, labels, _RULE)
    assert len(merged) == 3
    assert missing_feats == []
    # derived + label columns present
    for c in ("dilution_rate", "aspect_ratio", "wetting_angle_avg",
              "wetting_angle_diff", "quality_label"):
        assert c in merged.columns
    # feature columns retained, id columns not duplicated
    assert "local_peak_temperature" in merged.columns
    assert "experiment_id" in merged.columns
    assert "experiment_id_x" not in merged.columns


def test_feature_without_label_raises():
    feats = _features()
    labels = _labels(["R01_T1_S1", "R01_T1_S2"])   # missing R02_T1_S1
    with pytest.raises(ValueError, match="R02_T1_S1"):
        _m.build_section_dataset(feats, labels, _RULE)


def test_extra_label_reported_not_fatal():
    feats = _features()
    labels = _labels(list(feats["sample_id"]) + ["R09_T9_S9"])  # extra label
    merged, missing_feats = _m.build_section_dataset(feats, labels, _RULE)
    assert "R09_T9_S9" in missing_feats
    assert len(merged) == 3   # extra label dropped from the merge


def test_missing_sample_id_column_raises():
    feats = _features().drop(columns=["sample_id"])
    labels = _labels(["R01_T1_S1"])
    with pytest.raises(ValueError, match="sample_id"):
        _m.build_section_dataset(feats, labels, _RULE)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("test_section_ml_dataset: OK (synthetic/code-validation only)")
