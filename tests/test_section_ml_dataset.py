"""
Tests for the section ML dataset merge (script 14 build_section_dataset).

Synthetic data only — for code validation, not experimental results.
"""

import os
import sys
import importlib.util

import numpy as np
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


# --- Required-measurement validation (no auto "Bad" pollution) --------------

def _empty_labels(sample_ids):
    """Label rows that exist but carry NO measured values (template state)."""
    n = len(sample_ids)
    return pd.DataFrame({
        "sample_id": sample_ids,
        "H_mm": [np.nan] * n, "W_mm": [np.nan] * n, "D_mm": [np.nan] * n,
        "theta_left_deg": [np.nan] * n, "theta_right_deg": [np.nan] * n,
        "defect_presence": [np.nan] * n,
    })


def test_empty_measurements_raise_not_labelled_bad():
    feats = _features()
    labels = _empty_labels(list(feats["sample_id"]))
    with pytest.raises(_m.SectionLabelError) as exc:
        _m.build_section_dataset(feats, labels, _RULE)
    msg = str(exc.value)
    assert "Missing required section label values" in msg
    # offending sample_id + every missing required field listed
    assert "sample_id=R01_T1_S1" in msg
    for col in ("H_mm", "W_mm", "D_mm", "theta_left_deg", "theta_right_deg",
                "defect_presence"):
        assert col in msg
    # SectionLabelError is a ValueError subclass (so callers catching ValueError still work)
    assert isinstance(exc.value, ValueError)


def test_blank_string_measurements_also_rejected():
    feats = _features()
    labels = _labels(list(feats["sample_id"]))
    # Emulate a CSV that loaded these columns as text (blank / non-numeric).
    labels["D_mm"] = labels["D_mm"].astype(object)
    labels["theta_left_deg"] = labels["theta_left_deg"].astype(object)
    labels.loc[0, "D_mm"] = ""          # blank string, not numeric
    labels.loc[1, "theta_left_deg"] = "n/a"   # non-numeric text
    with pytest.raises(_m.SectionLabelError) as exc:
        _m.build_section_dataset(feats, labels, _RULE)
    msg = str(exc.value)
    assert "sample_id=R01_T1_S1 missing D_mm" in msg
    assert "sample_id=R01_T1_S2 missing theta_left_deg" in msg
    # the fully-measured row is NOT reported
    assert "R02_T1_S1" not in msg


def test_complete_measurements_autolabel_good_bad():
    feats = _features()
    labels = _labels(list(feats["sample_id"]))   # complete, no quality_label
    merged, _ = _m.build_section_dataset(feats, labels, _RULE)
    assert "quality_label" in merged.columns
    # auto labels only ever Good or Bad, never blank/NaN here
    assert merged["quality_label"].isin(["Good", "Bad"]).all()
    # continuous derived targets are populated (the n=0 bug must not recur)
    for tgt in ("dilution_rate", "aspect_ratio", "wetting_angle_avg",
                "wetting_angle_diff"):
        assert merged[tgt].notna().all()


def test_manual_quality_label_preserved():
    feats = _features()
    labels = _labels(list(feats["sample_id"]))
    # _labels would auto-label all Good; manual values must win.
    labels["quality_label"] = ["Bad", "Good", "Bad"]
    merged, _ = _m.build_section_dataset(feats, labels, _RULE)
    got = dict(zip(merged["sample_id"].astype(str), merged["quality_label"]))
    assert got["R01_T1_S1"] == "Bad"
    assert got["R01_T1_S2"] == "Good"
    assert got["R02_T1_S1"] == "Bad"
    # derived continuous labels still computed alongside the manual label
    assert merged["dilution_rate"].notna().all()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("test_section_ml_dataset: OK (synthetic/code-validation only)")
