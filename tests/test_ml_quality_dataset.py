"""
Tests for quality-label formulas and the ML dataset merge logic (script 11).

SIMULATED/synthetic data only — for code validation, not experimental results.
"""

import os
import sys
import tempfile
import importlib.util

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))


# --- Label formulas (as defined in docs/quality_label_template.md) ----------

def dilution_rate(D, H):
    return D / (D + H)


def aspect_ratio(W, H):
    return W / H


def wetting_angle_avg(left, right):
    return (left + right) / 2.0


def quality_label(dr, ar, wa, rule):
    good = (rule["dilution_rate_min"] <= dr <= rule["dilution_rate_max"] and
            rule["aspect_ratio_min"] <= ar <= rule["aspect_ratio_max"] and
            rule["wetting_angle_min"] <= wa <= rule["wetting_angle_max"])
    return "Good" if good else "Bad"


_RULE = {
    "dilution_rate_min": 0.30, "dilution_rate_max": 0.50,
    "aspect_ratio_min": 3.0, "aspect_ratio_max": 6.0,
    "wetting_angle_min": 30.0, "wetting_angle_max": 55.0,
}


def test_dilution_rate_formula():
    assert np.isclose(dilution_rate(D=1.0, H=1.0), 0.5)
    assert np.isclose(dilution_rate(D=0.9, H=1.2), 0.9 / 2.1)


def test_aspect_ratio_formula():
    assert np.isclose(aspect_ratio(W=5.0, H=1.0), 5.0)
    assert np.isclose(aspect_ratio(W=4.8, H=1.2), 4.0)


def test_wetting_angle_avg_formula():
    assert np.isclose(wetting_angle_avg(42, 48), 45.0)


def test_quality_rule_good_and_bad():
    # In-range -> Good
    assert quality_label(0.40, 4.0, 45.0, _RULE) == "Good"
    # High dilution -> Bad
    assert quality_label(0.60, 4.0, 45.0, _RULE) == "Bad"
    # Aspect ratio too low -> Bad
    assert quality_label(0.40, 2.0, 45.0, _RULE) == "Bad"
    # Wetting angle too high -> Bad
    assert quality_label(0.40, 4.0, 70.0, _RULE) == "Bad"


# --- Script 11 helpers ------------------------------------------------------

def _load_script_11():
    path = os.path.join(_ROOT, "scripts", "11_build_ml_quality_dataset.py")
    spec = importlib.util.spec_from_file_location("script_11", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_load_labels_prefers_csv():
    m = _load_script_11()
    with tempfile.TemporaryDirectory() as d:
        csv_path = os.path.join(d, "quality_labels.csv")
        pd.DataFrame({"sample_id": ["B0_01"], "quality_label": ["Good"]}).to_csv(
            csv_path, index=False)
        qcfg = {"label_file_csv": csv_path,
                "label_file_excel": os.path.join(d, "nope.xlsx")}
        labels, src = m._load_labels(qcfg)
        assert labels is not None and src == csv_path
        assert "sample_id" in labels.columns


def test_load_labels_missing_returns_none():
    m = _load_script_11()
    qcfg = {"label_file_csv": "/no/such.csv", "label_file_excel": "/no/such.xlsx"}
    labels, src = m._load_labels(qcfg)
    assert labels is None and src is None


def test_merge_reconciliation_detects_mismatch():
    # Simulate the reconciliation logic of script 11 directly.
    feat_ids = {"B0_01", "B0_02", "B100_01"}
    label_ids = {"B0_01", "B0_02"}                    # B100_01 has no label
    missing_labels = sorted(feat_ids - label_ids)
    missing_feats = sorted(label_ids - feat_ids)
    assert missing_labels == ["B100_01"]
    assert missing_feats == []


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("test_ml_quality_dataset: OK (synthetic/code-validation only)")
