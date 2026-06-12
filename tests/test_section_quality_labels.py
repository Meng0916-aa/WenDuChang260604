"""
Tests for section quality-label derivation (script 14 helpers).

Synthetic data only — for code validation, not experimental results.
"""

import os
import sys
import importlib.util

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))


def _load_script_14():
    path = os.path.join(_ROOT, "scripts", "14_build_section_ml_dataset.py")
    spec = importlib.util.spec_from_file_location("script_14", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_m = _load_script_14()
_RULE = dict(_m._DEFAULT_RULE)


def test_derived_columns_computed():
    df = pd.DataFrame({
        "sample_id": ["R01_T1_S1", "R01_T1_S2"],
        "H_mm": [1.0, 2.0],
        "W_mm": [5.0, 8.0],
        "D_mm": [1.0, 1.0],
        "theta_left_deg": [40.0, 50.0],
        "theta_right_deg": [44.0, 40.0],
    })
    out = _m.compute_derived_labels(df)
    # dilution_rate is a PERCENT: D/(D+H)*100
    assert np.isclose(out.loc[0, "dilution_rate"], 1.0 / 2.0 * 100.0)  # 50.0
    assert np.isclose(out.loc[1, "dilution_rate"], 1.0 / 3.0 * 100.0)
    assert np.isclose(out.loc[0, "aspect_ratio"], 5.0)
    assert np.isclose(out.loc[1, "aspect_ratio"], 4.0)
    assert np.isclose(out.loc[0, "wetting_angle_avg"], 42.0)
    assert np.isclose(out.loc[0, "wetting_angle_diff"], 4.0)
    assert np.isclose(out.loc[1, "wetting_angle_diff"], 10.0)


def test_user_supplied_values_preserved():
    df = pd.DataFrame({
        "sample_id": ["R01_T1_S1"],
        "H_mm": [1.0], "W_mm": [5.0], "D_mm": [1.0],
        "theta_left_deg": [40.0], "theta_right_deg": [44.0],
        "aspect_ratio": [9.9],   # pre-supplied -> must be kept
    })
    out = _m.compute_derived_labels(df)
    assert np.isclose(out.loc[0, "aspect_ratio"], 9.9)


def test_quality_rule_good_and_bad():
    df = pd.DataFrame({
        # Good: dr=40, ar=4, wa=45, no defect
        # Bad (dilution): dr=60
        # Bad (defect):  good ratios but defect_presence=1
        "dilution_rate": [40.0, 60.0, 40.0],
        "aspect_ratio": [4.0, 4.0, 4.0],
        "wetting_angle_avg": [45.0, 45.0, 45.0],
        "defect_presence": [0, 0, 1],
    })
    labels = _m.apply_quality_rule(df, _RULE).tolist()
    assert labels == ["Good", "Bad", "Bad"]


def test_ensure_quality_label_fills_only_missing():
    df = pd.DataFrame({
        "dilution_rate": [40.0, 40.0],
        "aspect_ratio": [4.0, 4.0],
        "wetting_angle_avg": [45.0, 45.0],
        "defect_presence": [0, 0],
        "quality_label": ["Bad", np.nan],   # row0 user-set, row1 auto
    })
    out = _m.ensure_quality_label(df, _RULE)
    assert out.loc[0, "quality_label"] == "Bad"   # preserved
    assert out.loc[1, "quality_label"] == "Good"  # auto-filled


def test_invalid_inputs_not_autolabelled_bad():
    # No quality_label, and the rule inputs are NaN -> must NOT become "Bad".
    df = pd.DataFrame({
        "dilution_rate": [np.nan],
        "aspect_ratio": [np.nan],
        "wetting_angle_avg": [np.nan],
        "defect_presence": [np.nan],
    })
    out = _m.ensure_quality_label(df, _RULE)
    assert pd.isna(out.loc[0, "quality_label"])


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("test_section_quality_labels: OK (synthetic/code-validation only)")
