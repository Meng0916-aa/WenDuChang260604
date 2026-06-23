"""
Tests for the formal metadata / measurement-configuration FREEZE.

Covers the 25 verification points of the parameter-freeze task:
camera/optics, image geometry & scan direction, frame rate & effective-frame
rule, temperature measurement range + four-state policy, substrate / logical
plate mapping, track repetition & cooling, and the 57-row master completeness.

Config/loader tests always run (committed configs). Tests that need the LOCAL
experiment_master.csv (git-ignored) skip when it is absent. No ROI / feature
extraction / matrix access happens here.
"""

import os
import sys
import csv

import yaml
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from config.physical_calibration import load_physical_calibration

CAL_FILE = os.path.join(_ROOT, "configs", "physical_calibration.yaml")
EXP_FILE = os.path.join(_ROOT, "configs", "experiments.yaml")
MASTER = os.path.join(_ROOT, "data", "metadata", "experiment_master.csv")
BUILD_SCRIPT = os.path.join(_ROOT, "scripts", "00_build_experiment_master.py")
AUDIT_SCRIPT = os.path.join(_ROOT, "scripts", "00b_build_metadata_audit.py")

TOL = 1e-9


def _yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _exp():
    return _yaml(EXP_FILE)


def _calib():
    return load_physical_calibration(CAL_FILE)


def _master_rows():
    if not os.path.isfile(MASTER):
        pytest.skip("local experiment_master.csv not present")
    with open(MASTER, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# === Image geometry & scan direction (points 1-4) =========================

def test_point01_scan_axis_is_y():
    assert _calib().scan_axis == "y"
    assert _calib().transverse_axis == "x"


def test_point02_image_scan_direction_upward():
    assert _calib().image_scan_direction == "upward"


def test_point03_array_scan_direction_decreasing():
    assert _calib().array_scan_direction == "decreasing_row_index"


def test_point04_physical_to_array_y_sign_is_minus_one():
    assert _calib().physical_to_array_y_sign == -1


def test_geometry_has_no_transform():
    g = _yaml(CAL_FILE)["image_geometry"]
    assert g["rotation_deg"] == 0
    assert g["flip_horizontal"] is False
    assert g["flip_vertical"] is False
    assert g["image_height_px"] == 512 and g["image_width_px"] == 640


# === Frame rate & effective-frame rule (points 5-7) =======================

def test_point05_frame_rate_is_52():
    cal = _calib()
    assert cal.frame_rate_fps == 52.0
    acq = _yaml(CAL_FILE)["acquisition"]
    assert acq["frame_rate_status"] == "confirmed"
    assert acq["frame_rate_source"] == "user_confirmed_experimental_setting"


def test_point06_effective_start_is_python_index_1():
    rule = _calib().effective_frame_rule
    assert rule["start_frame_number_1_based"] == 2
    assert rule["start_frame_index_0_based"] == 1
    assert rule["end_rule"] == "last_available_frame"


def test_point07_first_frame_excluded_as_startup():
    rule = _calib().effective_frame_rule
    assert rule["excluded_initial_frame_count"] == 1
    assert rule["excluded_initial_frame_reason"] == "camera_startup_frame"


# === Temperature measurement range + four-state policy (points 8-11) ======

def test_point08_valid_range_300_1800():
    cal = _calib()
    assert cal.valid_temperature_min_C == 300.0
    assert cal.valid_temperature_max_C == 1800.0
    assert cal.measurement_range_status == "confirmed"


def _classify(temp_C, cal):
    """Mirror the documented four-state policy from the config thresholds.

    This validates the FROZEN policy values; it does not run feature extraction
    or touch any matrix.
    """
    if temp_C < cal.valid_temperature_min_C:
        return "below_range"
    if temp_C <= cal.valid_temperature_max_C:
        return "valid"
    if temp_C < cal.hard_saturation_threshold_C:
        return "above_range"
    return "hard_saturation"


def test_point09_above_range_band():
    cal = _calib()
    # 1800 < T < 6500 -> above_range (these are NOT real temperatures)
    for t in (1800.1, 2739.0, 3085.0, 3700.0, 6499.9):
        assert _classify(t, cal) == "above_range", t
    # in-band stays valid
    assert _classify(1800.0, cal) == "valid"
    assert _classify(300.0, cal) == "valid"
    assert _classify(299.9, cal) == "below_range"


def test_point10_hard_saturation_band():
    cal = _calib()
    assert cal.hard_saturation_threshold_C == 6500.0
    assert _classify(6500.0, cal) == "hard_saturation"
    assert _classify(6553.5, cal) == "hard_saturation"   # uint16 ceiling sentinel
    assert cal.saturation_value_C == 6553.5


def test_point11_raw_matrix_preserved_policy():
    pol = _calib().analysis_policy
    assert pol["preserve_raw_matrix"] is True
    assert pol["interpolate_invalid_pixels"] is False
    assert pol["robust_peak_quantile"] == 0.999
    for k in ("below_range_policy", "above_range_policy", "hard_saturation_policy"):
        assert pol[k] == "mask_and_report"


# === Camera / optics (points 12-16) =======================================

def test_point12_working_distance_is_300_not_30():
    cal = _calib()
    assert cal.working_distance_mm == 300.0
    assert cal.working_distance_mm != 30.0


def test_point13_working_distance_reference():
    assert _calib().working_distance_reference == "protective_window_to_molten_pool"


def test_point14_defocus_is_plus_14():
    fx = _exp()["fixed_process_parameters"]
    assert fx["defocus_mm"] == 14.0
    assert fx["defocus_sign"] == "positive"


def test_point15_positive_defocus_definition():
    fx = _exp()["fixed_process_parameters"]
    assert fx["positive_defocus_definition"] == "focal_plane_above_substrate_surface"


def test_point16_laser_spot_diameter_is_1mm():
    assert _exp()["fixed_process_parameters"]["laser_spot_diameter_mm"] == 1.0


# === Powder feed (point 17) ===============================================

def test_point17_powder_feed_set_not_actual():
    pf = _exp()["powder_feed"]
    assert pf["powder_feed_set_g_min"] == 40.0
    assert pf["powder_feed_actual_g_min"] is None
    assert pf["powder_feed_actual_status"] == "not_measured"
    assert pf["powder_feed_source"] == "equipment_setpoint"


# === Substrate / logical plate (points 18-21) =============================

def test_point18_19_logical_plate_ids_unique():
    sub = _exp()["substrate"]
    assert sub["plate_count"] == 19
    assert sub["one_plate_per_condition"] is True
    assert sub["plate_id_type"] == "logical_condition_based_identifier"
    rows = _master_rows()
    plate_ids = {r["plate_id"] for r in rows}
    assert len(plate_ids) == 19
    # one plate_id per condition, exactly Plate-<condition_id>
    for r in rows:
        assert r["plate_id"] == f"Plate-{r['condition_id']}"


def test_point19_each_plate_has_three_tracks():
    rows = _master_rows()
    by_plate = {}
    for r in rows:
        by_plate.setdefault(r["plate_id"], []).append(r["track_id"])
    assert len(by_plate) == 19
    for p, tracks in by_plate.items():
        assert sorted(tracks) == ["T1", "T2", "T3"], p


def test_point20_all_substrate_316L():
    assert _exp()["substrate"]["material"] == "316L"
    rows = _master_rows()
    assert all(r["substrate_material"] == "316L" for r in rows)


def test_point21_all_substrate_40_16_8():
    sub = _exp()["substrate"]
    assert (sub["length_mm"], sub["width_mm"], sub["thickness_mm"]) == (40.0, 16.0, 8.0)
    rows = _master_rows()
    for r in rows:
        assert float(r["substrate_length_mm"]) == 40.0
        assert float(r["substrate_width_mm"]) == 16.0
        assert float(r["substrate_thickness_mm"]) == 8.0


# === Track repetition & cooling (points 22-23) ============================

def test_point22_track_order_1_2_3():
    rep = _exp()["track_repetition"]
    assert rep["track_order"] == {"T1": 1, "T2": 2, "T3": 3}
    rows = _master_rows()
    order = {r["track_id"]: int(r["track_order"]) for r in rows}
    assert order["T1"] == 1 and order["T2"] == 2 and order["T3"] == 3


def test_point23_cooling_interval_120s():
    rep = _exp()["track_repetition"]
    assert rep["cooling_interval_between_tracks_s"] == 120.0
    assert rep["cooling_interval_status"] == "confirmed"


# === 57-row master completeness (point 24) ================================

def test_point24_master_has_57_complete_rows():
    rows = _master_rows()
    assert len(rows) == 57
    assert len({r["sample_id"] for r in rows}) == 57
    # Core confirmed fields must be present (non-blank) on every row.
    required = [
        "scan_axis", "transverse_axis", "image_scan_direction",
        "array_scan_direction", "physical_to_array_y_sign",
        "frame_rate_fps", "effective_start_frame_index_0_based",
        "valid_temperature_min_C", "valid_temperature_max_C",
        "working_distance_mm", "working_distance_reference",
        "laser_spot_diameter_mm", "defocus_mm", "defocus_sign",
        "cooling_interval_between_tracks_s", "plate_id", "plate_id_type",
        "substrate_material", "substrate_length_mm", "substrate_width_mm",
        "substrate_thickness_mm", "powder_feed_set_g_min",
        "calibration_reference_axis", "pixel_size_x_mm", "pixel_size_y_mm",
    ]
    for r in rows:
        for k in required:
            assert str(r[k]).strip() != "", (r["sample_id"], k)
    # Intentionally blank (not recorded / not measured).
    for r in rows:
        assert str(r["powder_feed_actual_g_min"]).strip() == ""
        assert str(r["emissivity"]).strip() == ""
        assert str(r["transmission"]).strip() == ""


def test_master_working_distance_and_frame_rate_consistent():
    rows = _master_rows()
    assert all(abs(float(r["working_distance_mm"]) - 300.0) < TOL for r in rows)
    assert all(abs(float(r["frame_rate_fps"]) - 52.0) < TOL for r in rows)
    assert all(abs(float(r["defocus_mm"]) - 14.0) < TOL for r in rows)
    assert all(abs(float(r["powder_feed_set_g_min"]) - 40.0) < TOL for r in rows)


# === Radiometric inputs still not recorded ================================

def test_emissivity_and_transmission_not_recorded():
    temp = _yaml(CAL_FILE)["temperature_calibration"]
    assert temp["emissivity"] is None
    assert temp["transmission"] is None
    assert temp["emissivity_status"] == "not_recorded"
    assert temp["transmission_status"] == "not_recorded"


# === Point 25: this freeze runs NO ROI / feature extraction ===============

def test_point25_freeze_scripts_are_metadata_only():
    """The freeze scripts must not import numpy, write arrays, or run ROI /
    feature extraction — they only read configs and write metadata CSVs."""
    for path in (BUILD_SCRIPT, AUDIT_SCRIPT):
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
        assert "import numpy" not in src, path
        assert "np.save" not in src, path
        assert "from features" not in src and "import features" not in src, path
        assert "extract_experiment_features" not in src, path
        assert "extract_local_section_features" not in src, path
        assert "data/processed/roi" not in src, path
        assert "data/processed/matrix" not in src, path


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    ok = 0
    for fn in fns:
        try:
            fn()
            ok += 1
            print(f"  ok: {fn.__name__}")
        except Exception:
            print(f"  FAIL: {fn.__name__}")
            traceback.print_exc()
    print(f"{ok}/{len(fns)} passed")
