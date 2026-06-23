"""
Tests for the formal physical calibration + confirmed process metadata.

Config/loader tests always run (committed configs). Tests that need the LOCAL
experiment_master.csv (git-ignored) skip when it is absent.
"""

import os
import sys
import csv
import inspect

import yaml
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from config import physical_calibration as pcmod
from config.physical_calibration import load_physical_calibration, CalibrationError

CAL_FILE = os.path.join(_ROOT, "configs", "physical_calibration.yaml")
EXP_FILE = os.path.join(_ROOT, "configs", "experiments.yaml")
MASTER = os.path.join(_ROOT, "data", "metadata", "experiment_master.csv")

PIXEL = 0.0332889481
AREA = 0.0011081541
TOL = 1e-9


def _yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _master_rows():
    if not os.path.isfile(MASTER):
        pytest.skip("local experiment_master.csv not present")
    with open(MASTER, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# 1 -------------------------------------------------------------------------
def test_pixel_size_equals_5_over_150p2():
    assert abs(5.0 / 150.2 - PIXEL) < TOL


# 2 -------------------------------------------------------------------------
def test_pixel_area_equals_pixel_size_squared():
    assert abs(PIXEL * PIXEL - AREA) < TOL


# 3 -------------------------------------------------------------------------
def test_formal_file_uses_new_pixel_size():
    cal = load_physical_calibration(CAL_FILE)
    assert abs(cal.pixel_size_x_mm - PIXEL) < TOL
    assert abs(cal.pixel_size_y_mm - PIXEL) < TOL
    assert abs(cal.pixel_area_mm2 - AREA) < TOL


# 4 -------------------------------------------------------------------------
def test_formal_mode_does_not_enable_legacy():
    cfg = _yaml(CAL_FILE)
    assert cfg["legacy_calibration"]["enabled_for_formal_processing"] is False
    cal = load_physical_calibration(CAL_FILE)        # must load without error
    assert abs(cal.pixel_size_x_mm - 0.03128) > 1e-3  # NOT the legacy value
    # if legacy were enabled, formal load must raise
    bad = dict(cfg)
    bad["legacy_calibration"] = dict(cfg["legacy_calibration"],
                                     enabled_for_formal_processing=True)
    with pytest.raises(CalibrationError):
        pcmod._validate(bad, CAL_FILE, formal=True)


# 5 -------------------------------------------------------------------------
def test_legacy_value_marked_legacy():
    cfg = _yaml(CAL_FILE)
    lg = cfg["legacy_calibration"]
    assert lg["status"] == "legacy_pilot_only"
    assert abs(float(lg["approximate_pixel_size_mm"]) - 0.03128) < 1e-9
    # default.yaml still carries the legacy value but it is NOT the formal source
    dflt = _yaml(os.path.join(_ROOT, "configs", "default.yaml"))
    assert "calibration" in dflt
    assert dflt["calibration"]["config_file"] == "configs/physical_calibration.yaml"


# 6 -------------------------------------------------------------------------
def test_x_y_and_area_consistent():
    cal = load_physical_calibration(CAL_FILE)
    assert abs(cal.pixel_size_x_mm - cal.pixel_size_y_mm) < TOL  # isotropic
    assert abs(cal.pixel_area_mm2 - cal.pixel_size_x_mm * cal.pixel_size_y_mm) < 1e-6 * AREA


# 7 -------------------------------------------------------------------------
def test_19_conditions_confirmed():
    exp = _yaml(EXP_FILE)
    assert exp["process_parameters"]["source"] == "user_confirmed_response_surface_plan"
    assert exp["process_parameters"]["status"] == "confirmed"
    assert len(exp["conditions"]) == 19


# 8 -------------------------------------------------------------------------
def test_57_tracks_have_complete_process_params():
    rows = _master_rows()
    assert len(rows) == 57
    for r in rows:
        assert r["process_parameter_source"] == "user_confirmed_response_surface_plan"
        assert r["process_parameter_status"] == "confirmed"
        for k in ("laser_power_W", "scan_speed_mm_min", "magnetic_field_mT",
                  "powder_feed_g_min", "travel_distance_mm"):
            assert str(r[k]).strip() != "", (r["sample_id"], k)


# 9 -------------------------------------------------------------------------
def test_57_tracks_reference_formal_calibration():
    rows = _master_rows()
    for r in rows:
        assert r["calibration_id"] == "formal_150p2px_5mm"
        assert r["calibration_status"] == "confirmed"
        assert abs(float(r["pixel_size_x_mm"]) - PIXEL) < TOL
        assert abs(float(r["pixel_size_y_mm"]) - PIXEL) < TOL
        assert abs(float(r["pixel_area_mm2"]) - AREA) < TOL
        assert str(r["isotropic_scaling_assumed"]).lower() == "true"


# 10 ------------------------------------------------------------------------
def test_t1_t2_t3_process_params_consistent():
    rows = _master_rows()
    by_cond = {}
    for r in rows:
        by_cond.setdefault(r["condition_id"], []).append(r)
    assert len(by_cond) == 19
    for cond, grp in by_cond.items():
        assert len(grp) == 3
        keys = {(g["laser_power_W"], g["scan_speed_mm_min"], g["magnetic_field_mT"],
                 g["powder_feed_g_min"], g["travel_distance_mm"]) for g in grp}
        assert len(keys) == 1, cond


# 11 ------------------------------------------------------------------------
def test_frame_rate_stays_empty():
    cal = load_physical_calibration(CAL_FILE)
    assert cal.frame_rate_fps is None
    assert cal.has_frame_rate() is False
    rows = _master_rows()
    assert all(str(r["frame_rate_fps"]).strip() == "" for r in rows)


# 12 ------------------------------------------------------------------------
def test_scan_axis_and_direction_stay_empty():
    cal = load_physical_calibration(CAL_FILE)
    assert cal.scan_axis is None
    assert cal.scan_direction is None
    rows = _master_rows()
    assert all(str(r["scan_axis"]).strip() == "" for r in rows)
    assert all(str(r["scan_direction"]).strip() == "" for r in rows)


# 13 ------------------------------------------------------------------------
def test_time_feature_request_fails_without_frame_rate():
    cal = load_physical_calibration(CAL_FILE)
    with pytest.raises(CalibrationError):
        cal.require_frame_rate()


# 14 ------------------------------------------------------------------------
def test_calibration_is_pure_config_reader_no_feature_extraction():
    """The calibration entry point must not perform ROI / feature extraction
    or write arrays — it is a config reader only."""
    src = inspect.getsource(pcmod)
    # no numerical array work, no feature/ROI extraction code, no array writes
    assert "import numpy" not in src
    assert "np.save" not in src
    assert "from features" not in src and "import features" not in src
    assert "extract_experiment_features" not in src
    assert "extract_local_section_features" not in src
    assert "extract_roi" not in src


# 15 ------------------------------------------------------------------------
def test_loading_calibration_is_read_only():
    """Loading must not modify the calibration file or touch converted matrices."""
    before = os.path.getmtime(CAL_FILE)
    load_physical_calibration(CAL_FILE)
    assert os.path.getmtime(CAL_FILE) == before
    # no matrix files are created/removed by a load (count stable if dir exists)
    mdir = os.path.join(_ROOT, "data", "processed", "matrix")
    if os.path.isdir(mdir):
        n = len([x for x in os.listdir(mdir) if x.endswith(".npy")])
        load_physical_calibration(CAL_FILE)
        assert len([x for x in os.listdir(mdir) if x.endswith(".npy")]) == n


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
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
