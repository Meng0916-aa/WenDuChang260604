"""
Consistency tests for the FORMAL experiment design.

Validates configs/experiments.yaml against the real 19-condition Box-Behnken
plan (and, when the canonical raw data is present, against the data on disk).

These tests use the REAL formal config — not simulated data. Plain asserts only
(no hard pytest dependency), and a __main__ runner so they also run standalone.
Data-dependent checks are skipped (return early) when the raw-data root is absent
so the config checks still run on machines without the data.
"""

import os
import glob

import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG = os.path.join(_ROOT, "configs", "experiments.yaml")

# --- Formal expectations ---
EXPECTED_CONDITIONS = 19
TRACKS_PER_CONDITION = 3
EXPECTED_TOTAL_TRACKS = 57
EXPECTED_XTHERM_TOTAL = 9439
TRACKS = ["T1", "T2", "T3"]
FIXED_POWDER_FEED = 40
FIXED_TRAVEL_DISTANCE = 30
ALLOWED_ROLES = {
    "control_no_field",
    "control_high_field",
    "box_behnken_edge",
    "box_behnken_center_repeat",
}


def _load():
    with open(_CONFIG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _conditions():
    return _load()["conditions"]


def _data_root():
    """Canonical raw-data root, or None if it does not exist on this machine."""
    root = _load().get("raw_data_root")
    return root if root and os.path.isdir(root) else None


# ---------------- Config-only checks (always run) ----------------

def test_config_loads_and_has_design_block():
    cfg = _load()
    assert "experiment_design" in cfg
    assert "conditions" in cfg
    assert "raw_data_root" in cfg


def test_19_conditions():
    assert len(_conditions()) == EXPECTED_CONDITIONS


def test_condition_ids_are_expected_set():
    ids = [c["condition_id"] for c in _conditions()]
    expected = ["C1", "C2"] + [f"R{i}" for i in range(1, 18)]
    assert ids == expected, f"condition ids/order mismatch: {ids}"


def test_each_condition_has_three_tracks():
    for c in _conditions():
        assert c["tracks"] == TRACKS, f"{c['condition_id']} tracks={c['tracks']}"


def test_total_tracks_is_57():
    total = sum(len(c["tracks"]) for c in _conditions())
    assert total == EXPECTED_TOTAL_TRACKS


def test_sample_ids_unique_and_57():
    sample_ids = [
        f"{c['condition_id']}_{t}" for c in _conditions() for t in c["tracks"]
    ]
    assert len(sample_ids) == EXPECTED_TOTAL_TRACKS
    assert len(set(sample_ids)) == EXPECTED_TOTAL_TRACKS, "sample_id not unique"


def test_factor_levels_valid():
    cfg = _load()
    factors = cfg["experiment_design"]["factors"]
    allowed = {
        "laser_power_W": {factors["laser_power_W"][k] for k in ("low", "center", "high")},
        "scan_speed_mm_min": {factors["scan_speed_mm_min"][k] for k in ("low", "center", "high")},
        "magnetic_field_mT": {factors["magnetic_field_mT"][k] for k in ("low", "center", "high")},
    }
    # The declared levels must be exactly the BBD design levels.
    assert allowed["laser_power_W"] == {300, 450, 600}
    assert allowed["scan_speed_mm_min"] == {400, 600, 800}
    assert allowed["magnetic_field_mT"] == {0, 60, 120}
    # Every condition must use only declared levels.
    for c in _conditions():
        assert c["laser_power_W"] in allowed["laser_power_W"], c["condition_id"]
        assert c["scan_speed_mm_min"] in allowed["scan_speed_mm_min"], c["condition_id"]
        assert c["magnetic_field_mT"] in allowed["magnetic_field_mT"], c["condition_id"]


def test_fixed_parameters_constant():
    for c in _conditions():
        assert c["powder_feed_g_min"] == FIXED_POWDER_FEED, c["condition_id"]
        assert c["travel_distance_mm"] == FIXED_TRAVEL_DISTANCE, c["condition_id"]


def test_design_roles_valid():
    for c in _conditions():
        assert c["design_role"] in ALLOWED_ROLES, c["design_role"]


def test_controls_and_center_repeats():
    by_id = {c["condition_id"]: c for c in _conditions()}
    # Controls
    assert by_id["C1"]["design_role"] == "control_no_field"
    assert by_id["C1"]["magnetic_field_mT"] == 0
    assert by_id["C2"]["design_role"] == "control_high_field"
    assert by_id["C2"]["magnetic_field_mT"] == 120
    # Center-point replicates R13-R17 sit at the center of all three factors.
    for i in range(13, 18):
        c = by_id[f"R{i}"]
        assert c["design_role"] == "box_behnken_center_repeat", c["condition_id"]
        assert (c["laser_power_W"], c["scan_speed_mm_min"], c["magnetic_field_mT"]) == (450, 600, 60)


def test_track_order_map():
    cfg = _load()
    assert cfg["track_order_map"] == {"T1": 1, "T2": 2, "T3": 3}


# ---------------- Data-dependent checks (need the raw data) ----------------

def test_raw_folders_and_session_xml_exist():
    root = _data_root()
    if root is None:
        print("SKIP: raw_data_root not present on this machine.")
        return
    missing_dirs, missing_sessions = [], []
    n_tracks = 0
    for c in _conditions():
        for t in c["tracks"]:
            n_tracks += 1
            folder = os.path.join(root, c["condition_id"], t)
            if not os.path.isdir(folder):
                missing_dirs.append(folder)
            if not os.path.isfile(os.path.join(folder, "session.xml")):
                missing_sessions.append(folder)
    assert n_tracks == EXPECTED_TOTAL_TRACKS
    assert not missing_dirs, f"missing track folders: {missing_dirs[:5]}"
    assert not missing_sessions, f"missing session.xml: {missing_sessions[:5]}"


def test_total_xtherm_count_matches():
    root = _data_root()
    if root is None:
        print("SKIP: raw_data_root not present on this machine.")
        return
    total = 0
    for c in _conditions():
        for t in c["tracks"]:
            folder = os.path.join(root, c["condition_id"], t)
            total += len(glob.glob(os.path.join(folder, "*.xtherm")))
    assert total == EXPECTED_XTHERM_TOTAL, f"xtherm total={total}, expected {EXPECTED_XTHERM_TOTAL}"


def test_frame_count_decreases_with_scan_speed():
    """Sanity (not a parameter edit): with fixed travel distance, slower scan ->
    more frames. Mean frames per track must fall as 400 -> 600 -> 800 mm/min."""
    root = _data_root()
    if root is None:
        print("SKIP: raw_data_root not present on this machine.")
        return
    by_speed = {400: [], 600: [], 800: []}
    for c in _conditions():
        speed = c["scan_speed_mm_min"]
        for t in c["tracks"]:
            folder = os.path.join(root, c["condition_id"], t)
            by_speed[speed].append(len(glob.glob(os.path.join(folder, "*.xtherm"))))
    means = {s: (sum(v) / len(v)) for s, v in by_speed.items() if v}
    assert means[400] > means[600] > means[800], f"frame/speed sanity failed: {means}"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f"  ok: {fn.__name__}")
    print(f"test_experiment_design: {passed} checks passed (formal config; real data when present)")
