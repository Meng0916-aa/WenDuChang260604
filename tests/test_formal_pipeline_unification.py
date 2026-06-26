"""
Tests for the formal-pipeline unification, legacy isolation, portable path
resolution, the corrected extent-based tracking window, module decoupling, and
the dependency / gitignore rules.

Covers the 30 verification points of the "unify formal pipeline + fix tracking
ROI coverage" task. Config/loader/algorithm checks always run; checks that need
the real local matrices skip when absent. No formal ROI matrices / features are
produced here.
"""

import os
import sys
import subprocess

import yaml
import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from config import formal_config as fc
from config.path_resolution import resolve_data_root, DataRootError, normalize_path
from roi.roi_evaluation import (
    compute_extents, extent_stats, tracking_window_from_extents,
    evaluate_tracking_window, place_window, classify_tracking_window,
)


def _yaml(rel):
    with open(os.path.join(_ROOT, rel), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ===== Formal vs legacy config isolation (1-6) =============================

def test_01_default_yaml_marked_legacy():
    cs = _yaml("configs/default.yaml")["config_status"]
    assert cs["role"] == "legacy_pilot"
    assert cs["formal_processing_enabled"] is False
    assert cs["approved_for_57_track_pipeline"] is False


def test_02_default_roi_disabled():
    roi = _yaml("configs/default.yaml")["roi"]
    assert roi["enabled"] is False
    assert roi["status"] == "legacy_not_approved"


def test_03_formal_mode_rejects_legacy_roi():
    cfg = _yaml("configs/default.yaml")
    with pytest.raises(fc.FormalConfigError):
        fc.reject_legacy_roi(cfg)
    with pytest.raises(fc.FormalConfigError):
        fc.assert_not_legacy_default(cfg)


def test_04_formal_mode_refuses_dataset_path():
    for p in ("data/raw_xtherm/dataset/000001.xtherm",
              "data/exported/npy/dataset.npy",
              r"D:\x\dataset\f.xtherm"):
        with pytest.raises(fc.FormalConfigError):
            fc.assert_no_legacy_dataset_path(p)
    fc.assert_no_legacy_dataset_path("data/processed/matrix/R1_T1.npy")


def test_05_formal_pixel_size_only_from_calibration():
    cfg = _yaml("configs/default.yaml")
    with pytest.raises(fc.FormalConfigError):
        fc.assert_pixel_size_from_calibration(cfg)
    assert fc.formal_calibration_config() == "configs/physical_calibration.yaml"


def test_06_formal_pipeline_has_no_absolute_path():
    cfg = _yaml("configs/formal_pipeline.yaml")
    assert cfg["pipeline"]["status"] == "active"
    assert fc.find_absolute_path_values(cfg) == []
    assert fc.find_absolute_path_values(_yaml("configs/experiments.yaml")) == []


def test_formal_pipeline_explicit_authoritative_config_chain():
    cfg = _yaml("configs/formal_pipeline.yaml")
    chain = cfg["authoritative_config_chain"]
    assert chain["xtherm_format_config"] == "configs/xtherm_format.yaml"
    assert chain["physical_calibration_config"] == "configs/physical_calibration.yaml"
    assert chain["roi_strategy_config"] == "configs/roi_strategy.yaml"
    assert (
        chain["thermal_feature_contract_config"]
        == "configs/thermal_feature_contract.yaml"
    )
    assert cfg["processing"]["generate_formal_roi_matrices"] is False
    assert cfg["processing"]["formal_feature_extraction_enabled"] is False


# ===== Path resolution (7-12) =============================================

def test_07_cli_path_has_highest_priority():
    r = resolve_data_root(cli_arg="C:/cli/data",
                          env={"WENDUCHANG_DATA_ROOT": "D:/env/data"},
                          require_exists=False)
    assert r.source == "cli"
    assert r.path == normalize_path("C:/cli/data")


def test_08_env_beats_local_yaml():
    r = resolve_data_root(env={"WENDUCHANG_DATA_ROOT": "D:/env/data"},
                          require_exists=False)
    assert r.source == "env:WENDUCHANG_DATA_ROOT"


def test_09_local_yaml_is_git_ignored():
    out = subprocess.run(["git", "check-ignore", "configs/local.yaml"],
                         cwd=_ROOT, capture_output=True, text=True)
    assert out.returncode == 0 and "configs/local.yaml" in out.stdout
    out2 = subprocess.run(["git", "check-ignore", "configs/local.example.yaml"],
                          cwd=_ROOT, capture_output=True, text=True)
    assert out2.returncode == 1


def test_10_clear_error_when_no_data_root():
    with pytest.raises(DataRootError):
        resolve_data_root(env={}, local_config="configs/__none__.yaml",
                          experiments_config="configs/experiments.yaml",
                          require_exists=False)


def test_11_windows_path_parsed():
    assert normalize_path(r"D:\a\b\c") == os.path.normpath("D:/a/b/c")
    r = resolve_data_root(cli_arg=r"D:\WenDuChang-data-repo\raw_xtherm",
                          require_exists=False)
    assert "raw_xtherm" in r.path


def test_12_tests_do_not_need_real_d_drive():
    r = resolve_data_root(cli_arg="/nonexistent/posix/path", require_exists=False)
    assert r.path == os.path.normpath("/nonexistent/posix/path")
    with pytest.raises(DataRootError):
        resolve_data_root(cli_arg="/surely/not/here/xyz", require_exists=True)


# ===== Tracking window (13-22) ===========================================

def test_13_extent_computation():
    e = compute_extents((10, 20, 40, 60), (35.0, 22.0))
    assert e["left"] == 15.0      # 35 - 20
    assert e["right"] == 25.0     # 60 - 35
    assert e["top"] == 12.0       # 22 - 10
    assert e["bottom"] == 18.0    # 40 - 22


def test_14_float_centroid_and_half_open_bbox():
    e = compute_extents((0, 0, 10, 10), (4.5, 4.5))
    assert e["left"] == 4.5 and e["right"] == 5.5
    assert e["top"] == 4.5 and e["bottom"] == 5.5
    assert compute_extents(None, (1, 1)) is None
    assert compute_extents((0, 0, 5, 5), None) is None


def test_15_asymmetric_region_not_clipped_by_centroid_centering():
    bbox = (40, 10, 60, 120)     # width 110, height 20
    centroid = (30.0, 50.0)      # left of the bbox center (65)
    win = tracking_window_from_extents([bbox], [centroid], round_to=8,
                                       height=512, width=640)
    ev = evaluate_tracking_window([bbox], [centroid],
                                  win["win_height_px"], win["win_width_px"],
                                  512, 640)
    assert ev["clipped_frame_count"] == 0
    assert ev["fully_covered_frame_count"] == 1


def test_16_17_18_recommended_window_full_coverage_zero_clip():
    rng = np.random.RandomState(0)
    bboxes, centroids = [], []
    for _ in range(200):
        cy = rng.uniform(200, 320); cx = rng.uniform(150, 300)
        top = int(cy - rng.uniform(40, 70)); bot = int(cy + rng.uniform(40, 70))
        left = int(cx - rng.uniform(60, 90)); right = int(cx + rng.uniform(60, 90))
        bboxes.append((top, left, bot, right)); centroids.append((cx, cy))
    win = tracking_window_from_extents(bboxes, centroids, round_to=8, height=512, width=640)
    ev = evaluate_tracking_window(bboxes, centroids, win["win_height_px"],
                                  win["win_width_px"], 512, 640)
    assert ev["coverage_rate"] == 1.0       # 16/17: main 700/800 coverage 100%
    assert ev["clipped_frame_count"] == 0   # 18: zero clipped frames


def test_19_edge_window_translates_not_scales():
    rect, adjusted = place_window((5, 5), 50, 60, 512, 640)
    assert adjusted is True
    assert rect == (0, 0, 50, 60)                       # same size, translated
    assert (rect[2] - rect[0], rect[3] - rect[1]) == (50, 60)


def test_20_uniform_window_size_for_all_frames():
    win = tracking_window_from_extents(
        [(10, 10, 30, 40), (200, 300, 260, 360)],
        [(25.0, 20.0), (330.0, 230.0)], round_to=8, height=512, width=640)
    h, w = win["win_height_px"], win["win_width_px"]
    r1, _ = place_window((25, 20), h, w, 512, 640)
    r2, _ = place_window((330, 230), h, w, 512, 640)
    assert (r1[2] - r1[0], r1[3] - r1[1]) == (h, w)
    assert (r2[2] - r2[0], r2[3] - r2[1]) == (h, w)


def test_21_window_too_large_falls_back_to_global_roi():
    cand_area = 80000
    assert classify_tracking_window(300, 240, cand_area, True, True) == "auxiliary_only"
    assert classify_tracking_window(200, 200, cand_area, True, True) == "candidate_full_coverage"
    assert classify_tracking_window(200, 200, cand_area, True, False) == "rejected_clips_main_region"
    assert classify_tracking_window(600, 600, cand_area, False, True) == "auxiliary_only"


def test_22_extent_input_arrays_not_modified():
    bboxes = [(10, 10, 30, 40), (20, 20, 50, 60)]
    centroids = [(25.0, 20.0), (40.0, 35.0)]
    bcopy = list(bboxes); ccopy = list(centroids)
    tracking_window_from_extents(bboxes, centroids)
    extent_stats([compute_extents(b, c) for b, c in zip(bboxes, centroids)])
    assert bboxes == bcopy and centroids == ccopy


# ===== Module dependency (23-25) =========================================

def _read(rel):
    with open(os.path.join(_ROOT, rel), "r", encoding="utf-8") as f:
        return f.read()


def test_23_processing_does_not_depend_on_features():
    for mod in ("src/processing/temperature_mask.py", "src/processing/hot_region_mask.py"):
        src = _read(mod)
        assert "from features" not in src and "import features" not in src, mod


def test_24_single_threshold_mask_implementation():
    tm = _read("src/processing/temperature_mask.py")
    assert "def build_threshold_mask" in tm
    feat = _read("src/features/thermal_field_features.py")
    assert "from processing.temperature_mask import build_threshold_mask" in feat
    assert "return build_threshold_mask(" in feat


def test_25_legacy_wrapper_behaviour_preserved():
    from features.thermal_field_features import compute_high_temperature_mask
    from processing.temperature_mask import build_threshold_mask
    f = np.full((12, 12), 100.0, dtype=np.float32)
    f[3:8, 3:8] = 900.0
    a = compute_high_temperature_mask(f, 700.0)
    assert np.array_equal(a, build_threshold_mask(f, 700.0))
    assert np.array_equal(a, f >= 700.0)


# ===== Dependencies & ignore rules (26-30) ===============================

def test_26_requirements_has_no_torch():
    for line in _read("requirements.txt").splitlines():
        s = line.strip().lower()
        if s.startswith("#") or not s:
            continue
        assert not s.replace(" ", "").startswith("torch"), f"torch in requirements: {line}"


def test_27_readme_documents_pytorch_separately():
    readme = _read("README.md")
    assert "requirements.txt" in readme
    assert ("managed separately" in readme) or ("not in `requirements.txt`" in readme.lower())


def _check_ignore(path):
    out = subprocess.run(["git", "check-ignore", path], cwd=_ROOT,
                         capture_output=True, text=True)
    return out.returncode == 0     # True = ignored


def test_28_data_and_results_csv_still_ignored():
    assert _check_ignore("data/metadata/experiment_master.csv")
    assert _check_ignore("results/tables/roi_bbox_by_track.csv")


def test_29_fixtures_and_templates_csv_trackable():
    assert not _check_ignore("tests/fixtures/example.csv")
    assert not _check_ignore("templates/metadata_template.csv")
    assert not _check_ignore("examples/demo.csv")


def test_30_local_yaml_ignored():
    assert _check_ignore("configs/local.yaml")


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    ok = 0
    for fn in fns:
        try:
            fn(); ok += 1; print(f"  ok: {fn.__name__}")
        except Exception:
            print(f"  FAIL: {fn.__name__}"); traceback.print_exc()
    print(f"{ok}/{len(fns)} passed")
