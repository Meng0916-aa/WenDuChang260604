"""
Tests for the unified ROI-strategy evaluation (scripts/03a + src/processing +
src/roi). Algorithm checks use small SYNTHETIC frames (deterministic, fast);
the 57-row / 19-row output checks read the real result CSVs when present and
skip otherwise. No formal feature extraction / ROI matrices are produced here.

Covers the 28 verification points of the task.
"""

import os
import sys
import csv
import glob
import importlib.util

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from processing.hot_region_mask import (
    classify_temperature_states, extract_main_hot_region, remove_small_components,
    mask_bbox, mask_centroid, analyze_frame,
)
from roi.roi_evaluation import (
    effective_frame_indices, effective_frame_info, bbox_union, bbox_width_height,
    rect_touches_frame, coverage_fraction, min_edge_distance, expand_rect,
    clamp_rect, round_rect_up, build_global_roi_candidate, tracking_window_size,
    place_window, bbox_inside, evaluate_tracking_window,
)

MATRIX_ROOT = os.path.join(_ROOT, "data", "processed", "matrix")
MASTER = os.path.join(_ROOT, "data", "metadata", "experiment_master.csv")
TABLES = os.path.join(_ROOT, "results", "tables")
SCRIPT = os.path.join(_ROOT, "scripts", "03a_evaluate_roi_strategy.py")


def _load_script():
    spec = importlib.util.spec_from_file_location("eval_roi_03a", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _frame_with_envelope_and_core():
    """40x50 frame: background 100 C, 700 C envelope block, 800 C core block."""
    f = np.full((40, 50), 100.0, dtype=np.float32)
    f[10:20, 10:25] = 750.0      # envelope (>=700, <800): 10x15 = 150 px
    f[12:16, 12:18] = 900.0      # core (>=800): 4x6 = 24 px
    return f


# 1 -------------------------------------------------------------------------
def test_effective_frames_are_frames_1_onwards():
    assert effective_frame_indices(5) == [1, 2, 3, 4]
    assert effective_frame_indices(1) == []
    info = effective_frame_info(150)
    assert info["effective_frame_count"] == 149
    assert info["excluded_frame_count"] == 1
    assert info["effective_start_index_0_based"] == 1


# 2 -------------------------------------------------------------------------
def test_raw_frame_not_modified():
    f = _frame_with_envelope_and_core()
    before = f.copy()
    analyze_frame(f)
    extract_main_hot_region(f, 700.0)
    classify_temperature_states(f)
    assert np.array_equal(f, before)            # input untouched


# 3-5 -----------------------------------------------------------------------
def test_temperature_state_classification():
    f = np.array([[100.0, 300.0, 1800.0],
                  [1801.0, 6499.0, 6500.0]], dtype=np.float32)
    st = classify_temperature_states(f, 300.0, 1800.0, 6500.0)
    assert st["below_range"][0, 0] and not st["below_range"][0, 1]
    assert st["valid"][0, 1] and st["valid"][0, 2]          # 300 and 1800 inclusive
    assert st["above_range"][1, 0] and st["above_range"][1, 1]   # 1801, 6499
    assert not st["above_range"][1, 2]
    assert st["hard_saturation"][1, 2]                      # 6500 inclusive
    assert not st["hard_saturation"][1, 1]
    # exhaustive & mutually exclusive
    total = sum(int(st[k].sum()) for k in st)
    assert total == f.size


# 6 -------------------------------------------------------------------------
def test_above_range_continuous_with_body_is_retained():
    f = _frame_with_envelope_and_core()
    f[14, 14] = 2739.0                       # above-range pixel inside the body
    mask, info = extract_main_hot_region(f, 700.0, min_component_area=9)
    assert info["above_range_connected_count"] == 1
    assert info["above_range_isolated_count"] == 0
    assert mask[14, 14]                      # kept as part of the main body


# 7 -------------------------------------------------------------------------
def test_isolated_above_range_pixel_excluded():
    f = _frame_with_envelope_and_core()
    f[0, 0] = 2739.0                         # lone above-range pixel, no valid neighbor
    mask, info = extract_main_hot_region(f, 700.0, min_component_area=1)
    assert not mask[0, 0]                     # excluded from the main body
    assert info["above_range_isolated_count"] >= 1


# 8 -------------------------------------------------------------------------
def test_hard_saturation_inside_body_no_hole():
    f = _frame_with_envelope_and_core()
    f[14, 14] = 6553.5                       # hard-saturation sentinel inside body
    mask, info = extract_main_hot_region(f, 700.0)
    assert info["hard_saturation_count"] == 1
    assert mask[14, 14]                       # no hole at the saturated pixel


def test_fill_holes_closes_sub_threshold_interior():
    f = np.full((30, 30), 100.0, dtype=np.float32)
    f[5:20, 5:20] = 900.0                    # solid hot block
    f[12, 12] = 200.0                        # carve a sub-threshold interior hole
    mask, _ = extract_main_hot_region(f, 700.0, fill_holes=True)
    assert mask[12, 12]                       # interior hole filled (mask only)


# 9 -------------------------------------------------------------------------
def test_700_envelope_area_and_bbox():
    f = _frame_with_envelope_and_core()
    mask, _ = extract_main_hot_region(f, 700.0, min_component_area=9)
    assert int(mask.sum()) == 150
    assert mask_bbox(mask) == (10, 10, 20, 25)


# 10 ------------------------------------------------------------------------
def test_800_core_area_and_bbox():
    f = _frame_with_envelope_and_core()
    mask, _ = extract_main_hot_region(f, 800.0, min_component_area=9)
    assert int(mask.sum()) == 24
    assert mask_bbox(mask) == (12, 12, 16, 18)


# 11 ------------------------------------------------------------------------
def test_small_component_cleanup():
    f = _frame_with_envelope_and_core()
    f[0, 0:4] = 750.0                        # tiny 4-px valid blob (< min area 9)
    mask, info = extract_main_hot_region(f, 700.0, min_component_area=9)
    assert mask_bbox(mask) == (10, 10, 20, 25)   # only the big body survives
    # remove_small_components drops the tiny blob too
    raw = f >= 700.0
    cleaned = remove_small_components(raw, 9)
    assert not cleaned[0, 0]


# 12 ------------------------------------------------------------------------
def test_multiframe_bbox_union():
    assert bbox_union([(10, 10, 20, 20), (5, 15, 25, 18), None]) == (5, 10, 25, 20)
    assert bbox_union([None, None]) is None


# 13 ------------------------------------------------------------------------
def test_old_roi_half_open_interval():
    m = np.zeros((512, 640), dtype=bool)
    m[499, 10] = True       # inside rows [200, 500)
    m[500, 10] = True       # outside (500 is exclusive)
    cov = coverage_fraction(m, (200, 0, 500, 600))
    assert abs(cov - 0.5) < 1e-9


# 14 ------------------------------------------------------------------------
def test_coverage_fraction_value():
    m = np.zeros((20, 20), dtype=bool)
    m[1, 1] = m[1, 2] = m[1, 3] = m[15, 15] = True   # 4 px; 3 inside rect
    cov = coverage_fraction(m, (0, 0, 5, 5))
    assert abs(cov - 0.75) < 1e-9
    assert np.isnan(coverage_fraction(np.zeros((5, 5), bool), (0, 0, 5, 5)))


# 15 ------------------------------------------------------------------------
def test_min_edge_distance_value():
    assert min_edge_distance((210, 10, 490, 590), (200, 0, 500, 600)) == 10
    assert min_edge_distance((195, 10, 480, 500), (200, 0, 500, 600)) == -5   # over the top
    assert min_edge_distance(None, (200, 0, 500, 600)) is None


# 16 ------------------------------------------------------------------------
def test_global_bbox_merge():
    boxes = [(297, 128, 428, 287), (281, 123, 405, 290), (286, 127, 434, 289)]
    assert bbox_union(boxes) == (281, 123, 434, 290)


# 17 ------------------------------------------------------------------------
def test_margin_expansion():
    assert expand_rect((100, 100, 200, 200), 20, 512, 640) == (80, 80, 220, 220)
    # clamps at the frame
    assert expand_rect((5, 5, 500, 630), 20, 512, 640) == (0, 0, 512, 640)


# 18 ------------------------------------------------------------------------
def test_candidate_roi_within_frame():
    cand = build_global_roi_candidate((0, 0, 510, 638), 20, 512, 640, round_to=8)
    assert 0 <= cand["candidate_top"]
    assert 0 <= cand["candidate_left"]
    assert cand["candidate_bottom_exclusive"] <= 512
    assert cand["candidate_right_exclusive"] <= 640
    # rounding only grows, never exceeds the frame
    r = round_rect_up((100, 100, 150, 150), 16, 512, 640)
    assert (r[2] - r[0]) % 16 == 0 and (r[3] - r[1]) % 16 == 0
    assert r[2] <= 512 and r[3] <= 640
    assert clamp_rect((-5, -5, 700, 700), 512, 640) == (0, 0, 512, 640)


# 19, 21 --------------------------------------------------------------------
def test_tracking_window_size_is_uniform_and_no_scaling():
    win_h, win_w = tracking_window_size([100, 120, 110], [80, 90, 85],
                                        percentile=99, edge_safety_margin=10,
                                        round_to=8, height=512, width=640)
    # same fixed size used everywhere; placing at different centers keeps the size
    r1, _ = place_window((100, 100), win_h, win_w, 512, 640)
    r2, _ = place_window((300, 250), win_h, win_w, 512, 640)
    assert (r1[2] - r1[0]) == win_h and (r1[3] - r1[1]) == win_w
    assert (r2[2] - r2[0]) == win_h and (r2[3] - r2[1]) == win_w


# 20 ------------------------------------------------------------------------
def test_tracking_window_center_moves_and_edge_adjusts():
    r1, adj1 = place_window((100, 100), 50, 50, 512, 640)
    r2, adj2 = place_window((300, 250), 50, 50, 512, 640)
    assert r1[:2] != r2[:2]               # center moved
    assert not adj1 and not adj2
    r3, adj3 = place_window((5, 5), 50, 50, 512, 640)
    assert adj3 and r3 == (0, 0, 50, 50)  # shifted inside the frame


# 22 ------------------------------------------------------------------------
def test_tracking_window_coverage_counts():
    bboxes = [(100, 100, 120, 120), (0, 0, 60, 60), None]
    centroids = [(110, 110), (30, 30), None]
    ev = evaluate_tracking_window(bboxes, centroids, 50, 50, 512, 640)
    assert ev["total_frames"] == 2        # None frame skipped
    assert ev["fully_covered_frame_count"] == 1
    assert ev["clipped_frame_count"] == 1
    assert bbox_inside((100, 100, 120, 120), (100, 100, 150, 150))


# 23 ------------------------------------------------------------------------
def test_dataset_npy_excluded(tmp_path):
    mod = _load_script()
    p = tmp_path / "m.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "condition_id", "track_id"])
        w.writeheader()
        w.writerow({"sample_id": "C1_T1", "condition_id": "C1", "track_id": "T1"})
        w.writerow({"sample_id": "dataset", "condition_id": "", "track_id": ""})
        w.writerow({"sample_id": "R1_T1", "condition_id": "R1", "track_id": "T1"})
    rows = mod._load_master(str(p))
    sids = {r["sample_id"] for r in rows}
    assert "dataset" not in sids
    assert sids == {"C1_T1", "R1_T1"}


# 24 ------------------------------------------------------------------------
def test_loads_57_track_matrices():
    if not (os.path.isfile(MASTER) and os.path.isdir(MATRIX_ROOT)):
        pytest.skip("master csv / matrix root not present")
    with open(MASTER, "r", encoding="utf-8") as f:
        sids = [r["sample_id"].strip() for r in csv.DictReader(f)]
    assert len(sids) == 57
    have = [s for s in sids if os.path.isfile(os.path.join(MATRIX_ROOT, f"{s}.npy"))]
    assert len(have) == 57
    assert "dataset" not in sids


# 25 ------------------------------------------------------------------------
def test_dry_run_writes_no_roi_matrix():
    if not (os.path.isfile(MASTER) and os.path.isdir(MATRIX_ROOT)):
        pytest.skip("master csv / matrix root not present")
    mod = _load_script()
    before = set(glob.glob(os.path.join(MATRIX_ROOT, "*.npy")))
    res = mod.main(["--dry-run", "--max-tracks", "2", "--max-frames-per-track", "8"])
    after = set(glob.glob(os.path.join(MATRIX_ROOT, "*.npy")))
    assert before == after                                   # no new matrices
    assert "summary" in res
    # dry-run created no ROI .npy anywhere under results/
    roi_npys = glob.glob(os.path.join(_ROOT, "results", "**", "*roi*.npy"), recursive=True)
    assert roi_npys == []


# 26 ------------------------------------------------------------------------
def test_main_table_has_57_rows_when_present():
    path = os.path.join(TABLES, "roi_bbox_by_track.csv")
    if not os.path.isfile(path):
        pytest.skip("roi_bbox_by_track.csv not generated yet")
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 57
    assert len({r["sample_id"] for r in rows}) == 57


# 27 ------------------------------------------------------------------------
def test_repeatability_summary_has_19_rows_when_present():
    path = os.path.join(TABLES, "roi_repeatability_summary.csv")
    if not os.path.isfile(path):
        pytest.skip("roi_repeatability_summary.csv not generated yet")
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 19


# 28 ------------------------------------------------------------------------
def test_no_formal_feature_extraction_in_pipeline():
    with open(SCRIPT, "r", encoding="utf-8") as f:
        src = f.read()
    assert "extract_experiment_features" not in src
    assert "extract_local_section_features" not in src
    assert "np.save" not in src
    assert "from models" not in src and "import models" not in src
    # the pipeline reuses the hot-region / roi modules (no second algorithm)
    assert "from processing.hot_region_mask import" in src
    assert "from roi.roi_evaluation import" in src


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
