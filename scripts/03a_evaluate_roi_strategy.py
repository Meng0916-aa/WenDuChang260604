"""
03a_evaluate_roi_strategy.py

UNIFIED ROI CANDIDATE EVALUATION + ANALYSIS-COORDINATE PLAN (evaluation only).

Answers, for the 57 formal single-track temperature matrices:
  1. where the main melt-pool hot region sits in the image,
  2. whether the legacy ROI (top=200,left=0,h=300,w=600) covers every main hot
     region,
  3. whether ONE fixed global ROI can serve all 57 tracks,
  4. how far the melt pool travels upward in the image,
  5. whether a fixed ROI would include too much background,
  6. whether a two-layer (fixed global ROI + fixed-size tracking window) plan is
     needed,
  7. which samples touch an edge / are offset / carry abnormal hot regions.

This script DOES NOT crop or write any ROI .npy, never modifies the raw matrices
(read-only / mmap), and runs no feature extraction / ML / response-surface work.
It writes evaluation tables + a JSON summary (+ QC figures) under results/.

Effective frames per track = frames[1:] (the first frame is the camera startup
frame and is excluded). dataset.npy is excluded explicitly.

Usage:
    python scripts/03a_evaluate_roi_strategy.py \
        --master-csv data/metadata/experiment_master.csv \
        --matrix-root data/processed/matrix \
        --calibration-config configs/physical_calibration.yaml \
        --output-root results
"""

import os
import sys
import csv
import json
import argparse

import numpy as np
import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from config.physical_calibration import load_physical_calibration
from processing.hot_region_mask import analyze_frame
from roi.roi_evaluation import (
    effective_frame_indices, effective_frame_info, bbox_union, bbox_width_height,
    rect_touches_frame, coverage_fraction, min_edge_distance,
    build_global_roi_candidate, size_percentiles, tracking_window_size,
    evaluate_tracking_window,
)

EXCLUDED_STEMS = {"dataset"}            # data/processed/matrix/dataset.npy excluded
DEFAULT_OLD_ROI = (200, 0, 300, 600)    # legacy candidate: top, left, height, width


# ---------------------------------------------------------------------------
# small IO helpers
# ---------------------------------------------------------------------------

def _abs(path):
    return path if os.path.isabs(path) else os.path.join(_ROOT, path)


def _load_master(master_csv):
    with open(_abs(master_csv), "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        sid = (r.get("sample_id") or "").strip()
        if not sid or sid in EXCLUDED_STEMS:
            continue
        out.append(r)
    return out


def _old_roi_rect(default_cfg_path):
    """Legacy ROI as a half-open rect (top, left, bottom_excl, right_excl)."""
    top, left, h, w = DEFAULT_OLD_ROI
    try:
        with open(_abs(default_cfg_path), "r", encoding="utf-8") as f:
            roi = (yaml.safe_load(f) or {}).get("roi", {})
        if all(roi.get(k) is not None for k in ("top", "left", "height", "width")):
            top, left, h, w = (int(roi["top"]), int(roi["left"]),
                               int(roi["height"]), int(roi["width"]))
    except FileNotFoundError:
        pass
    return (top, left, top + h, left + w)


def _f(value):
    """JSON/CSV-safe: None -> '', round floats lightly."""
    if value is None:
        return ""
    if isinstance(value, float):
        if np.isnan(value):
            return ""
        return round(value, 6)
    return value


# ---------------------------------------------------------------------------
# per-track evaluation (read-only)
# ---------------------------------------------------------------------------

def evaluate_track(matrix_path, env_thr, core_thr, vmin, vmax, hard, min_area,
                   max_frames=None):
    """Read one matrix (mmap, read-only) and aggregate over effective frames.

    Returns a dict of per-track aggregates + the 700/800 union masks + the
    per-frame bbox/centroid lists (for the global tracking-window analysis).
    """
    arr = np.load(matrix_path, mmap_mode="r")
    if arr.ndim != 3:
        raise ValueError(f"{matrix_path}: expected N x H x W, got {arr.shape}")
    n, h, w = arr.shape
    idx = effective_frame_indices(n)
    if max_frames is not None:
        idx = idx[:int(max_frames)]

    union700 = np.zeros((h, w), dtype=bool)
    union800 = np.zeros((h, w), dtype=bool)
    centroids, bboxes700, bboxes800 = [], [], []
    widths700, heights700 = [], []
    above_conn = above_iso = hard_cnt = 0
    n_no_env = n_no_core = 0

    for i in idx:
        frame = np.asarray(arr[i], dtype=np.float32)     # copy of ONE frame; raw untouched
        res = analyze_frame(frame, env_thr, core_thr, vmin, vmax, hard, min_area)
        if res["bbox700"] is not None:
            union700 |= res["envelope_mask"]
            bw, bh = bbox_width_height(res["bbox700"])
            widths700.append(bw); heights700.append(bh)
        else:
            n_no_env += 1
        if res["bbox800"] is not None:
            union800 |= res["core_mask"]
        else:
            n_no_core += 1
        centroids.append(res["centroid"])
        bboxes700.append(res["bbox700"])
        bboxes800.append(res["bbox800"])
        above_conn += res["above_range_connected_count"]
        above_iso += res["above_range_isolated_count"]
        hard_cnt += res["hard_saturation_count"]

    del arr   # release mmap

    track_bbox700 = bbox_union(bboxes700)
    track_bbox800 = bbox_union(bboxes800)
    cxs = [c[0] for c in centroids if c is not None]
    cys = [c[1] for c in centroids if c is not None]
    cx_min = min(cxs) if cxs else None
    cx_max = max(cxs) if cxs else None
    cy_min = min(cys) if cys else None
    cy_max = max(cys) if cys else None
    vtravel = (cy_max - cy_min) if (cy_min is not None) else None

    return {
        "frame_shape": (n, h, w),
        "effective": effective_frame_info(n) if max_frames is None
                     else {"original_frame_count": n,
                           "effective_frame_count": len(idx),
                           "excluded_frame_count": 1},
        "union700": union700,
        "union800": union800,
        "track_bbox700": track_bbox700,
        "track_bbox800": track_bbox800,
        "centroids": centroids,
        "bboxes700": bboxes700,
        "widths700": widths700,
        "heights700": heights700,
        "centroid_x_min": cx_min, "centroid_x_max": cx_max,
        "centroid_y_min": cy_min, "centroid_y_max": cy_max,
        "centroid_vertical_travel_px": vtravel,
        "mean_centroid_x": float(np.mean(cxs)) if cxs else None,
        "mean_centroid_y": float(np.mean(cys)) if cys else None,
        "above_range_connected_count": int(above_conn),
        "above_range_isolated_count": int(above_iso),
        "hard_saturation_count": int(hard_cnt),
        "n_frames_no_envelope": int(n_no_env),
        "n_frames_no_core": int(n_no_core),
    }


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def run(args):
    calib = load_physical_calibration(_abs(args.calibration_config))
    H, W = calib.image_height_px, calib.image_width_px
    px_x, px_y = calib.pixel_size_x_mm, calib.pixel_size_y_mm
    vmin = calib.valid_temperature_min_C
    vmax = calib.valid_temperature_max_C
    hard = calib.hard_saturation_threshold_C
    env_thr = float(args.envelope_threshold)
    core_thr = float(args.core_threshold)
    edge_margin = int(args.edge_safety_margin)

    samples = _load_master(args.master_csv)
    if args.max_tracks:
        samples = samples[:int(args.max_tracks)]
    old_roi = _old_roi_rect("configs/default.yaml")

    matrix_root = _abs(args.matrix_root)
    per_track = {}
    missing = []
    all_widths, all_heights = [], []
    all_frame_bboxes, all_frame_centroids = [], []

    for r in samples:
        sid = r["sample_id"].strip()
        path = os.path.join(matrix_root, f"{sid}.npy")
        if not os.path.isfile(path):
            missing.append(sid)
            continue
        t = evaluate_track(path, env_thr, core_thr, vmin, vmax, hard,
                           int(args.min_component_area),
                           max_frames=args.max_frames_per_track)
        t["sample_id"] = sid
        t["condition_id"] = r.get("condition_id", "")
        t["track_id"] = r.get("track_id", "")
        per_track[sid] = t
        all_widths.extend(t["widths700"])
        all_heights.extend(t["heights700"])
        all_frame_bboxes.extend(t["bboxes700"])
        all_frame_centroids.extend(t["centroids"])
        print(f"[03a] {sid}: eff={t['effective']['effective_frame_count']} "
              f"bbox700={t['track_bbox700']} vtravel_px="
              f"{None if t['centroid_vertical_travel_px'] is None else round(t['centroid_vertical_travel_px'],1)}")

    if not per_track:
        raise SystemExit("[03a] no matrices found — nothing to evaluate")

    # --- global 700/800 union bbox over all tracks ---
    raw_global_700 = bbox_union([t["track_bbox700"] for t in per_track.values()])
    raw_global_800 = bbox_union([t["track_bbox800"] for t in per_track.values()])

    cand = build_global_roi_candidate(raw_global_700, int(args.global_margin),
                                       H, W, round_to=int(args.round_to))
    cand_rect = cand["rect"] if cand else (0, 0, H, W)

    # --- tracking-window candidates from the 700 bbox size distribution ---
    width_pcts = size_percentiles(all_widths, (95, 99))
    height_pcts = size_percentiles(all_heights, (95, 99))
    if args.tracking_window_height and args.tracking_window_width:
        rec_win = (int(args.tracking_window_height), int(args.tracking_window_width))
    else:
        rec_win = tracking_window_size(all_widths, all_heights, percentile=99,
                                       edge_safety_margin=edge_margin,
                                       round_to=int(args.round_to),
                                       height=H, width=W)
    rec_win_h, rec_win_w = rec_win

    # window candidate set: p95, p99, recommended
    def _win_from(pw, ph):
        ww = min(int(np.ceil((pw + 2 * edge_margin) / args.round_to) * args.round_to), W)
        wh = min(int(np.ceil((ph + 2 * edge_margin) / args.round_to) * args.round_to), H)
        return (wh, ww)
    win_candidates = []
    seen = set()
    for (wh, ww) in [_win_from(width_pcts[95], height_pcts[95]),
                     _win_from(width_pcts[99], height_pcts[99]),
                     (rec_win_h, rec_win_w)]:
        if (wh, ww) not in seen:
            seen.add((wh, ww)); win_candidates.append((wh, ww))

    total_eff_frames = sum(t["effective"]["effective_frame_count"] for t in per_track.values())
    win_rows = []
    for (wh, ww) in win_candidates:
        ev = evaluate_tracking_window(all_frame_bboxes, all_frame_centroids, wh, ww, H, W)
        ev["estimated_output_size_bytes"] = int(total_eff_frames * wh * ww * 4)
        win_rows.append(ev)
    rec_cov = next((e for e in win_rows if (e["window_height_px"], e["window_width_px"]) == (rec_win_h, rec_win_w)), win_rows[-1])

    # --- per-track coverage vs old + candidate ROI, edge touch, status ---
    track_rows = []
    old_cov700_min = old_cov800_min = 1.0
    old_edge_fail = []
    cand_cov700_min = cand_cov800_min = 1.0
    cand_edge_fail = []
    edge_touch_samples = []
    no_env_samples, no_core_samples = [], []

    for sid, t in per_track.items():
        u700, u800 = t["union700"], t["union800"]
        b700, b800 = t["track_bbox700"], t["track_bbox800"]

        old_c700 = coverage_fraction(u700, old_roi)
        old_c800 = coverage_fraction(u800, old_roi)
        old_edge = min_edge_distance(b700, old_roi)
        cand_c700 = coverage_fraction(u700, cand_rect)
        cand_c800 = coverage_fraction(u800, cand_rect)
        cand_edge = min_edge_distance(b700, cand_rect)

        touch = rect_touches_frame(b700, H, W)
        touches = bool(touch["top"] or touch["bottom"] or touch["left"] or touch["right"])

        warnings = []
        if b700 is None:
            warnings.append("no_700_envelope"); no_env_samples.append(sid)
        if b800 is None:
            warnings.append("no_800_core"); no_core_samples.append(sid)
        if touches:
            warnings.append("touches_frame_edge"); edge_touch_samples.append(sid)
        if not np.isnan(old_c800) and old_c800 < 1.0:
            warnings.append("old_roi_core_loss")
        if not np.isnan(old_c700) and old_c700 < 0.999:
            warnings.append("old_roi_envelope_loss")
        if old_edge is not None and old_edge < edge_margin:
            warnings.append("old_roi_margin_short"); old_edge_fail.append(sid)
        if cand_edge is not None and cand_edge < edge_margin and not touches:
            warnings.append("candidate_margin_short"); cand_edge_fail.append(sid)

        if not np.isnan(old_c700):
            old_cov700_min = min(old_cov700_min, old_c700)
        if not np.isnan(old_c800):
            old_cov800_min = min(old_cov800_min, old_c800)
        if not np.isnan(cand_c700):
            cand_cov700_min = min(cand_cov700_min, cand_c700)
        if not np.isnan(cand_c800):
            cand_cov800_min = min(cand_cov800_min, cand_c800)

        w700, h700 = bbox_width_height(b700)
        roi_status = "ok" if not warnings else "review"
        track_rows.append({
            "sample_id": sid,
            "condition_id": t["condition_id"],
            "track_id": t["track_id"],
            "effective_frame_count": t["effective"]["effective_frame_count"],
            "bbox700_top": _f(b700[0] if b700 else None),
            "bbox700_left": _f(b700[1] if b700 else None),
            "bbox700_bottom_exclusive": _f(b700[2] if b700 else None),
            "bbox700_right_exclusive": _f(b700[3] if b700 else None),
            "bbox800_top": _f(b800[0] if b800 else None),
            "bbox800_left": _f(b800[1] if b800 else None),
            "bbox800_bottom_exclusive": _f(b800[2] if b800 else None),
            "bbox800_right_exclusive": _f(b800[3] if b800 else None),
            "bbox700_width_px": _f(w700), "bbox700_height_px": _f(h700),
            "bbox700_width_mm": _f(w700 * px_x), "bbox700_height_mm": _f(h700 * px_y),
            "centroid_x_min": _f(t["centroid_x_min"]), "centroid_x_max": _f(t["centroid_x_max"]),
            "centroid_y_min": _f(t["centroid_y_min"]), "centroid_y_max": _f(t["centroid_y_max"]),
            "centroid_vertical_travel_px": _f(t["centroid_vertical_travel_px"]),
            "centroid_vertical_travel_mm": _f(
                None if t["centroid_vertical_travel_px"] is None
                else t["centroid_vertical_travel_px"] * px_y),
            "old_roi_coverage_700": _f(old_c700), "old_roi_coverage_800": _f(old_c800),
            "old_roi_min_edge_distance_px": _f(old_edge),
            "candidate_roi_coverage_700": _f(cand_c700),
            "candidate_roi_coverage_800": _f(cand_c800),
            "candidate_roi_min_edge_distance_px": _f(cand_edge),
            "above_range_connected_count": t["above_range_connected_count"],
            "above_range_isolated_count": t["above_range_isolated_count"],
            "hard_saturation_count": t["hard_saturation_count"],
            "roi_status": roi_status,
            "warnings": ";".join(warnings),
        })

    old_roi_usable = bool(old_cov800_min >= 1.0 and old_cov700_min >= 0.999
                          and not old_edge_fail and not edge_touch_samples)

    # --- recommendation ---
    cand_fraction = cand["candidate_fraction_of_full_frame"] if cand else 1.0
    # background ratio: mean melt-pool 700 area vs candidate area
    mean_area700 = float(np.mean([w * h for w, h in zip(all_widths, all_heights)])) if all_widths else 0.0
    cand_area = cand["candidate_area_px"] if cand else (H * W)
    bg_ratio = 1.0 - (mean_area700 / cand_area) if cand_area else 1.0
    vtravels = [t["centroid_vertical_travel_px"] for t in per_track.values()
                if t["centroid_vertical_travel_px"] is not None]
    median_vtravel = float(np.median(vtravels)) if vtravels else 0.0

    reasons = []
    manual = sorted(set(no_env_samples) | set(no_core_samples))
    if manual:
        strategy = "manual_review_required"
        reasons.append(f"{len(manual)} sample(s) lack a 700 envelope or 800 core")
    elif edge_touch_samples and len(edge_touch_samples) > 0.10 * len(per_track):
        strategy = "use_full_frame"
        reasons.append(f"{len(edge_touch_samples)} sample(s) touch the frame edge")
    elif cand_fraction >= 0.85:
        strategy = "use_full_frame"
        reasons.append(f"candidate ROI is {cand_fraction:.0%} of the full frame "
                       "after margin — cropping saves little")
    elif bg_ratio >= 0.70 or median_vtravel >= 0.10 * H:
        strategy = "global_roi_plus_tracking_window"
        reasons.append(
            f"a single fixed ROI is ~{bg_ratio:.0%} per-frame background (the melt pool "
            f"fills only ~{(1 - bg_ratio):.0%} of it) and the pool travels ~{median_vtravel:.0f}px "
            f"vertically: use the fixed global ROI for absolute position / trajectory / signed "
            f"offset, and a fixed-size tracking window (same size for all tracks, moving center) "
            f"for per-frame melt-pool morphology / width / gradients / asymmetry")
        reasons.append(
            "the single fixed global ROI (option A) remains valid and covers 100% if only "
            "position / coverage metrics are needed - the window is additive, not a replacement")
    else:
        strategy = "single_global_fixed_roi"
        reasons.append(f"candidate ROI covers 100% with background ratio "
                       f"~{bg_ratio:.0%} and modest travel (~{median_vtravel:.0f}px)")
    if edge_touch_samples and strategy != "use_full_frame":
        reasons.append(f"NOTE: edge-touch samples need review: {edge_touch_samples[:5]}")

    # --- repeatability per condition (19 rows) ---
    by_cond = {}
    for sid, t in per_track.items():
        by_cond.setdefault(t["condition_id"], []).append(t)
    repeat_rows = []
    worst_cond, worst_spread = None, -1.0
    for cond, grp in sorted(by_cond.items()):
        mxs = [g["mean_centroid_x"] for g in grp if g["mean_centroid_x"] is not None]
        mys = [g["mean_centroid_y"] for g in grp if g["mean_centroid_y"] is not None]
        vts = [g["centroid_vertical_travel_px"] for g in grp if g["centroid_vertical_travel_px"] is not None]
        ws = [bbox_width_height(g["track_bbox700"])[0] for g in grp]
        hs = [bbox_width_height(g["track_bbox700"])[1] for g in grp]
        any_touch = any(rect_touches_frame(g["track_bbox700"], H, W) and
                        any(rect_touches_frame(g["track_bbox700"], H, W).values()) for g in grp)
        std_x = float(np.std(mxs)) if mxs else float("nan")
        std_y = float(np.std(mys)) if mys else float("nan")
        spread = float(np.hypot(std_x if mxs else 0.0, std_y if mys else 0.0))
        if spread > worst_spread:
            worst_spread, worst_cond = spread, cond
        repeat_rows.append({
            "condition_id": cond,
            "n_tracks": len(grp),
            "mean_centroid_x": _f(float(np.mean(mxs)) if mxs else None),
            "mean_centroid_y": _f(float(np.mean(mys)) if mys else None),
            "std_centroid_x": _f(std_x), "std_centroid_y": _f(std_y),
            "centroid_position_spread_px": _f(spread),
            "vertical_travel_px_mean": _f(float(np.mean(vts)) if vts else None),
            "bbox700_width_px_mean": _f(float(np.mean(ws)) if ws else None),
            "bbox700_height_px_mean": _f(float(np.mean(hs)) if hs else None),
            "any_edge_touch": bool(any_touch),
            "position_inconsistency_flag": bool(spread > 0.15 * max(H, W)),
        })

    # --- exception list ---
    exc_rows = []
    for tr in track_rows:
        if tr["warnings"]:
            exc_rows.append({"sample_id": tr["sample_id"],
                             "condition_id": tr["condition_id"],
                             "track_id": tr["track_id"],
                             "issue": tr["warnings"],
                             "roi_status": tr["roi_status"]})
    for rr in repeat_rows:
        if rr["position_inconsistency_flag"]:
            exc_rows.append({"sample_id": "", "condition_id": rr["condition_id"],
                             "track_id": "", "issue": "condition_position_inconsistent",
                             "roi_status": "review"})

    results = {
        "track_rows": track_rows,
        "repeat_rows": repeat_rows,
        "exc_rows": exc_rows,
        "win_rows": win_rows,
        "per_track": per_track,          # masks/centroids/bboxes for QC figures
        "old_roi": old_roi,
        "candidate_rect": cand_rect,
        "recommended_window": (rec_win_h, rec_win_w),
        "summary": _build_summary(
            per_track, calib, old_roi, old_cov700_min, old_cov800_min,
            old_roi_usable, old_edge_fail, raw_global_700, raw_global_800, cand,
            cand_cov700_min, cand_cov800_min, cand_edge_fail, win_candidates,
            (rec_win_h, rec_win_w), rec_cov, H, W, px_x, px_y, vmin, vmax, hard,
            edge_margin, strategy, reasons, manual, edge_touch_samples,
            worst_cond, width_pcts, height_pcts),
    }
    return results


def _build_summary(per_track, calib, old_roi, old700, old800, old_usable,
                   old_edge_fail, raw700, raw800, cand, cand700, cand800,
                   cand_edge_fail, win_candidates, rec_win, rec_cov, H, W,
                   px_x, px_y, vmin, vmax, hard, edge_margin, strategy, reasons,
                   manual, edge_touch_samples, worst_cond, width_pcts, height_pcts):
    rec_win_h, rec_win_w = rec_win
    return {
        "sample_count": len(per_track),
        "effective_frame_rule": "frames[1:] (exclude startup frame 0; effective = frames 2..last)",
        "temperature_spatial_mask_policy": {
            "valid_range_C": [vmin, vmax], "hard_saturation_C": hard,
            "above_range": "kept if spatially continuous with main body; isolated splatter excluded",
            "hard_saturation_spatial": "filled within main body (mask only); raw never modified",
            "raw_matrix_modified": False,
        },
        "old_roi": {"top": old_roi[0], "left": old_roi[1],
                    "bottom_exclusive": old_roi[2], "right_exclusive": old_roi[3]},
        "old_roi_min_coverage_700": _f(old700),
        "old_roi_min_coverage_800": _f(old800),
        "old_roi_usable": bool(old_usable),
        "old_roi_edge_fail_samples": old_edge_fail,
        "raw_global_bbox_700": list(raw700) if raw700 else None,
        "raw_global_bbox_800": list(raw800) if raw800 else None,
        "recommended_global_roi": (None if cand is None else {
            "top": cand["candidate_top"], "left": cand["candidate_left"],
            "bottom_exclusive": cand["candidate_bottom_exclusive"],
            "right_exclusive": cand["candidate_right_exclusive"],
            "height": cand["candidate_height"], "width": cand["candidate_width"],
            "area_px": cand["candidate_area_px"],
            "area_mm2": _f(cand["candidate_area_px"] * px_x * px_y),
            "fraction_of_full_frame": _f(cand["candidate_fraction_of_full_frame"]),
        }),
        "global_roi_min_coverage_700": _f(cand700),
        "global_roi_min_coverage_800": _f(cand800),
        "global_roi_edge_fail_samples": cand_edge_fail,
        "bbox700_width_p95_px": _f(width_pcts[95]), "bbox700_width_p99_px": _f(width_pcts[99]),
        "bbox700_height_p95_px": _f(height_pcts[95]), "bbox700_height_p99_px": _f(height_pcts[99]),
        "tracking_window_candidates": [
            {"height_px": h, "width_px": w} for (h, w) in win_candidates],
        "recommended_tracking_window": {
            "height_px": rec_win_h, "width_px": rec_win_w,
            "height_mm": _f(rec_win_h * px_y), "width_mm": _f(rec_win_w * px_x)},
        "tracking_window_coverage_rate": _f(rec_cov["coverage_rate"]),
        "tracking_window_clipped_frame_count": rec_cov["clipped_frame_count"],
        "tracking_window_edge_adjusted_frame_count": rec_cov["edge_adjusted_frame_count"],
        "full_frame_dimensions": [H, W],
        "full_frame_physical_size_mm": [_f(H * px_y), _f(W * px_x)],
        "edge_touch_samples": edge_touch_samples,
        "worst_repeatability_condition": worst_cond,
        "recommended_strategy": strategy,
        "recommendation_reasons": reasons,
        "manual_review_samples": manual,
        "calibration_id": calib.calibration_id,
        "pixel_size_x_mm": px_x, "pixel_size_y_mm": px_y,
    }


# ---------------------------------------------------------------------------
# writers
# ---------------------------------------------------------------------------

def _write_csv(path, rows, columns):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in columns})


def write_outputs(results, output_root):
    tables = os.path.join(_abs(output_root), "tables")
    qc = os.path.join(_abs(output_root), "qc", "roi")
    os.makedirs(qc, exist_ok=True)

    tr = results["track_rows"]
    _write_csv(os.path.join(tables, "roi_bbox_by_track.csv"), tr, list(tr[0].keys()))
    _write_csv(os.path.join(tables, "roi_repeatability_summary.csv"),
               results["repeat_rows"], list(results["repeat_rows"][0].keys()))
    exc = results["exc_rows"] or [{"sample_id": "", "condition_id": "",
                                   "track_id": "", "issue": "none", "roi_status": "ok"}]
    _write_csv(os.path.join(tables, "roi_exception_list.csv"), exc, list(exc[0].keys()))
    _write_csv(os.path.join(tables, "tracking_window_coverage_summary.csv"),
               results["win_rows"], list(results["win_rows"][0].keys()))

    with open(os.path.join(qc, "roi_strategy_summary.json"), "w", encoding="utf-8") as f:
        json.dump(results["summary"], f, indent=2, ensure_ascii=False)
    return tables, qc


def build_argparser():
    p = argparse.ArgumentParser(description="Unified ROI strategy evaluation (no ROI matrices written).")
    p.add_argument("--master-csv", default="data/metadata/experiment_master.csv")
    p.add_argument("--matrix-root", default="data/processed/matrix")
    p.add_argument("--calibration-config", default="configs/physical_calibration.yaml")
    p.add_argument("--output-root", default="results")
    p.add_argument("--envelope-threshold", type=float, default=700.0)
    p.add_argument("--core-threshold", type=float, default=800.0)
    p.add_argument("--min-component-area", type=int, default=9)
    p.add_argument("--global-margin", type=int, default=20)
    p.add_argument("--edge-safety-margin", type=int, default=10)
    p.add_argument("--tracking-window-height", type=int, default=None)
    p.add_argument("--tracking-window-width", type=int, default=None)
    p.add_argument("--round-to", type=int, default=8)
    p.add_argument("--dry-run", action="store_true",
                   help="evaluate but write NO tables/JSON/figures (never writes ROI .npy)")
    p.add_argument("--no-figures", action="store_true", help="skip QC figure generation")
    p.add_argument("--max-tracks", type=int, default=None, help="limit tracks (debug)")
    p.add_argument("--max-frames-per-track", type=int, default=None, help="limit frames (debug)")
    return p


def main(argv=None):
    args = build_argparser().parse_args(argv)
    results = run(args)
    s = results["summary"]

    print("-" * 64)
    print(f"[03a] samples evaluated     : {s['sample_count']}")
    print(f"[03a] old ROI min cov 700/800: {s['old_roi_min_coverage_700']} / {s['old_roi_min_coverage_800']}  usable={s['old_roi_usable']}")
    print(f"[03a] raw global bbox 700   : {s['raw_global_bbox_700']}")
    print(f"[03a] candidate global ROI  : {s['recommended_global_roi']}")
    print(f"[03a] candidate min cov 700/800: {s['global_roi_min_coverage_700']} / {s['global_roi_min_coverage_800']}")
    print(f"[03a] recommended tracking window: {s['recommended_tracking_window']} cov={s['tracking_window_coverage_rate']}")
    print(f"[03a] RECOMMENDED STRATEGY  : {s['recommended_strategy']}")
    for r in s["recommendation_reasons"]:
        print(f"        - {r}")
    if s["manual_review_samples"]:
        print(f"[03a] manual-review samples : {s['manual_review_samples']}")

    if args.dry_run:
        print("[03a] DRY-RUN: no tables/JSON/figures written; no ROI matrices created.")
        return results

    tables, qc = write_outputs(results, args.output_root)
    print(f"[03a] tables -> {tables}")
    print(f"[03a] json   -> {os.path.join(qc, 'roi_strategy_summary.json')}")

    if not args.no_figures:
        try:
            from visualization.roi_qc_figures import generate_all_figures
            generate_all_figures(results, args.matrix_root, args.master_csv,
                                  args.calibration_config, qc,
                                  envelope_threshold=args.envelope_threshold,
                                  core_threshold=args.core_threshold,
                                  min_component_area=args.min_component_area)
            print(f"[03a] figures -> {qc}")
        except Exception as e:   # figures are QC-only; never fail the evaluation
            print(f"[03a] WARNING: figure generation skipped ({type(e).__name__}: {e})")
    print("[03a] Reminder: NO ROI .npy written; raw matrices read-only / unmodified.")
    return results


if __name__ == "__main__":
    main()
