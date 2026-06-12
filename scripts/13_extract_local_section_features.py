"""
13_extract_local_section_features.py

Extract LOCAL thermal-field features for every cross-section listed in the
section plan. The ML sample unit is a cross-section position (e.g. R01_T1_S1),
NOT a frame.

Input:
  - data/metadata/section_plan.csv   (LOCAL, git-ignored; see
    docs/section_level_ml_dataset.md). One row per section, with the section
    position, the ROI file it maps to, and the process parameters.
  - data/processed/roi/<roi_file>    ROI temperature matrices (N x H x W,
    float32 Celsius) produced by script 03.

Output:
  - results/tables/local_section_features.csv
    One row per cross-section: ids + process params + local_* features.

This script never parses .xtherm and never modifies raw data. If the section
plan is missing it prints clear guidance and exits WITHOUT fabricating data.

Usage:
    python scripts/13_extract_local_section_features.py --config configs/default.yaml
"""

import os
import sys
import csv
import argparse

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from utils.config import load_config
from features.local_section_features import (
    extract_local_section_features, LOCAL_FEATURE_KEYS, MIN_WINDOW_FRAMES,
)

# Identity / process columns carried straight through from the plan.
_ID_COLS = ["sample_id", "experiment_id", "track_id", "section_id",
            "section_position_mm"]
_PROCESS_COLS = ["laser_power_W", "scan_speed_mm_min", "powder_feed_g_min",
                 "magnetic_field_mT"]


def _resolve_window_half_frames(row, scfg, n_frames, travel_mm):
    """Per-row window half-width in frames (mm-based if configured)."""
    if bool(scfg.get("use_window_half_mm", False)):
        half_mm = row.get("window_half_mm", None)
        if half_mm is None or (isinstance(half_mm, float) and np.isnan(half_mm)):
            half_mm = scfg.get("default_window_half_mm", None)
        if half_mm is not None:
            frames_per_mm = (n_frames - 1) / float(travel_mm) if travel_mm > 0 else 0.0
            return int(round(float(half_mm) * frames_per_mm))
    val = row.get("window_half_frames", None)
    if val is None or (isinstance(val, float) and np.isnan(val)):
        val = scfg.get("default_window_half_frames", 10)
    return int(val)


def _row_value(row, key, default=None):
    val = row.get(key, default)
    if val is None:
        return default
    if isinstance(val, float) and np.isnan(val):
        return default
    return val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    scfg = config.get("section_samples", {}) or {}
    tfcfg = config.get("thermal_field_features", {}) or {}
    roi_dir = tfcfg.get("input_dir", config["paths"]["processed_roi"])

    plan_path = scfg.get("section_plan_csv",
                         "data/metadata/section_plan.csv")
    out_table = scfg.get(
        "local_feature_output",
        os.path.join(config["paths"]["results_tables"], "local_section_features.csv"))
    default_travel = float(scfg.get("default_travel_distance_mm", 30.0))
    default_fps = float(tfcfg.get("frame_rate_fps", 1000))

    # 1. Section plan (LOCAL file — do NOT fabricate if missing).
    if not os.path.exists(plan_path):
        print(f"[13] Section plan not found: {plan_path}")
        print("[13] Create it locally from docs/section_level_ml_dataset.md "
              "(one row per cross-section, key column: sample_id).")
        print("[13] This file stays LOCAL and is not uploaded. NOT generating "
              "any pseudo data.")
        return
    plan = pd.read_csv(plan_path)

    required = {"sample_id", "experiment_id", "roi_file", "section_position_mm"}
    missing_cols = required - set(plan.columns)
    if missing_cols:
        raise SystemExit(
            f"[13] section plan {plan_path} is missing required column(s): "
            f"{sorted(missing_cols)} (see docs/section_level_ml_dataset.md).")

    # 2. sample_id must be unique.
    ids = plan["sample_id"].astype(str)
    dups = sorted(ids[ids.duplicated()].unique())
    if dups:
        raise SystemExit(
            f"[13] duplicate sample_id(s) in {plan_path}: {dups}. "
            f"Every cross-section sample_id must be unique.")

    # 3. All referenced ROI files must exist.
    roi_files = sorted(plan["roi_file"].astype(str).unique())
    missing_roi = [rf for rf in roi_files
                   if not os.path.exists(os.path.join(roi_dir, rf))]
    if missing_roi:
        raise SystemExit(
            f"[13] {len(missing_roi)} ROI file(s) referenced by the plan are "
            f"missing under {roi_dir}: {missing_roi}. Run scripts 02->03 first "
            f"or fix the roi_file column.")

    print(f"[13] section plan: {len(plan)} section(s) from {plan_path}")
    print(f"[13] ROI source  : {roi_dir} ({len(roi_files)} distinct file(s))")

    roi_cache = {}
    rows, n_small = [], 0
    for _, row in plan.iterrows():
        rd = row.to_dict()
        sample_id = str(rd["sample_id"])
        roi_file = str(rd["roi_file"])

        if roi_file not in roi_cache:
            roi_cache[roi_file] = np.load(
                os.path.join(roi_dir, roi_file)).astype(np.float32)
        roi = roi_cache[roi_file]
        n_frames = roi.shape[0]

        travel_mm = float(_row_value(rd, "travel_distance_mm", default_travel))
        fps = float(_row_value(rd, "frame_rate_fps", default_fps))
        half_frames = _resolve_window_half_frames(rd, scfg, n_frames, travel_mm)
        pos_mm = float(rd["section_position_mm"])

        feats = extract_local_section_features(
            roi, section_position_mm=pos_mm, travel_distance_mm=travel_mm,
            window_half_frames=half_frames, frame_rate_fps=fps, config=config)

        if feats["window_frame_count"] < MIN_WINDOW_FRAMES:
            n_small += 1
            print(f"      WARNING {sample_id}: local window has only "
                  f"{feats['window_frame_count']} frame(s) "
                  f"(< {MIN_WINDOW_FRAMES}).")

        out = {}
        for k in _ID_COLS:
            out[k] = rd.get(k, "")
        for k in _PROCESS_COLS:
            out[k] = _row_value(rd, k, "")
        out["roi_file"] = roi_file
        out["travel_distance_mm"] = travel_mm
        out["frame_rate_fps"] = fps
        out["window_half_frames"] = half_frames
        out.update(feats)
        rows.append(out)
        print(f"      {sample_id:14s} exp={str(rd['experiment_id']):20s} "
              f"frames[{feats['frame_start']}:{feats['frame_end']}] "
              f"Tpeak={feats['local_peak_temperature']:.1f}C")

    # Column order: ids, process params, bookkeeping, then local_* features.
    lead = (_ID_COLS + _PROCESS_COLS +
            ["roi_file", "travel_distance_mm", "frame_rate_fps",
             "window_half_frames", "frame_center", "frame_start", "frame_end",
             "window_frame_count"])
    fieldnames = lead + LOCAL_FEATURE_KEYS

    os.makedirs(os.path.dirname(out_table) or ".", exist_ok=True)
    with open(out_table, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    n_exp = plan["experiment_id"].astype(str).nunique()
    print("-" * 60)
    print(f"[13] loaded section samples : {len(rows)}")
    print(f"[13] unique experiments     : {n_exp}")
    print(f"[13] output path            : {out_table}")
    print(f"[13] feature columns        : {LOCAL_FEATURE_KEYS}")
    if n_small:
        print(f"[13] NOTE: {n_small} section(s) had a window < {MIN_WINDOW_FRAMES} "
              f"frames — features still written, interpret cooling/fluctuation "
              f"with care.")


if __name__ == "__main__":
    main()
