"""
00_build_experiment_master.py

Build the per-single-track metadata map (experiment_master.csv) from:
  - the formal design          configs/experiments.yaml   (process parameters)
  - the formal calibration     configs/physical_calibration.yaml (spatial scale)
  - the REAL raw-data tree     (folder / session / frame counts)

One row per single track -> 19 conditions x 3 tracks = 57 rows.

- Process parameters come from configs/experiments.yaml (the single source of
  truth) plus its `process_parameters` provenance block.
- Spatial calibration values come from configs/physical_calibration.yaml via the
  validated loader (src/config/physical_calibration.py) -> a single active
  calibration, never the legacy pixel size.
- READ-ONLY with respect to raw data: only lists files, never opens/modifies.
- Output (data/metadata/experiment_master.csv) is LOCAL ONLY (git-ignored *.csv).

Unknown metadata is left BLANK on purpose (never guessed): frame_rate_fps,
effective_start_frame, effective_end_frame, scan_axis, scan_direction, plate_id.

Usage:
    python scripts/00_build_experiment_master.py \
        --config configs/experiments.yaml \
        --calibration-config configs/physical_calibration.yaml
"""

import os
import sys
import csv
import glob
import argparse

import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from config.physical_calibration import load_physical_calibration

# Exact column order for experiment_master.csv (one row per single track).
COLUMNS = [
    "condition_id", "track_id", "sample_id", "design_role",
    "laser_power_W", "scan_speed_mm_min", "magnetic_field_mT",
    "powder_feed_g_min", "travel_distance_mm",
    "process_parameter_source", "process_parameter_status",
    "calibration_id", "calibration_status", "calibration_source",
    "pixel_size_x_mm", "pixel_size_y_mm", "pixel_area_mm2",
    "isotropic_scaling_assumed",
    "raw_folder", "session_xml", "xtherm_count",
    "frame_rate_fps", "effective_start_frame", "effective_end_frame",
    "scan_axis", "scan_direction",
    "track_order", "plate_id", "processing_status", "notes",
]

# Fields intentionally left blank until confirmed from real records.
BLANK_FIELDS = (
    "frame_rate_fps", "effective_start_frame", "effective_end_frame",
    "scan_axis", "scan_direction", "plate_id",
)

PROCESSING_STATUS = "raw_only"


def load_experiments(config_path):
    if not os.path.isabs(config_path):
        config_path = os.path.join(_ROOT, config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def count_xtherm(folder):
    """Number of .xtherm files directly under folder (read-only, not recursive)."""
    if not os.path.isdir(folder):
        return 0
    found = set()
    for pat in ("*.xtherm", "*.XTHERM"):
        found.update(glob.glob(os.path.join(folder, pat)))
    return len(found)


def build_rows(cfg, calib, proc):
    """Build the 57 track rows from config + calibration + real raw-data folders."""
    raw_root = cfg["raw_data_root"]
    track_order_map = cfg.get("track_order_map", {"T1": 1, "T2": 2, "T3": 3})

    sp = calib.spatial
    calib_fields = {
        "calibration_id": calib.calibration_id,
        "calibration_status": sp["calibration_status"],
        "calibration_source": sp["calibration_source"],
        "pixel_size_x_mm": calib.pixel_size_x_mm,
        "pixel_size_y_mm": calib.pixel_size_y_mm,
        "pixel_area_mm2": calib.pixel_area_mm2,
        # textual lower-case to match the documented form (true/false)
        "isotropic_scaling_assumed": str(bool(calib.isotropic)).lower(),
    }

    rows = []
    warnings = []
    for cond in cfg["conditions"]:
        cid = cond["condition_id"]
        for track in cond["tracks"]:
            folder = "/".join([raw_root.rstrip("/"), cid, track])
            session_path = "/".join([folder, "session.xml"])
            session_val = session_path if os.path.isfile(session_path) else ""
            xcount = count_xtherm(folder)

            if not os.path.isdir(folder):
                warnings.append(f"MISSING raw folder: {folder}")
            elif xcount == 0:
                warnings.append(f"NO .xtherm in: {folder}")
            if os.path.isdir(folder) and not session_val:
                warnings.append(f"MISSING session.xml in: {folder}")

            row = {
                "condition_id": cid,
                "track_id": track,
                "sample_id": f"{cid}_{track}",
                "design_role": cond["design_role"],
                "laser_power_W": cond["laser_power_W"],
                "scan_speed_mm_min": cond["scan_speed_mm_min"],
                "magnetic_field_mT": cond["magnetic_field_mT"],
                "powder_feed_g_min": cond["powder_feed_g_min"],
                "travel_distance_mm": cond["travel_distance_mm"],
                "process_parameter_source": proc["source"],
                "process_parameter_status": proc["status"],
                "raw_folder": folder,
                "session_xml": session_val,
                "xtherm_count": xcount,
                "track_order": track_order_map[track],
                "processing_status": PROCESSING_STATUS,
                "notes": "",
            }
            row.update(calib_fields)
            for blank in BLANK_FIELDS:
                row[blank] = ""
            rows.append(row)
    return rows, warnings


def write_csv(rows, output_path):
    if not os.path.isabs(output_path):
        output_path = os.path.join(_ROOT, output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--calibration-config",
                        default="configs/physical_calibration.yaml")
    parser.add_argument("--output", default="data/metadata/experiment_master.csv")
    args = parser.parse_args()

    cfg = load_experiments(args.config)
    proc = cfg.get("process_parameters",
                   {"source": "", "status": ""})
    calib = load_physical_calibration(args.calibration_config)   # validates

    rows, warnings = build_rows(cfg, calib, proc)
    out = write_csv(rows, args.output)

    n_conditions = len({r["condition_id"] for r in rows})
    n_sample_ids = len({r["sample_id"] for r in rows})
    total_xtherm = sum(int(r["xtherm_count"]) for r in rows)
    n_session = sum(1 for r in rows if r["session_xml"])
    n_proc = sum(1 for r in rows if r["process_parameter_status"] == "confirmed")
    n_calib = sum(1 for r in rows if r["calibration_status"] == "confirmed")

    print("[00] experiment_master built (LOCAL ONLY; not committed)")
    print("-" * 64)
    print(f"[00] output             : {out}")
    print(f"[00] conditions         : {n_conditions}")
    print(f"[00] track rows         : {len(rows)}")
    print(f"[00] unique sample_id   : {n_sample_ids}")
    print(f"[00] session.xml found  : {n_session}")
    print(f"[00] total .xtherm      : {total_xtherm}")
    print(f"[00] process confirmed  : {n_proc}")
    print(f"[00] calibration done   : {n_calib}  (id={calib.calibration_id}, "
          f"px={calib.pixel_size_x_mm} mm, area={calib.pixel_area_mm2} mm^2)")
    print(f"[00] processing_status  : all '{PROCESSING_STATUS}'")
    print(f"[00] blank fields       : {', '.join(BLANK_FIELDS)} (by design)")

    if warnings:
        print("-" * 64)
        print(f"[00] WARNING: {len(warnings)} issue(s) with raw folders:")
        for w in warnings[:20]:
            print(f"        {w}")
        if len(warnings) > 20:
            print(f"        ... and {len(warnings) - 20} more")
    print("-" * 64)
    print("[00] Reminder: raw .xtherm files are only listed, never modified.")


if __name__ == "__main__":
    main()
