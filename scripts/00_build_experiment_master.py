"""
00_build_experiment_master.py

Build the per-single-track metadata map (experiment_master.csv) from:
  - the formal design          configs/experiments.yaml   (process parameters,
                               fixed hardware params, powder feed, substrate,
                               track-repetition / cooling)
  - the formal calibration     configs/physical_calibration.yaml (spatial scale,
                               image geometry / scan direction, temperature
                               measurement range, frame rate, working distance,
                               effective-frame rule)
  - the REAL raw-data tree     (folder / session / frame counts)

One row per single track -> 19 conditions x 3 tracks = 57 rows.

- All formal values come from the two committed config files (single sources of
  truth). Nothing here is guessed: parameters that are genuinely not recorded
  (emissivity, transmission, actual powder feed) are written BLANK.
- Spatial calibration + image geometry + acquisition values come from
  configs/physical_calibration.yaml via the validated loader
  (src/config/physical_calibration.py) -> one active calibration, never legacy.
- READ-ONLY with respect to raw data: only lists files, never opens/modifies.
- Output (data/metadata/experiment_master.csv) is LOCAL ONLY (git-ignored *.csv).

Logical plate id: each condition uses ONE independent 316L plate; its three
single tracks (T1/T2/T3) share that plate -> plate_id = "Plate-<condition_id>".
This is a data-management identifier, NOT a physical stamp on the plate.

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
    # --- identity ---
    "condition_id", "track_id", "sample_id", "design_role", "track_order",
    # --- substrate / logical plate ---
    "plate_id", "plate_id_type",
    "substrate_material",
    "substrate_length_mm", "substrate_width_mm", "substrate_thickness_mm",
    # --- response-surface factors ---
    "laser_power_W", "scan_speed_mm_min", "magnetic_field_mT",
    # --- powder feed (set vs actual) ---
    "powder_feed_g_min",            # backward-compat alias of the SET value
    "powder_feed_set_g_min", "powder_feed_actual_g_min", "powder_feed_actual_status",
    # --- fixed laser / optics + run geometry ---
    "laser_spot_diameter_mm", "defocus_mm", "defocus_sign",
    "travel_distance_mm", "cooling_interval_between_tracks_s",
    # --- process-parameter provenance ---
    "process_parameter_source", "process_parameter_status",
    # --- acquisition / time (frame index based) ---
    "frame_rate_fps",
    "effective_start_frame_number_1_based", "effective_start_frame_index_0_based",
    "effective_end_rule", "excluded_initial_frame_reason",
    # --- image geometry / scan direction ---
    "scan_axis", "transverse_axis",
    "image_scan_direction", "array_scan_direction", "physical_to_array_y_sign",
    "scan_direction",               # backward-compat alias of image_scan_direction
    # --- spatial calibration ---
    "calibration_id", "calibration_status", "calibration_source",
    "calibration_reference_axis",
    "pixel_size_x_mm", "pixel_size_y_mm", "pixel_area_mm2", "isotropic_scaling_assumed",
    # --- temperature measurement range ---
    "valid_temperature_min_C", "valid_temperature_max_C", "hard_saturation_threshold_C",
    # --- camera / optics ---
    "working_distance_mm", "working_distance_reference",
    # --- radiometric inputs (not recorded) ---
    "emissivity", "transmission",
    # --- raw-data location (read-only listing) ---
    "raw_folder", "session_xml", "xtherm_count",
    # --- status ---
    "processing_status", "metadata_status", "notes",
]

PROCESSING_STATUS = "raw_only"
METADATA_STATUS = "confirmed_core_parameters"


def _blank(value):
    """Render None as an empty CSV cell (intentionally-blank, not guessed)."""
    return "" if value is None else value


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

    fixed = cfg.get("fixed_process_parameters", {})
    powder = cfg.get("powder_feed", {})
    substrate = cfg.get("substrate", {})
    repetition = cfg.get("track_repetition", {})

    sp = calib.spatial

    # Values that are identical for every track (formal, confirmed).
    common = {
        # powder feed
        "powder_feed_set_g_min": powder.get("powder_feed_set_g_min"),
        "powder_feed_actual_g_min": powder.get("powder_feed_actual_g_min"),  # None -> blank
        "powder_feed_actual_status": powder.get("powder_feed_actual_status"),
        # fixed laser / optics
        "laser_spot_diameter_mm": fixed.get("laser_spot_diameter_mm"),
        "defocus_mm": fixed.get("defocus_mm"),
        "defocus_sign": fixed.get("defocus_sign"),
        "cooling_interval_between_tracks_s": repetition.get("cooling_interval_between_tracks_s"),
        # process provenance
        "process_parameter_source": proc.get("source", ""),
        "process_parameter_status": proc.get("status", ""),
        # substrate
        "plate_id_type": substrate.get("plate_id_type"),
        "substrate_material": substrate.get("material"),
        "substrate_length_mm": substrate.get("length_mm"),
        "substrate_width_mm": substrate.get("width_mm"),
        "substrate_thickness_mm": substrate.get("thickness_mm"),
        # acquisition / time (frame-index based)
        "frame_rate_fps": calib.frame_rate_fps,
        "effective_start_frame_number_1_based":
            calib.effective_frame_rule.get("start_frame_number_1_based"),
        "effective_start_frame_index_0_based":
            calib.effective_frame_rule.get("start_frame_index_0_based"),
        "effective_end_rule": calib.effective_frame_rule.get("end_rule"),
        "excluded_initial_frame_reason":
            calib.effective_frame_rule.get("excluded_initial_frame_reason"),
        # image geometry / scan direction
        "scan_axis": calib.scan_axis,
        "transverse_axis": calib.transverse_axis,
        "image_scan_direction": calib.image_scan_direction,
        "array_scan_direction": calib.array_scan_direction,
        "physical_to_array_y_sign": calib.physical_to_array_y_sign,
        "scan_direction": calib.scan_direction,
        # spatial calibration
        "calibration_id": calib.calibration_id,
        "calibration_status": sp["calibration_status"],
        "calibration_source": sp["calibration_source"],
        "calibration_reference_axis": calib.calibration_reference_axis,
        "pixel_size_x_mm": calib.pixel_size_x_mm,
        "pixel_size_y_mm": calib.pixel_size_y_mm,
        "pixel_area_mm2": calib.pixel_area_mm2,
        "isotropic_scaling_assumed": str(bool(calib.isotropic)).lower(),
        # temperature measurement range
        "valid_temperature_min_C": calib.valid_temperature_min_C,
        "valid_temperature_max_C": calib.valid_temperature_max_C,
        "hard_saturation_threshold_C": calib.hard_saturation_threshold_C,
        # camera / optics
        "working_distance_mm": calib.working_distance_mm,
        "working_distance_reference": calib.working_distance_reference,
        # radiometric inputs (not recorded -> blank)
        "emissivity": calib.temperature.get("emissivity"),
        "transmission": calib.temperature.get("transmission"),
        # status
        "processing_status": PROCESSING_STATUS,
        "metadata_status": METADATA_STATUS,
        "notes": "",
    }

    rows = []
    warnings = []
    for cond in cfg["conditions"]:
        cid = cond["condition_id"]
        plate_id = f"Plate-{cid}"
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
                "track_order": track_order_map[track],
                "plate_id": plate_id,
                "laser_power_W": cond["laser_power_W"],
                "scan_speed_mm_min": cond["scan_speed_mm_min"],
                "magnetic_field_mT": cond["magnetic_field_mT"],
                # backward-compat alias of the SET powder feed value
                "powder_feed_g_min": cond["powder_feed_g_min"],
                "travel_distance_mm": cond["travel_distance_mm"],
                "raw_folder": folder,
                "session_xml": session_val,
                "xtherm_count": xcount,
            }
            row.update(common)
            rows.append({k: _blank(row.get(k)) for k in COLUMNS})
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
    proc = cfg.get("process_parameters", {"source": "", "status": ""})
    calib = load_physical_calibration(args.calibration_config)   # validates

    rows, warnings = build_rows(cfg, calib, proc)
    out = write_csv(rows, args.output)

    n_conditions = len({r["condition_id"] for r in rows})
    n_sample_ids = len({r["sample_id"] for r in rows})
    n_plates = len({r["plate_id"] for r in rows})
    tracks_per_plate = {p: 0 for p in {r["plate_id"] for r in rows}}
    for r in rows:
        tracks_per_plate[r["plate_id"]] += 1
    plates_with_3 = sum(1 for v in tracks_per_plate.values() if v == 3)
    total_xtherm = sum(int(r["xtherm_count"]) for r in rows)
    n_session = sum(1 for r in rows if r["session_xml"])

    print("[00] experiment_master built (LOCAL ONLY; not committed)")
    print("-" * 64)
    print(f"[00] output             : {out}")
    print(f"[00] conditions         : {n_conditions}")
    print(f"[00] track rows         : {len(rows)}")
    print(f"[00] unique sample_id   : {n_sample_ids}")
    print(f"[00] unique plate_id    : {n_plates}  (expect 19; 1 plate/condition)")
    print(f"[00] plates with 3 tks  : {plates_with_3}/{n_plates}")
    print(f"[00] session.xml found  : {n_session}")
    print(f"[00] total .xtherm      : {total_xtherm}")
    print(f"[00] frame_rate_fps     : {calib.frame_rate_fps} (confirmed)")
    print(f"[00] scan_axis          : {calib.scan_axis} / {calib.image_scan_direction} "
          f"(array {calib.array_scan_direction}, y-sign {calib.physical_to_array_y_sign})")
    print(f"[00] valid temp range C : {calib.valid_temperature_min_C}..{calib.valid_temperature_max_C}")
    print(f"[00] working distance   : {calib.working_distance_mm} mm "
          f"({calib.working_distance_reference})")
    print(f"[00] calibration        : id={calib.calibration_id}, "
          f"px={calib.pixel_size_x_mm} mm, axis={calib.calibration_reference_axis}")
    print(f"[00] blank by design    : powder_feed_actual_g_min, emissivity, transmission")

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
