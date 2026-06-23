"""
00b_build_metadata_audit.py

Emit three LOCAL audit tables that summarise the confirmation status of the
formal metadata after the parameter freeze. Read-only over the configs; writes
only to results/tables/ (git-ignored). No ROI, no feature extraction, no matrix
access.

Outputs (LOCAL ONLY, not committed):
  results/tables/physical_metadata_audit.csv      parameter-level status
  results/tables/track_process_parameter_audit.csv 57-row per-track parameters
  results/tables/calibration_reference_audit.csv  spatial-calibration provenance

Sources of truth:
  configs/experiments.yaml            (process / fixed / substrate / repetition)
  configs/physical_calibration.yaml   (spatial + geometry + temperature + time)

Usage:
    python scripts/00b_build_metadata_audit.py
"""

import os
import sys
import csv
import argparse

import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from config.physical_calibration import load_physical_calibration


def _load_yaml(path):
    if not os.path.isabs(path):
        path = os.path.join(_ROOT, path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write(path, fieldnames, rows):
    if not os.path.isabs(path):
        path = os.path.join(_ROOT, path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def physical_metadata_audit(exp, calib):
    fixed = exp.get("fixed_process_parameters", {})
    powder = exp.get("powder_feed", {})
    substrate = exp.get("substrate", {})
    repetition = exp.get("track_repetition", {})
    temp = calib.temperature

    rows = [
        ("spatial_calibration",
         f"150.2 px = 5 mm -> {calib.pixel_size_x_mm} mm/px (axis={calib.calibration_reference_axis})",
         "confirmed", "user_confirmed"),
        ("scan_axis", calib.scan_axis, "confirmed", "user_confirmed"),
        ("image_scan_direction", calib.image_scan_direction, "confirmed", "user_confirmed"),
        ("array_scan_direction", calib.array_scan_direction, "confirmed", "user_confirmed"),
        ("physical_to_array_y_sign", calib.physical_to_array_y_sign, "confirmed", "user_confirmed"),
        ("frame_rate_fps", calib.frame_rate_fps, "confirmed",
         "user_confirmed_experimental_setting"),
        ("effective_frame_rule",
         f"start_1based={calib.effective_frame_rule.get('start_frame_number_1_based')}, "
         f"start_0based={calib.effective_frame_rule.get('start_frame_index_0_based')}, "
         f"end={calib.effective_frame_rule.get('end_rule')}",
         "confirmed", "user_confirmed"),
        ("valid_temperature_range_C",
         f"{calib.valid_temperature_min_C}..{calib.valid_temperature_max_C}",
         "confirmed", "user_confirmed"),
        ("hard_saturation_threshold_C", calib.hard_saturation_threshold_C,
         "confirmed", "user_confirmed"),
        ("working_distance_mm", calib.working_distance_mm, "confirmed", "user_confirmed"),
        ("working_distance_reference", calib.working_distance_reference,
         "confirmed", "user_confirmed"),
        ("substrate_material", substrate.get("material"), "confirmed", "user_confirmed"),
        ("substrate_size_mm",
         f"{substrate.get('length_mm')} x {substrate.get('width_mm')} x {substrate.get('thickness_mm')}",
         "confirmed", "user_confirmed"),
        ("plate_assignment",
         f"{substrate.get('plate_count')} plates, {substrate.get('tracks_per_plate')} tracks/plate, "
         f"one_plate_per_condition={substrate.get('one_plate_per_condition')}",
         "confirmed", "user_confirmed"),
        ("laser_spot_diameter_mm", fixed.get("laser_spot_diameter_mm"),
         "confirmed", "user_confirmed"),
        ("defocus_mm",
         f"{fixed.get('defocus_mm')} ({fixed.get('defocus_sign')}: "
         f"{fixed.get('positive_defocus_definition')})",
         "confirmed", "user_confirmed"),
        ("cooling_interval_between_tracks_s",
         repetition.get("cooling_interval_between_tracks_s"), "confirmed", "user_confirmed"),
        ("powder_feed_set_g_min", powder.get("powder_feed_set_g_min"),
         "confirmed", "equipment_setpoint"),
        ("powder_feed_actual_g_min", powder.get("powder_feed_actual_g_min"),
         "not_measured", "not_recorded"),
        ("emissivity", temp.get("emissivity"), "not_recorded", "not_recorded"),
        ("transmission", temp.get("transmission"), "not_recorded", "not_recorded"),
        ("exposure_time_us", calib.acquisition.get("exposure_time_us"),
         "not_recorded", "not_recorded"),
        ("lens_model", calib.acquisition.get("lens_model"), "not_recorded", "not_recorded"),
    ]
    fieldnames = ["parameter", "value", "status", "source"]
    out = [{"parameter": p, "value": "" if v is None else v, "status": s, "source": src}
           for (p, v, s, src) in rows]
    return fieldnames, out


def track_process_parameter_audit(exp, calib):
    fixed = exp.get("fixed_process_parameters", {})
    powder = exp.get("powder_feed", {})
    repetition = exp.get("track_repetition", {})
    track_order_map = exp.get("track_order_map", {"T1": 1, "T2": 2, "T3": 3})
    proc = exp.get("process_parameters", {})

    fieldnames = [
        "sample_id", "condition_id", "track_id", "track_order", "plate_id",
        "laser_power_W", "scan_speed_mm_min", "magnetic_field_mT",
        "powder_feed_set_g_min", "powder_feed_actual_g_min",
        "laser_spot_diameter_mm", "defocus_mm", "defocus_sign",
        "travel_distance_mm", "cooling_interval_between_tracks_s",
        "frame_rate_fps", "process_parameter_status",
    ]
    rows = []
    for cond in exp["conditions"]:
        cid = cond["condition_id"]
        for t in cond["tracks"]:
            actual = powder.get("powder_feed_actual_g_min")
            rows.append({
                "sample_id": f"{cid}_{t}",
                "condition_id": cid,
                "track_id": t,
                "track_order": track_order_map[t],
                "plate_id": f"Plate-{cid}",
                "laser_power_W": cond["laser_power_W"],
                "scan_speed_mm_min": cond["scan_speed_mm_min"],
                "magnetic_field_mT": cond["magnetic_field_mT"],
                "powder_feed_set_g_min": powder.get("powder_feed_set_g_min"),
                "powder_feed_actual_g_min": "" if actual is None else actual,
                "laser_spot_diameter_mm": fixed.get("laser_spot_diameter_mm"),
                "defocus_mm": fixed.get("defocus_mm"),
                "defocus_sign": fixed.get("defocus_sign"),
                "travel_distance_mm": cond["travel_distance_mm"],
                "cooling_interval_between_tracks_s":
                    repetition.get("cooling_interval_between_tracks_s"),
                "frame_rate_fps": calib.frame_rate_fps,
                "process_parameter_status": proc.get("status", ""),
            })
    return fieldnames, rows


def calibration_reference_audit(cal_cfg, calib):
    sp = cal_cfg["spatial_calibration"]
    legacy = cal_cfg.get("legacy_calibration", {})
    rows = [
        ("calibration_reference", "150.2 px = 5 mm", "confirmed", "user_confirmed"),
        ("calibration_reference_axis", calib.calibration_reference_axis,
         "confirmed", "user_confirmed"),
        ("pixel_size_x_mm", calib.pixel_size_x_mm, "confirmed", "assumed_equal_to_y"),
        ("pixel_size_y_mm", calib.pixel_size_y_mm, "confirmed", "measured"),
        ("pixel_area_mm2", calib.pixel_area_mm2, "confirmed", "derived"),
        ("isotropic_scaling_assumed", str(bool(calib.isotropic)).lower(),
         "confirmed", "assumption"),
        ("x_y_anisotropy_verified", str(bool(sp.get("x_y_anisotropy_verified"))).lower(),
         "not_verified", "n/a"),
        ("physical_to_array_y_sign", calib.physical_to_array_y_sign,
         "confirmed", "user_confirmed"),
        ("legacy_pixel_size_mm",
         f"{legacy.get('approximate_pixel_size_mm')} (95.9 px = 3 mm)",
         "legacy_pilot_only_not_active", "legacy"),
        ("legacy_enabled_for_formal_processing",
         str(bool(legacy.get("enabled_for_formal_processing", False))).lower(),
         "disabled", "policy"),
    ]
    fieldnames = ["parameter", "value", "status", "source"]
    out = [{"parameter": p, "value": "" if v is None else v, "status": s, "source": src}
           for (p, v, s, src) in rows]
    return fieldnames, out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--calibration-config",
                        default="configs/physical_calibration.yaml")
    parser.add_argument("--out-dir", default="results/tables")
    args = parser.parse_args()

    exp = _load_yaml(args.config)
    cal_cfg = _load_yaml(args.calibration_config)
    calib = load_physical_calibration(args.calibration_config)   # validates

    p1 = _write(os.path.join(args.out_dir, "physical_metadata_audit.csv"),
                *physical_metadata_audit(exp, calib))
    p2 = _write(os.path.join(args.out_dir, "track_process_parameter_audit.csv"),
                *track_process_parameter_audit(exp, calib))
    p3 = _write(os.path.join(args.out_dir, "calibration_reference_audit.csv"),
                *calibration_reference_audit(cal_cfg, calib))

    print("[00b] metadata audit tables written (LOCAL ONLY; not committed)")
    for p in (p1, p2, p3):
        print(f"        {p}")
    print("[00b] Reminder: read-only over configs; no matrices were touched.")


if __name__ == "__main__":
    main()
