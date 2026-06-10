"""
10_extract_thermal_field_features.py

Extract per-experiment thermal-field features from ROI temperature matrices
for the ML quality-assessment main line.

Input:
  - <thermal_field_features.input_dir>/*.npy  (default data/processed/roi)
    Each .npy is one experiment's ROI matrix, shape N x H x W (float32 Celsius).
    The file stem is the experiment_id.

Output:
  - <thermal_field_features.output_table>     (thermal_field_features.csv)
    ONE row per experiment (per-experiment aggregation avoids data leakage on
    small, frame-correlated datasets).

SIM_*.npy inputs are tagged as simulated and a warning is printed — they only
validate the code chain, not experimental conclusions.

Usage:
    python scripts/10_extract_thermal_field_features.py --config configs/default.yaml
"""

import os
import sys
import csv
import glob
import argparse

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from utils.config import load_config
from features.thermal_field_features import extract_experiment_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    tfcfg = config.get("thermal_field_features", {}) or {}

    in_dir = tfcfg.get("input_dir", config["paths"]["processed_roi"])
    out_table = tfcfg.get(
        "output_table",
        os.path.join(config["paths"]["results_tables"], "thermal_field_features.csv"))
    fps = float(tfcfg.get("frame_rate_fps", 1000))

    npy_files = sorted(glob.glob(os.path.join(in_dir, "*.npy")))
    if not npy_files:
        print(f"[10] No ROI matrices (*.npy) found in {in_dir}.")
        print("[10] Run scripts 02->03 on real exported data first. "
              "Nothing to extract.")
        return

    sim_files = [p for p in npy_files
                 if os.path.basename(p).startswith("SIM_")]
    simulated = len(sim_files) > 0 and len(sim_files) == len(npy_files)
    if sim_files:
        print(f"[10] WARNING: {len(sim_files)}/{len(npy_files)} ROI file(s) are "
              f"SIMULATED (SIM_*.npy). Features validate the CODE CHAIN ONLY and "
              f"do NOT represent experimental conclusions.")

    print(f"[10] input: {len(npy_files)} ROI matrix file(s) from {in_dir}")

    rows, n_failed = [], 0
    for path in npy_files:
        exp_id = os.path.splitext(os.path.basename(path))[0]
        try:
            roi = np.load(path).astype(np.float32)
            feats = extract_experiment_features(roi, frame_rate_fps=fps,
                                                config=config)
        except Exception as e:                       # noqa: BLE001 (report+continue)
            n_failed += 1
            print(f"      ERROR {os.path.basename(path)}: "
                  f"{type(e).__name__}: {e}")
            continue
        row = {"experiment_id": exp_id,
               "simulated": exp_id.startswith("SIM_")}
        row.update(feats)
        rows.append(row)
        print(f"      {exp_id:20s} N={roi.shape[0]:4d} HxW={roi.shape[1]}x{roi.shape[2]} "
              f"Tmax={feats['peak_temperature']:.1f}C")

    if not rows:
        print("[10] No features extracted (all inputs failed). Nothing written.")
        sys.exit(1)

    # Write one row per experiment.
    lead = ["experiment_id", "simulated"]
    rest = sorted({k for r in rows for k in r} - set(lead))
    fieldnames = lead + rest
    os.makedirs(os.path.dirname(out_table) or ".", exist_ok=True)
    with open(out_table, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"[10] thermal-field features -> {out_table} "
          f"({len(rows)} experiments, one row each)")
    if n_failed:
        print(f"[10] WARNING: {n_failed} file(s) failed (see ERROR lines). "
              f"Nothing was deleted.")
    if simulated:
        print("[10] NOTE: table is SIMULATED — code-chain validation only, "
              "NOT experimental results.")


if __name__ == "__main__":
    main()
