"""
04_extract_thermal_cycle.py

Read ROI temperature matrices (N x H x W float32 Celsius) from
data/processed/roi, extract the three thermal-cycle curves
(tmax, center_average, hot_zone_average), and save one CSV per experiment
to data/processed/thermal_cycle.

Output CSV columns: frame, tmax, center_average, hot_zone_average

Usage:
    python scripts/04_extract_thermal_cycle.py --config configs/default.yaml
"""

import os
import sys
import glob
import argparse

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from utils.config import load_config
from preprocess.thermal_cycle import extract_all_cycles, save_thermal_cycles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config["paths"]
    tcfg = config["thermal_cycle"]

    in_dir = paths["processed_roi"]
    out_dir = paths["processed_thermal_cycle"]
    os.makedirs(out_dir, exist_ok=True)

    radius = int(tcfg.get("center_average_radius", 5))
    threshold = float(tcfg.get("hot_zone_threshold_celsius", 800.0))

    files = sorted(glob.glob(os.path.join(in_dir, "*.npy")))
    if not files:
        print(f"[04] No ROI matrices found under {in_dir}. Run 03 first.")
        return

    print(f"[04] input: {len(files)} ROI file(s) from {in_dir}")
    print(f"[04] center_radius={radius}px hot_zone_threshold={threshold}C")

    n_done = 0
    for filepath in files:
        data = np.load(filepath).astype(np.float32)
        cycles = extract_all_cycles(data, center_radius=radius, hot_threshold=threshold)
        stem = os.path.splitext(os.path.basename(filepath))[0]
        out_path = os.path.join(out_dir, f"{stem}.csv")
        save_thermal_cycles(cycles, out_path)
        n_done += 1
        print(f"      {os.path.basename(filepath):30s} N={data.shape[0]:4d} "
              f"-> {os.path.basename(out_path)}")

    print(f"[04] wrote {n_done} thermal-cycle CSV(s) -> {out_dir}")


if __name__ == "__main__":
    main()
