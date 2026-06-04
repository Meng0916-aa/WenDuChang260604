"""
09_analyze_temporal_features.py

Temporal feature analysis of thermal-cycle curves for the Chapter-3 results
analysis. Reads thermal-cycle CSVs (output of script 04), extracts temporal
features, and writes a feature table plus figures.

Input:
  - <temporal_analysis.input_dir>/*.csv   (default data/processed/thermal_cycle)

Output:
  - <temporal_analysis.output_table>                  (temporal_features.csv)
  - <output_figure_dir>/temporal_feature_overview.*   (png + pdf)
  - <output_figure_dir>/temporal_curve_<id>.*         (a few typical curves)

If the data comes from SIM_*.csv, all outputs are tagged SIMULATED and a
warning is printed — simulated data validates the code chain only.

Usage:
    python scripts/09_analyze_temporal_features.py --config configs/default.yaml
"""

import os
import sys
import glob
import argparse

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from utils.config import load_config
from features.temporal_features import (
    compute_time_axis, smooth_curve, compute_derivative,
    batch_extract_temporal_features,
)
from visualization.plot_temporal_features import (
    plot_curve_with_derivative, plot_temporal_feature_overview,
)


def _plot_examples(csv_files, tcfg, vcfg, fig_dir, max_examples):
    """Plot curve + dT/dt for the first `max_examples` thermal-cycle CSVs."""
    fps = float(tcfg.get("frame_rate_fps", 1000))
    smooth_window = int(tcfg.get("smooth_window", 1))
    dpi = int(vcfg.get("dpi", 200))
    save_png = bool(vcfg.get("save_png", True))
    save_pdf = bool(vcfg.get("save_pdf", True))

    written = []
    for path in csv_files[:max_examples]:
        arr = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float32)
        names = arr.dtype.names or ()
        if "tmax" not in names:
            continue
        curve = np.atleast_1d(np.asarray(arr["tmax"], dtype=np.float32))
        if curve.size < 2:
            continue
        t = compute_time_axis(curve.size, fps)
        deriv = compute_derivative(smooth_curve(curve, smooth_window), t)

        exp_id = os.path.splitext(os.path.basename(path))[0]
        simulated = exp_id.startswith("SIM_")
        tag = " (SIMULATED)" if simulated else ""
        suffix = "_SIMULATED" if simulated else ""
        out_stub = os.path.join(fig_dir, f"temporal_curve_{exp_id}{suffix}")
        written += plot_curve_with_derivative(
            t, curve, deriv, out_stub,
            title=f"Thermal Cycle and dT/dt - {exp_id} (tmax){tag}",
            dpi=dpi, save_png=save_png, save_pdf=save_pdf)
    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    tcfg = config.get("temporal_analysis", {}) or {}
    vcfg = config.get("visualization", {}) or {}

    input_dir = tcfg.get("input_dir",
                         config["paths"]["processed_thermal_cycle"])
    output_table = tcfg.get(
        "output_table",
        os.path.join(config["paths"]["results_tables"], "temporal_features.csv"))
    fig_dir = tcfg.get("output_figure_dir", config["paths"]["results_figures"])
    max_examples = int(tcfg.get("max_examples_to_plot", 3))

    # 1. Locate thermal-cycle CSVs
    csv_files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    if not csv_files:
        print(f"[09] No thermal-cycle CSVs found in {input_dir}.")
        print("[09] Run scripts 02->04 on real exported data, or 05 to generate "
              "SIMULATED curves for a code-chain test. Nothing to analyze.")
        return

    sim_files = [p for p in csv_files
                 if os.path.basename(p).startswith("SIM_")]
    if sim_files:
        print(f"[09] WARNING: {len(sim_files)}/{len(csv_files)} input file(s) are "
              f"SIMULATED (SIM_*.csv). Results validate the CODE CHAIN ONLY and "
              f"do NOT represent experimental conclusions.")

    print(f"[09] input: {len(csv_files)} thermal-cycle CSV(s) from {input_dir}")

    # 2. Extract features -> table
    rows = batch_extract_temporal_features(input_dir, output_table, config)
    print(f"[09] features -> {output_table} ({len(rows)} experiments)")

    # 3. Overview figure
    os.makedirs(fig_dir, exist_ok=True)
    overview = plot_temporal_feature_overview(
        output_table, fig_dir,
        dpi=int(vcfg.get("dpi", 200)),
        save_png=bool(vcfg.get("save_png", True)),
        save_pdf=bool(vcfg.get("save_pdf", True)))
    for p in overview:
        print(f"[09] overview  -> {p}")

    # 4. Typical curve + derivative examples
    examples = _plot_examples(csv_files, tcfg, vcfg, fig_dir, max_examples)
    for p in examples:
        print(f"[09] example   -> {p}")

    if sim_files:
        print("[09] NOTE: outputs are tagged SIMULATED — code-chain demonstration "
              "only, NOT experimental results.")


if __name__ == "__main__":
    main()
