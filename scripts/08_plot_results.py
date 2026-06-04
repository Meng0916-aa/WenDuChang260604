"""
08_plot_results.py

Read evaluation outputs and render figures (PNG + PDF) under results/figures.

Input:
  - results/tables/lstm_predictions.csv
  - results/tables/lstm_metrics.csv
  - results/tables/lstm_metrics_by_magnetic_group.csv     (optional)

Output (results/figures/, each saved as .png and .pdf):
  - prediction_curve_sample_<i>[_SIMULATED]
  - metrics_overview[_SIMULATED]
  - metrics_by_magnetic_group[_SIMULATED]                 (if available)

matplotlib only. If the underlying data is SIMULATED, "_SIMULATED" is added
to every figure name and title so plots are never mistaken for experiments.

Usage:
    python scripts/08_plot_results.py --config configs/default.yaml
"""

import os
import sys
import csv
import argparse
from collections import defaultdict

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from utils.config import load_config
from visualization.plot_curves import plot_prediction_curve, save_figure
from visualization.plot_metrics import plot_metric_bars, plot_grouped_metric_bars


def _read_predictions(path):
    """Return {sample_index: (y_true[list], y_pred[list])} and simulated flag-less."""
    samples = defaultdict(lambda: ([], []))
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            idx = int(row["sample_index"])
            samples[idx][0].append(float(row["y_true"]))
            samples[idx][1].append(float(row["y_pred"]))
    return samples


def _read_metrics_row(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else {}


def _read_grouped_metrics(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _is_simulated(row) -> bool:
    return str(row.get("simulated", "")).lower() in ("true", "1", "yes")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config["paths"]
    vcfg = config.get("visualization", {})
    tables = paths["results_tables"]
    fig_dir = paths["results_figures"]
    dpi = int(vcfg.get("dpi", 200))
    save_png = bool(vcfg.get("save_png", True))
    save_pdf = bool(vcfg.get("save_pdf", True))
    figsize = tuple(vcfg.get("figsize", [8, 4]))
    n_curves = int(vcfg.get("num_curve_samples", 3))

    ecfg = config.get("evaluation", {})
    pred_path = os.path.join(tables, ecfg.get("predictions_file", "lstm_predictions.csv"))
    metrics_path = os.path.join(tables, ecfg.get("metrics_file", "lstm_metrics.csv"))
    group_path = os.path.join(
        tables, ecfg.get("metrics_by_group_file", "lstm_metrics_by_magnetic_group.csv"))

    if not os.path.exists(pred_path) or not os.path.exists(metrics_path):
        raise FileNotFoundError(
            f"Missing {pred_path} or {metrics_path}. Run 07 first.")

    metrics_row = _read_metrics_row(metrics_path)
    simulated = _is_simulated(metrics_row)
    tag = " (SIMULATED)" if simulated else ""
    suffix = "_SIMULATED" if simulated else ""

    written = []

    # 1. Prediction curves for the first N samples
    samples = _read_predictions(pred_path)
    for i in sorted(samples.keys())[:n_curves]:
        yt, yp = samples[i]
        fig = plot_prediction_curve(
            np.array(yt), np.array(yp),
            title=f"Thermal Cycle Prediction - sample {i}{tag}",
            figsize=figsize)
        written += save_figure(fig, fig_dir, f"prediction_curve_sample_{i}{suffix}",
                               dpi=dpi, save_png=save_png, save_pdf=save_pdf)

    # 2. Overall metrics bar chart
    metric_keys = ["rmse", "mae", "waveform_similarity"]
    vals = [float(metrics_row.get(k, 0.0)) for k in metric_keys]
    fig = plot_metric_bars(metric_keys, vals, metric_name="LSTM test metrics",
                           title=f"LSTM baseline test metrics{tag}",
                           ylabel="Value", figsize=figsize)
    written += save_figure(fig, fig_dir, f"metrics_overview{suffix}",
                           dpi=dpi, save_png=save_png, save_pdf=save_pdf)

    # 3. Magnetic-group comparison (only if the file exists)
    if os.path.exists(group_path):
        rows = _read_grouped_metrics(group_path)
        if rows:
            group_labels = [r["group"] for r in rows]
            mpg = {r["group"]: {k: float(r.get(k, 0.0)) for k in metric_keys}
                   for r in rows}
            fig = plot_grouped_metric_bars(
                group_labels, mpg, metric_keys=metric_keys,
                title=f"Metrics by magnetic group{tag}", figsize=figsize)
            written += save_figure(fig, fig_dir, f"metrics_by_magnetic_group{suffix}",
                                   dpi=dpi, save_png=save_png, save_pdf=save_pdf)
    else:
        print("[08] magnetic group metrics file not found — skipping group plot.")

    print(f"[08] figures saved to {fig_dir} ({len(written)} files)")
    for p in written:
        print(f"      {p}")
    if simulated:
        print("[08] NOTE: figures are based on SIMULATED data — code-chain "
              "demonstration only, NOT experimental results.")


if __name__ == "__main__":
    main()
