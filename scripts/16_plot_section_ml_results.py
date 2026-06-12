"""
16_plot_section_ml_results.py

Plot the section-level ML results produced by script 15. Reads the metrics /
predictions / feature-importance tables and writes figures under
results/figures/section_ml/.

Figures:
  - regression_pred_vs_true_<target>_<model>_<input_set>.png
  - classification_confusion_matrix_<model>_<input_set>.png
  - feature_importance_<target>_<model>_<input_set>.png   (random_forest)
  - input_set_performance_comparison.png

matplotlib only (no seaborn). If a result table is missing, a note is printed
and that group of figures is skipped — the script never crashes. No images are
committed to git.

Usage:
    python scripts/16_plot_section_ml_results.py --config configs/default.yaml
"""

import os
import sys
import argparse

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from utils.config import load_config
from visualization.plot_section_ml_results import (
    plot_regression_prediction, plot_confusion_matrix, plot_feature_importance,
    plot_input_set_comparison, confusion_matrix_from_predictions,
)


def _read(path, label):
    if not path or not os.path.exists(path):
        print(f"[16] NOTE: {label} not found ({path}); skipping those figures.")
        return None
    df = pd.read_csv(path)
    if df.empty:
        print(f"[16] NOTE: {label} is empty ({path}); skipping those figures.")
        return None
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    mlcfg = config.get("section_ml", {}) or {}
    tables = config["paths"]["results_tables"]
    vcfg = config.get("visualization", {}) or {}
    dpi = int(vcfg.get("dpi", 200))

    fig_dir = mlcfg.get("output_figures_dir",
                        os.path.join(config["paths"]["results_figures"], "section_ml"))

    reg_pred = _read(mlcfg.get("output_regression_predictions",
                               os.path.join(tables, "section_ml_regression_predictions.csv")),
                     "regression predictions")
    cls_pred = _read(mlcfg.get("output_classification_predictions",
                               os.path.join(tables, "section_ml_classification_predictions.csv")),
                     "classification predictions")
    importance = _read(mlcfg.get("output_feature_importance",
                                 os.path.join(tables, "section_ml_feature_importance.csv")),
                       "feature importance")
    reg_metrics = _read(mlcfg.get("output_regression_metrics",
                                  os.path.join(tables, "section_ml_regression_metrics.csv")),
                        "regression metrics")
    cls_metrics = _read(mlcfg.get("output_classification_metrics",
                                  os.path.join(tables, "section_ml_classification_metrics.csv")),
                        "classification metrics")

    written = []

    # 1. Regression pred-vs-true scatter per (target, model, input_set).
    if reg_pred is not None:
        for (iset, target, model), g in reg_pred.groupby(
                ["input_set", "target", "model"]):
            name = f"regression_pred_vs_true_{target}_{model}_{iset}"
            written += plot_regression_prediction(
                g["y_true"].to_numpy(float), g["y_pred"].to_numpy(float),
                fig_dir, name=name,
                title=f"{target} — {model} ({iset})", dpi=dpi)

    # 2. Classification confusion matrix per (model, input_set).
    if cls_pred is not None:
        for (iset, model), g in cls_pred.groupby(["input_set", "model"]):
            cm, labels = confusion_matrix_from_predictions(
                g["y_true"].astype(str).tolist(), g["y_pred"].astype(str).tolist())
            name = f"classification_confusion_matrix_{model}_{iset}"
            written += plot_confusion_matrix(
                cm, labels, fig_dir, name=name,
                title=f"Confusion matrix — {model} ({iset})", dpi=dpi)

    # 3. Random-forest feature importance per (target, model, input_set).
    if importance is not None:
        for (iset, target, model), g in importance.groupby(
                ["input_set", "target", "model"]):
            imp = [{"feature": f, "importance": v}
                   for f, v in zip(g["feature"], g["importance"])]
            name = f"feature_importance_{target}_{model}_{iset}"
            written += plot_feature_importance(
                imp, fig_dir, name=name,
                title=f"RF importance — {target} ({iset})", dpi=dpi)

    # 4. Input-set performance comparison (classification F1 if available, else
    #    regression R^2).
    if cls_metrics is not None:
        rows = cls_metrics.to_dict("records")
        written += plot_input_set_comparison(
            rows, fig_dir, metric_key="f1", metric_label="Macro F1",
            title="Input-set comparison (classification F1)", dpi=dpi)
    elif reg_metrics is not None:
        rows = reg_metrics.to_dict("records")
        written += plot_input_set_comparison(
            rows, fig_dir, metric_key="r2", metric_label="R^2",
            title="Input-set comparison (regression R^2)", dpi=dpi)

    if not written:
        print("[16] Nothing plotted. Run scripts 13->14->15 first to produce "
              "results/tables/section_ml_*.csv.")
        return
    print(f"[16] wrote {len(written)} figure file(s) to {fig_dir}:")
    for p in written:
        print(f"      {p}")


if __name__ == "__main__":
    main()
