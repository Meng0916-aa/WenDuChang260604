"""
12_train_ml_quality_model.py

Train and compare traditional ML models for cladding quality assessment on the
merged feature+label table. Small-sample friendly (LeaveOneOut by default).

Input:
  - results/tables/ml_quality_dataset.csv             (from script 11)

Output:
  - results/tables/ml_quality_metrics.csv
  - results/tables/ml_quality_predictions.csv
  - results/tables/ml_feature_importance.csv
  - results/figures/ml_quality_confusion_matrix.png/.pdf   (classification)
  - results/figures/ml_feature_importance.png/.pdf
  - results/figures/ml_regression_prediction.png/.pdf      (regression)

Config: ml_quality.{task,target,cv,models,...}. Default task = classification
on 'quality_label'; switch to regression via config.

Usage:
    python scripts/12_train_ml_quality_model.py --config configs/default.yaml
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
from models.ml_quality import run_models
from visualization.plot_ml_quality import (
    plot_confusion_matrix, plot_feature_importance, plot_regression_prediction,
)

# Columns that are NOT thermal-field features (ids, meta, and label columns).
_NON_FEATURE_COLS = {
    "experiment_id", "sample_id", "simulated", "notes",
    "file_name", "magnetic_group",
    # cross-section label columns
    "cladding_height_H", "cladding_width_W", "molten_depth_D",
    "wetting_angle_left", "wetting_angle_right",
    "dilution_rate", "aspect_ratio", "wetting_angle_avg",
    "quality_label",
}


def _select_features(df: pd.DataFrame, target: str) -> list:
    """Numeric columns that are thermal-field features (exclude ids/labels)."""
    cols = []
    for c in df.columns:
        if c in _NON_FEATURE_COLS or c == target:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def _write_metrics_csv(path, task, metrics, n_samples, simulated):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        if task == "classification":
            cols = ["model", "accuracy", "precision", "recall", "f1",
                    "n_samples", "simulated"]
        else:
            cols = ["model", "rmse", "mae", "r2", "n_samples", "simulated"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for name, m in metrics.items():
            row = {"model": name, "n_samples": n_samples, "simulated": simulated}
            row.update({k: m[k] for k in cols if k in m})
            w.writerow(row)


def _write_predictions_csv(path, ids, y_true, predictions, simulated):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    model_names = list(predictions.keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        cols = ["sample_id", "y_true"] + [f"pred_{m}" for m in model_names] + ["simulated"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i in range(len(y_true)):
            row = {"sample_id": ids[i], "y_true": y_true[i], "simulated": simulated}
            for m in model_names:
                row[f"pred_{m}"] = predictions[m][i]
            w.writerow(row)


def _write_importance_csv(path, importance, simulated):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["feature", "importance", "simulated"])
        w.writeheader()
        for d in importance:
            w.writerow({"feature": d["feature"], "importance": d["importance"],
                        "simulated": simulated})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    mlcfg = config.get("ml_quality", {}) or {}
    tables = config["paths"]["results_tables"]
    fig_dir = config["paths"]["results_figures"]
    vcfg = config.get("visualization", {}) or {}

    dataset_path = mlcfg.get("dataset_table",
                             os.path.join(tables, "ml_quality_dataset.csv"))
    if not os.path.exists(dataset_path):
        print(f"[12] Dataset not found: {dataset_path}. Run script 11 first.")
        return

    df = pd.read_csv(dataset_path)
    task = str(mlcfg.get("task", "classification")).lower()
    target = mlcfg.get("target", "quality_label")
    if task == "regression":
        # For regression, the target defaults to a geometric ratio if the
        # configured classification target isn't numeric.
        if target == "quality_label" or target not in df.columns:
            target = "dilution_rate"
    cv = mlcfg.get("cv", "leave_one_out")
    model_names = mlcfg.get("models", ["rf", "svm", "knn", "logistic_regression"])
    random_state = int(mlcfg.get("random_state", 42))
    n_splits = int(mlcfg.get("n_splits", 5))

    if target not in df.columns:
        raise SystemExit(
            f"[12] target '{target}' not in dataset columns. Available: "
            f"{list(df.columns)}")

    feature_cols = _select_features(df, target)
    if not feature_cols:
        raise SystemExit("[12] no numeric feature columns found in dataset.")

    # Drop rows with missing target / features (and report, don't silently lose).
    work = df.dropna(subset=[target] + feature_cols)
    dropped = len(df) - len(work)
    if dropped:
        print(f"[12] WARNING: dropped {dropped} row(s) with missing "
              f"target/feature values.")

    X = work[feature_cols].to_numpy(dtype=np.float64)
    y_raw = work[target].to_numpy()
    ids = work.get("experiment_id", work.get("sample_id",
                   pd.Series(range(len(work))))).astype(str).tolist()
    n = len(work)
    simulated = bool(work.get("simulated", pd.Series([False] * n)).astype(str)
                     .str.lower().isin(["true", "1", "yes"]).any())

    print(f"[12] dataset: {n} samples x {len(feature_cols)} features "
          f"| task={task} target='{target}' cv={cv}")
    if n < 5:
        print(f"[12] WARNING: only {n} sample(s). Results are EXPLORATORY ONLY; "
              f"do NOT claim generalization from so few samples.")
    if simulated:
        print("[12] WARNING: SIMULATED data — code-chain validation only, "
              "NOT experimental conclusions.")

    if task == "classification":
        # Guard: need at least 2 classes.
        if len(np.unique(y_raw)) < 2:
            raise SystemExit(
                f"[12] classification needs >=2 classes in '{target}', got "
                f"{np.unique(y_raw).tolist()}.")

    results = run_models(X, y_raw, feature_cols, task, model_names, cv,
                         n_splits=n_splits, random_state=random_state)
    tag = "_SIMULATED" if simulated else ""

    # --- Tables ---
    _write_metrics_csv(mlcfg.get("output_metrics",
                                 os.path.join(tables, "ml_quality_metrics.csv")),
                       task, results["metrics"], n, simulated)
    _write_predictions_csv(mlcfg.get("output_predictions",
                                     os.path.join(tables, "ml_quality_predictions.csv")),
                           ids, y_raw.tolist(), results["predictions"], simulated)
    _write_importance_csv(mlcfg.get("output_feature_importance",
                                    os.path.join(tables, "ml_feature_importance.csv")),
                          results["feature_importance"], simulated)

    # --- Console summary ---
    print("[12] model comparison:")
    for name, m in results["metrics"].items():
        if task == "classification":
            print(f"      {name:20s} acc={m['accuracy']:.3f} f1={m['f1']:.3f}")
        else:
            print(f"      {name:20s} rmse={m['rmse']:.4f} r2={m['r2']:.3f}")

    # --- Figures ---
    dpi = int(vcfg.get("dpi", 200))
    written = []
    importance = results["feature_importance"]
    title_tag = " (SIMULATED)" if simulated else ""

    written += plot_feature_importance(
        importance, fig_dir, name=f"ml_feature_importance{tag}",
        title=f"Random Forest Feature Importance{title_tag}", dpi=dpi)

    if task == "classification":
        # Use RF's confusion matrix for the figure (first available model).
        cm_model = "rf" if "rf" in results["metrics"] else list(results["metrics"])[0]
        cm = results["metrics"][cm_model]["confusion_matrix"]
        labels = results["metrics"][cm_model]["labels"]
        written += plot_confusion_matrix(
            cm, labels, fig_dir, name=f"ml_quality_confusion_matrix{tag}",
            title=f"Confusion Matrix ({cm_model}){title_tag}", dpi=dpi)
    else:
        reg_model = "rf" if "rf" in results["predictions"] else list(results["predictions"])[0]
        written += plot_regression_prediction(
            y_raw.astype(float), results["predictions"][reg_model], fig_dir,
            name=f"ml_regression_prediction{tag}",
            title=f"Regression true vs predicted ({reg_model}){title_tag}", dpi=dpi)

    print(f"[12] tables  -> {tables}")
    for p in written:
        print(f"[12] figure  -> {p}")
    if simulated:
        print("[12] NOTE: all outputs tagged SIMULATED — code-chain demonstration "
              "only, NOT experimental results.")


if __name__ == "__main__":
    main()
