"""
15_train_section_quality_model.py

Train and compare traditional ML models for SECTION-LEVEL cladding quality on
the merged section dataset (script 14). The ML sample unit is a cross-section.

Leakage guard (mandatory): sections are GROUPED by experiment_id and split with
GroupKFold / LeaveOneGroupOut — never a random train/test split, and sections of
the same experiment never straddle train and test.

For each input set {process_only, thermal_only, fused}:
  - regression on each section_ml.regression_targets (ridge/svr/random_forest/knn)
  - classification on section_ml.classification_target (logistic_regression/svm/
    random_forest/knn); skipped (with a WARNING) if the target has one class.

Input:
  - results/tables/section_ml_dataset.csv              (from script 14)

Output:
  - results/tables/section_ml_regression_metrics.csv
  - results/tables/section_ml_regression_predictions.csv
  - results/tables/section_ml_classification_metrics.csv
  - results/tables/section_ml_classification_predictions.csv
  - results/tables/section_ml_feature_importance.csv

Usage:
    python scripts/15_train_section_quality_model.py --config configs/default.yaml
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
from models.section_quality_ml import run_section_models, has_multiple_classes


def select_input_features(df: pd.DataFrame, wanted: list):
    """Return (present, missing) feature columns from a requested list."""
    present = [c for c in wanted if c in df.columns]
    missing = [c for c in wanted if c not in df.columns]
    return present, missing


def _write_rows(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    mlcfg = config.get("section_ml", {}) or {}
    tables = config["paths"]["results_tables"]

    dataset_path = mlcfg.get("dataset_csv",
                             os.path.join(tables, "section_ml_dataset.csv"))
    if not os.path.exists(dataset_path):
        print(f"[15] Section dataset not found: {dataset_path}.")
        print("[15] Run scripts 13 (local features) and 14 (merge labels) first.")
        return

    df = pd.read_csv(dataset_path)
    group_col = mlcfg.get("group_column", "experiment_id")
    id_col = mlcfg.get("sample_id_column", "sample_id")
    cv = mlcfg.get("cv", "group_kfold")
    n_splits = int(mlcfg.get("n_splits", 5))
    random_state = int(mlcfg.get("random_state", 42))
    input_sets = mlcfg.get("input_sets", {}) or {}
    reg_targets = mlcfg.get("regression_targets", []) or []
    cls_target = mlcfg.get("classification_target", "quality_label")
    reg_models = mlcfg.get("regression_models",
                           ["ridge", "svr", "random_forest", "knn"])
    cls_models = mlcfg.get("classification_models",
                           ["logistic_regression", "svm", "random_forest", "knn"])

    if group_col not in df.columns:
        raise SystemExit(f"[15] group column '{group_col}' not in dataset.")
    n_groups_total = df[group_col].astype(str).nunique()
    if n_groups_total < 2:
        raise SystemExit(
            f"[15] only {n_groups_total} experiment group(s) in '{group_col}'. "
            f"Section-level GroupKFold needs >= 2. Add more experiments.")

    print(f"[15] dataset: {len(df)} section(s), {n_groups_total} experiment "
          f"group(s) | cv={cv} n_splits={n_splits}")

    reg_metrics_rows, reg_pred_rows = [], []
    cls_metrics_rows, cls_pred_rows = [], []
    importance_rows = []

    for set_name, wanted in input_sets.items():
        present, missing = select_input_features(df, wanted)
        if missing:
            print(f"[15] WARNING: input set '{set_name}' missing column(s): "
                  f"{missing} (using {len(present)} available).")
        if not present:
            print(f"[15] input set '{set_name}': no usable features, skipped.")
            continue

        # ---------------- Regression ----------------
        for target in reg_targets:
            if target not in df.columns:
                print(f"[15] regression target '{target}' not in dataset, skipped.")
                continue
            work = df.dropna(subset=present + [target, group_col])
            if len(work) < 2 or work[group_col].astype(str).nunique() < 2:
                print(f"[15] [{set_name}/{target}] insufficient rows/groups after "
                      f"NaN drop ({len(work)} rows), skipped.")
                continue
            X = work[present].to_numpy(dtype=np.float64)
            y = work[target].to_numpy(dtype=np.float64)
            groups = work[group_col].astype(str).to_numpy()
            ids = work.get(id_col, pd.Series(range(len(work)))).astype(str).tolist()

            res = run_section_models(X, y, groups, present, "regression",
                                     reg_models, cv=cv, n_splits=n_splits,
                                     random_state=random_state)
            for m, mt in res["metrics"].items():
                reg_metrics_rows.append({
                    "input_set": set_name, "task": "regression", "target": target,
                    "model": m, "mae": mt["mae"], "rmse": mt["rmse"], "r2": mt["r2"],
                    "n_samples": len(work), "n_groups": res["n_groups"],
                    "cv_method": res["cv_method"]})
            for m, preds in res["predictions"].items():
                for i in range(len(work)):
                    reg_pred_rows.append({
                        "input_set": set_name, "target": target, "model": m,
                        "sample_id": ids[i], "experiment_id": groups[i],
                        "y_true": float(y[i]), "y_pred": float(preds[i])})
            for d in res["feature_importance"]:
                importance_rows.append({
                    "input_set": set_name, "task": "regression", "target": target,
                    "model": "random_forest", "feature": d["feature"],
                    "importance": d["importance"]})
            print(f"[15] [{set_name}/reg/{target}] "
                  + " ".join(f"{m}:R2={mt['r2']:.3f}"
                             for m, mt in res["metrics"].items()))

        # ---------------- Classification ----------------
        if cls_target in df.columns:
            work = df.dropna(subset=present + [cls_target, group_col])
            if len(work) >= 2 and work[group_col].astype(str).nunique() >= 2:
                y = work[cls_target].astype(str).to_numpy()
                if not has_multiple_classes(y):
                    print(f"[15] [{set_name}/cls] WARNING: target '{cls_target}' "
                          f"has a single class {np.unique(y).tolist()} — "
                          f"classification skipped.")
                else:
                    X = work[present].to_numpy(dtype=np.float64)
                    groups = work[group_col].astype(str).to_numpy()
                    ids = work.get(id_col, pd.Series(range(len(work)))).astype(str).tolist()
                    res = run_section_models(X, y, groups, present, "classification",
                                             cls_models, cv=cv, n_splits=n_splits,
                                             random_state=random_state)
                    for m, mt in res["metrics"].items():
                        cls_metrics_rows.append({
                            "input_set": set_name, "task": "classification",
                            "target": cls_target, "model": m,
                            "accuracy": mt["accuracy"], "precision": mt["precision"],
                            "recall": mt["recall"], "f1": mt["f1"],
                            "n_samples": len(work), "n_groups": res["n_groups"],
                            "cv_method": res["cv_method"]})
                    for m, preds in res["predictions"].items():
                        for i in range(len(work)):
                            cls_pred_rows.append({
                                "input_set": set_name, "target": cls_target,
                                "model": m, "sample_id": ids[i],
                                "experiment_id": groups[i],
                                "y_true": str(y[i]), "y_pred": str(preds[i])})
                    for d in res["feature_importance"]:
                        importance_rows.append({
                            "input_set": set_name, "task": "classification",
                            "target": cls_target, "model": "random_forest",
                            "feature": d["feature"], "importance": d["importance"]})
                    print(f"[15] [{set_name}/cls/{cls_target}] "
                          + " ".join(f"{m}:f1={mt['f1']:.3f}"
                                     for m, mt in res["metrics"].items()))
            else:
                print(f"[15] [{set_name}/cls] insufficient rows/groups, skipped.")

    # ---------------- Write tables ----------------
    _write_rows(mlcfg.get("output_regression_metrics",
                          os.path.join(tables, "section_ml_regression_metrics.csv")),
                ["input_set", "task", "target", "model", "mae", "rmse", "r2",
                 "n_samples", "n_groups", "cv_method"], reg_metrics_rows)
    _write_rows(mlcfg.get("output_regression_predictions",
                          os.path.join(tables, "section_ml_regression_predictions.csv")),
                ["input_set", "target", "model", "sample_id", "experiment_id",
                 "y_true", "y_pred"], reg_pred_rows)
    _write_rows(mlcfg.get("output_classification_metrics",
                          os.path.join(tables, "section_ml_classification_metrics.csv")),
                ["input_set", "task", "target", "model", "accuracy", "precision",
                 "recall", "f1", "n_samples", "n_groups", "cv_method"],
                cls_metrics_rows)
    _write_rows(mlcfg.get("output_classification_predictions",
                          os.path.join(tables, "section_ml_classification_predictions.csv")),
                ["input_set", "target", "model", "sample_id", "experiment_id",
                 "y_true", "y_pred"], cls_pred_rows)
    _write_rows(mlcfg.get("output_feature_importance",
                          os.path.join(tables, "section_ml_feature_importance.csv")),
                ["input_set", "task", "target", "model", "feature", "importance"],
                importance_rows)

    print("-" * 60)
    print(f"[15] tables -> {os.path.dirname(mlcfg.get('output_regression_metrics', os.path.join(tables, 'x'))) or tables}")
    print("[15] input-set summary (lower RMSE / higher F1 is better):")
    for set_name in input_sets:
        regs = [r for r in reg_metrics_rows if r["input_set"] == set_name]
        clss = [r for r in cls_metrics_rows if r["input_set"] == set_name]
        if regs:
            best = min(regs, key=lambda r: r["rmse"])
            print(f"      {set_name:14s} best reg: {best['target']}/{best['model']} "
                  f"RMSE={best['rmse']:.3f} R2={best['r2']:.3f}")
        if clss:
            best = max(clss, key=lambda r: r["f1"])
            print(f"      {set_name:14s} best cls: {best['model']} "
                  f"f1={best['f1']:.3f} acc={best['accuracy']:.3f}")
    print("[15] Leakage guard: all splits grouped by experiment_id "
          "(GroupKFold / LeaveOneGroupOut), no random shuffling of sections.")


if __name__ == "__main__":
    main()
