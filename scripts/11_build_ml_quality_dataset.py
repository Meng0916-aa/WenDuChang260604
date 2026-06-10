"""
11_build_ml_quality_dataset.py

Merge per-experiment thermal-field features with cross-section quality labels
into a single ML-ready table.

Input:
  - results/tables/thermal_field_features.csv         (from script 10)
  - quality label file (LOCAL, git-ignored), one of:
      data/metadata/quality_labels.csv   (preferred)
      data/metadata/quality_labels.xlsx  (needs openpyxl)
    Join key: feature 'experiment_id' == label 'sample_id'.

Output:
  - results/tables/ml_quality_dataset.csv             (features + labels)

Safety:
  - If the label file is missing, prints clear guidance and exits.
  - Reports any sample_ids present in features-but-not-labels and vice versa.
  - Does NOT silently drop unmatched samples: it exits non-zero if any mismatch
    exists, so you must reconcile the lists first.

Usage:
    python scripts/11_build_ml_quality_dataset.py --config configs/default.yaml
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from utils.config import load_config


def _load_labels(qcfg: dict):
    """Load the quality-label table from CSV (preferred) or Excel."""
    csv_path = qcfg.get("label_file_csv")
    xlsx_path = qcfg.get("label_file_excel")

    if csv_path and os.path.exists(csv_path):
        return pd.read_csv(csv_path), csv_path
    if xlsx_path and os.path.exists(xlsx_path):
        try:
            return pd.read_excel(xlsx_path), xlsx_path
        except ImportError as e:
            raise SystemExit(
                f"[11] Excel label file found ({xlsx_path}) but reading it needs "
                f"'openpyxl', which is not installed. Either install openpyxl in "
                f"the conda env, or provide a CSV label file instead "
                f"({csv_path}). Underlying error: {e}")
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    tfcfg = config.get("thermal_field_features", {}) or {}
    qcfg = config.get("quality_labels", {}) or {}
    mlcfg = config.get("ml_quality", {}) or {}

    feat_path = tfcfg.get(
        "output_table",
        os.path.join(config["paths"]["results_tables"], "thermal_field_features.csv"))
    out_path = mlcfg.get(
        "dataset_table",
        os.path.join(config["paths"]["results_tables"], "ml_quality_dataset.csv"))

    # 1. Features
    if not os.path.exists(feat_path):
        print(f"[11] Feature table not found: {feat_path}. Run script 10 first.")
        return
    features = pd.read_csv(feat_path)
    if "experiment_id" not in features.columns:
        raise SystemExit("[11] feature table has no 'experiment_id' column.")

    # 2. Labels (local file)
    labels, label_src = _load_labels(qcfg)
    if labels is None:
        print("[11] No quality-label file found. Expected one of:")
        print(f"        {qcfg.get('label_file_csv')}")
        print(f"        {qcfg.get('label_file_excel')}")
        print("[11] Create it from docs/quality_label_template.md (key column: "
              "sample_id). These files stay LOCAL and are not uploaded.")
        return
    if "sample_id" not in labels.columns:
        raise SystemExit(
            f"[11] label file {label_src} has no 'sample_id' column "
            f"(see docs/quality_label_template.md).")

    print(f"[11] features: {len(features)} rows from {feat_path}")
    print(f"[11] labels:   {len(labels)} rows from {label_src}")

    # 3. Reconcile sample ids (NO silent dropping)
    feat_ids = set(features["experiment_id"].astype(str))
    label_ids = set(labels["sample_id"].astype(str))
    missing_labels = sorted(feat_ids - label_ids)   # features without a label
    missing_feats = sorted(label_ids - feat_ids)    # labels without features

    if missing_labels:
        print(f"[11] WARNING: {len(missing_labels)} experiment(s) have features "
              f"but NO label: {missing_labels}")
    if missing_feats:
        print(f"[11] WARNING: {len(missing_feats)} label(s) have NO matching "
              f"features: {missing_feats}")
    if missing_labels or missing_feats:
        print("[11] Refusing to silently drop samples. Reconcile the two lists "
              "(fix sample_id naming, add missing labels, or re-run script 10) "
              "and try again.")
        sys.exit(1)

    # 4. Merge (inner join on the reconciled, fully-matching ids)
    merged = features.merge(labels, left_on="experiment_id",
                            right_on="sample_id", how="inner")
    n = len(merged)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    merged.to_csv(out_path, index=False)

    simulated = bool(merged.get("simulated", pd.Series([False] * n)).astype(str)
                     .str.lower().isin(["true", "1", "yes"]).any())

    print(f"[11] merged dataset -> {out_path} ({n} samples, "
          f"{merged.shape[1]} columns)")
    # Note available targets for script 12.
    cls_target = mlcfg.get("target", "quality_label")
    reg_targets = [c for c in ("dilution_rate", "aspect_ratio", "wetting_angle_avg")
                   if c in merged.columns]
    print(f"[11] classification target available: "
          f"{cls_target in merged.columns} ('{cls_target}')")
    print(f"[11] regression targets available: {reg_targets}")
    if n < 5:
        print(f"[11] WARNING: only {n} sample(s) — exploratory only, "
              f"do NOT over-interpret downstream model results.")
    if simulated:
        print("[11] NOTE: dataset includes SIMULATED rows — code-chain "
              "validation only, NOT experimental results.")


if __name__ == "__main__":
    main()
