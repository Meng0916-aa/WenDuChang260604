"""
07_evaluate_model.py

Evaluate the trained LSTM baseline on the held-out test split. Metrics are
computed in CELSIUS (predictions are inverse-normalized using the normalizer
saved during training).

Input:
  - <paths.processed_samples>/<dataset.samples_file>   (.npz, test split)
  - results/checkpoints/best_lstm.pt
  - results/checkpoints/normalizer.npz                 (if normalization used)

Output:
  - results/tables/lstm_metrics.csv                    (overall)
  - results/tables/lstm_predictions.csv                (per sample/step)
  - results/tables/lstm_metrics_by_magnetic_group.csv  (if grouping available)

Usage:
    python scripts/07_evaluate_model.py --config configs/default.yaml
"""

import os
import sys
import argparse
from importlib import import_module

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from utils.config import load_config
from training.train import select_device
from training.evaluate import (
    load_checkpoint, load_normalizer, evaluate_with_predictions,
    save_metrics_csv, save_predictions_csv, save_grouped_metrics_csv,
)

# Reuse the lightweight dataset wrapper from the training script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ArrayWindowDataset = import_module("06_train_model").ArrayWindowDataset


def _has_real_groups(mag_groups) -> bool:
    """True if at least one test sample has a known (non-'unknown') group."""
    return bool(len(mag_groups)) and any(g != "unknown" for g in mag_groups)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config["paths"]
    ecfg = config.get("evaluation", {})

    bundle = os.path.join(paths["processed_samples"],
                          config["dataset"].get("samples_file", "window_samples.npz"))
    if not os.path.exists(bundle):
        raise FileNotFoundError(
            f"Samples bundle not found: {bundle}. Run 05 first.")
    ckpt_path = os.path.join(paths["results_checkpoints"], "best_lstm.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. Run 06 first.")

    data = np.load(bundle, allow_pickle=True)
    simulated = bool(data["simulated"])
    test_ds = ArrayWindowDataset(data["X_test"], data["y_test"])
    exp_test = data["exp_test"] if "exp_test" in data else None
    mag_test = data["mag_group_test"] if "mag_group_test" in data else np.array([])

    device = select_device(config)
    model, ckpt = load_checkpoint(ckpt_path, device=device)

    norm_path = os.path.join(paths["results_checkpoints"], "normalizer.npz")
    normalizer = load_normalizer(norm_path) if ckpt.get("normalized", False) else None

    result = evaluate_with_predictions(model, test_ds, config, normalizer)
    metrics, y_true, y_pred = result["metrics"], result["y_true"], result["y_pred"]

    extra = {"model": "lstm", "split": "test",
             "best_epoch": ckpt.get("epoch", -1), "simulated": simulated}

    # 1. Overall metrics (Celsius)
    metrics_path = os.path.join(paths["results_tables"],
                                ecfg.get("metrics_file", "lstm_metrics.csv"))
    save_metrics_csv(metrics, metrics_path, extra=extra)

    # 2. Per-(sample, step) predictions
    pred_path = os.path.join(paths["results_tables"],
                             ecfg.get("predictions_file", "lstm_predictions.csv"))
    save_predictions_csv(y_true, y_pred, pred_path, exp_ids=exp_test)

    # 3. Per-experiment grouped metrics (optional)
    if ecfg.get("group_by_experiment", True) and exp_test is not None:
        per_exp_path = os.path.join(paths["results_tables"],
                                    "lstm_metrics_by_experiment.csv")
        save_grouped_metrics_csv(y_true, y_pred, exp_test, per_exp_path, extra=extra)

    # 4. Magnetic-group metrics (only if real grouping metadata exists)
    group_path = os.path.join(
        paths["results_tables"],
        ecfg.get("metrics_by_group_file", "lstm_metrics_by_magnetic_group.csv"))
    if _has_real_groups(mag_test):
        save_grouped_metrics_csv(y_true, y_pred, mag_test, group_path, extra=extra)
        group_msg = f"by-group metrics -> {group_path}"
    else:
        group_msg = ("magnetic group metadata not available "
                     "(fill magnetic_field_groups.experiment_ids in the config)")

    # ---- console summary (concise) ----
    print("[07] LSTM baseline test metrics (Celsius):")
    for k, v in metrics.items():
        print(f"      {k:20s} = {v:.6f}")
    print(f"[07] metrics      -> {metrics_path}")
    print(f"[07] predictions  -> {pred_path}")
    print(f"[07] {group_msg}")
    if simulated:
        print("[07] NOTE: metrics computed on SIMULATED data — they do NOT "
              "represent experimental results.")


if __name__ == "__main__":
    main()
