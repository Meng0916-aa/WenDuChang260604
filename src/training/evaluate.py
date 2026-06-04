"""
Evaluation for the LSTM thermal-cycle-prediction baseline.

Loads a trained checkpoint and (if the model was trained with normalization)
the saved StandardNormalizer, runs prediction over a dataset, inverse-
transforms predictions and ground truth back to Celsius, and reports
RMSE / MAE / waveform_similarity computed in Celsius.

Outputs (written by scripts/07_evaluate_model.py):
  - results/tables/lstm_metrics.csv                 (overall)
  - results/tables/lstm_predictions.csv             (sample_index, step, ...)
  - results/tables/lstm_metrics_by_magnetic_group.csv (if grouping available)
"""

import os
import csv

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.lstm import LSTMForecastModel
from utils.metrics import compute_all_metrics, compute_grouped_metrics
from preprocess.normalize import StandardNormalizer
from training.train import select_device, make_target


# ---------------------------------------------------------------------------
# Checkpoint / normalizer loading
# ---------------------------------------------------------------------------

def load_checkpoint(path: str, device=None) -> tuple:
    """Rebuild an LSTMForecastModel from a checkpoint; return (model, ckpt)."""
    device = device or torch.device("cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    m = ckpt["model_config"]
    lstm_cfg = m.get("lstm", {})
    model = LSTMForecastModel(
        input_dim=int(ckpt["input_dim"]),
        hidden_dim=m.get("hidden_dim", 64),
        num_layers=m.get("num_layers", 2),
        pred_len=int(ckpt["pred_len"]),
        dropout=m.get("dropout", 0.2),
        bidirectional=lstm_cfg.get("bidirectional", True),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt


def load_normalizer(path: str):
    """Load the saved normalizer, or None if the file does not exist."""
    if path and os.path.exists(path):
        return StandardNormalizer.load(path)
    return None


# ---------------------------------------------------------------------------
# Prediction (returns Celsius-domain arrays)
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(model, loader, device, normalizer=None) -> tuple:
    """
    Run the model and return (y_true, y_pred) in CELSIUS, shape (N, pred_len).

    The model predicts in normalized space when a normalizer is given; both
    predictions and ground truth are inverse-transformed before returning.
    """
    model.eval()
    trues, preds = [], []
    for X, y in loader:
        target = make_target(y)                  # raw Celsius (N, pred_len)
        Xin = X.to(device)
        if normalizer is not None:
            Xin = normalizer.transform(Xin)
        out = model(Xin).cpu()
        if normalizer is not None:
            out = normalizer.inverse_transform_target(out)
        preds.append(out.numpy())
        trues.append(target.numpy())
    if not preds:
        return np.empty((0, 0)), np.empty((0, 0))
    return np.concatenate(trues, 0), np.concatenate(preds, 0)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def evaluate_dataset(model, dataset, config: dict, normalizer=None) -> dict:
    """Evaluate on a dataset; return {rmse, mae, waveform_similarity} (Celsius)."""
    device = select_device(config)
    model.to(device)
    loader = DataLoader(dataset,
                        batch_size=int(config["training"].get("batch_size", 32)),
                        shuffle=False)
    y_true, y_pred = predict(model, loader, device, normalizer)
    return compute_all_metrics(y_true, y_pred)


def evaluate_with_predictions(model, dataset, config: dict, normalizer=None) -> dict:
    """
    Evaluate and also return the raw Celsius prediction arrays so callers can
    write predictions.csv and grouped metrics.

    Returns: {metrics, y_true, y_pred}.
    """
    device = select_device(config)
    model.to(device)
    loader = DataLoader(dataset,
                        batch_size=int(config["training"].get("batch_size", 32)),
                        shuffle=False)
    y_true, y_pred = predict(model, loader, device, normalizer)
    return {
        "metrics": compute_all_metrics(y_true, y_pred),
        "y_true": y_true,
        "y_pred": y_pred,
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_metrics_csv(metrics: dict, path: str, extra: dict = None) -> None:
    """Write a single-row metrics CSV (extra columns prepended)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    row = dict(extra or {})
    row.update(metrics)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def save_predictions_csv(y_true: np.ndarray, y_pred: np.ndarray, path: str,
                         exp_ids=None) -> None:
    """
    Write per-(sample, step) predictions in Celsius.

    Columns: sample_index, experiment_id, step, y_true, y_pred, error
             (error = y_pred - y_true). experiment_id omitted if exp_ids None.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    n, pred_len = y_true.shape
    has_ids = exp_ids is not None
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["sample_index"]
        if has_ids:
            header.append("experiment_id")
        header += ["step", "y_true", "y_pred", "error"]
        writer.writerow(header)
        for i in range(n):
            for s in range(pred_len):
                yt = float(y_true[i, s])
                yp = float(y_pred[i, s])
                row = [i]
                if has_ids:
                    row.append(exp_ids[i])
                row += [s, f"{yt:.6f}", f"{yp:.6f}", f"{yp - yt:.6f}"]
                writer.writerow(row)


def save_grouped_metrics_csv(y_true: np.ndarray, y_pred: np.ndarray,
                             group_labels, path: str, extra: dict = None) -> dict:
    """
    Compute per-group metrics and write them (one row per group).

    Returns the grouped-metrics dict. `extra` columns (e.g. model, simulated)
    are repeated on every row.
    """
    grouped = compute_grouped_metrics(y_true, y_pred, group_labels)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    base = dict(extra or {})
    fieldnames = (["group"] + list(base.keys())
                  + ["rmse", "mae", "waveform_similarity", "sample_count"])
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for group, m in grouped.items():
            row = {"group": group}
            row.update(base)
            row.update(m)
            writer.writerow(row)
    return grouped
