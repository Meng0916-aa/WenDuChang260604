"""
Evaluation metrics for thermal cycle prediction.

Provides RMSE, MAE, and a waveform similarity index.
All functions operate on numpy float32 arrays.
"""

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Square Error.

    Args:
        y_true: ground truth, shape (...,).
        y_pred: prediction, same shape as y_true.

    Returns:
        Scalar RMSE value.
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error.

    Args:
        y_true: ground truth, shape (...,).
        y_pred: prediction, same shape as y_true.

    Returns:
        Scalar MAE value.
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def waveform_similarity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Waveform similarity based on Pearson correlation.

    Measures how well the predicted temperature curve matches the shape
    of the ground-truth curve, ignoring amplitude offset.

    Args:
        y_true: ground truth, shape (...,).
        y_pred: prediction, same shape as y_true.

    Returns:
        Similarity in [0, 1]. 1 = identical shape, 0 = no correlation.
    """
    y_true = np.asarray(y_true, dtype=np.float32).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float32).ravel()

    # Handle constant (zero-variance) signals
    std_true = np.std(y_true)
    std_pred = np.std(y_pred)
    if std_true < 1e-12 and std_pred < 1e-12:
        return 1.0
    if std_true < 1e-12 or std_pred < 1e-12:
        return 0.0

    corr = np.corrcoef(y_true, y_pred)[0, 1]
    if np.isnan(corr):
        return 0.0

    # Map [-1, 1] → [0, 1]
    return float(max(0.0, (corr + 1.0) / 2.0))


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute RMSE, MAE, and waveform similarity at once.

    Args:
        y_true: ground truth.
        y_pred: prediction.

    Returns:
        Dict with keys 'rmse', 'mae', 'waveform_similarity'.
    """
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "waveform_similarity": waveform_similarity(y_true, y_pred),
    }


def compute_grouped_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                            group_labels) -> dict:
    """
    Compute metrics separately for each group of samples.

    Args:
        y_true: (N, ...) ground truth.
        y_pred: (N, ...) predictions, aligned with y_true on axis 0.
        group_labels: length-N sequence assigning each sample to a group
                      (e.g. experiment id, or 'with_B' / 'without_B').

    Returns:
        Dict mapping group -> {rmse, mae, waveform_similarity, sample_count}.
        Groups are returned in sorted order for stable output.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.asarray(group_labels)

    out = {}
    for g in sorted(set(labels.tolist())):
        mask = labels == g
        m = compute_all_metrics(y_true[mask], y_pred[mask])
        m["sample_count"] = int(mask.sum())
        out[g] = m
    return out
