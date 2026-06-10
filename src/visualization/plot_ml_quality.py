"""
Plots for ML quality assessment (script 12).

- Confusion matrix (classification).
- Random Forest feature importance.
- Regression prediction (true vs predicted scatter).

matplotlib only (no seaborn), default colours, English labels. Figures are
saved as PNG and PDF via the shared saver. If results are derived from
SIMULATED inputs, callers pass a SIMULATED-tagged name/title.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visualization.plot_curves import save_figure


def plot_confusion_matrix(cm, labels, output_dir, name="ml_quality_confusion_matrix",
                          title="Confusion Matrix", dpi=200,
                          save_png=True, save_pdf=True, figsize=(5, 4)) -> list:
    """
    Plot a confusion matrix as an annotated image.

    Args:
        cm: 2-D array-like (n_classes x n_classes).
        labels: class labels (length n_classes).
        output_dir: figures directory.
    """
    cm = np.asarray(cm, dtype=float)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    thresh = cm.max() / 2.0 if cm.size and cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    return save_figure(fig, output_dir, name, dpi=dpi,
                       save_png=save_png, save_pdf=save_pdf)


def plot_feature_importance(importance, output_dir, name="ml_feature_importance",
                            title="Random Forest Feature Importance", dpi=200,
                            save_png=True, save_pdf=True, top_k=None,
                            figsize=(8, 5)) -> list:
    """
    Horizontal bar chart of feature importances.

    Args:
        importance: list of {"feature","importance"} dicts (any order) OR a
                    dict {feature: importance}.
        output_dir: figures directory.
        top_k: if set, show only the top-k features.
    """
    if isinstance(importance, dict):
        pairs = list(importance.items())
    else:
        pairs = [(d["feature"], d["importance"]) for d in importance]
    pairs.sort(key=lambda t: t[1], reverse=True)
    if top_k:
        pairs = pairs[:top_k]
    pairs.reverse()  # largest on top in a horizontal bar

    feats = [p[0] for p in pairs]
    vals = [p[1] for p in pairs]

    fig, ax = plt.subplots(figsize=figsize)
    y = range(len(feats))
    ax.barh(list(y), vals)
    ax.set_yticks(list(y))
    ax.set_yticklabels(feats, fontsize=8)
    ax.set_xlabel("Importance")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return save_figure(fig, output_dir, name, dpi=dpi,
                       save_png=save_png, save_pdf=save_pdf)


def plot_regression_prediction(y_true, y_pred, output_dir,
                               name="ml_regression_prediction",
                               title="Regression: true vs predicted", dpi=200,
                               save_png=True, save_pdf=True, figsize=(5, 5)) -> list:
    """
    Scatter of predicted vs true values with the y = x reference line.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(y_true, y_pred)
    if y_true.size:
        lo = float(min(y_true.min(), y_pred.min()))
        hi = float(max(y_true.max(), y_pred.max()))
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
    ax.set_xlabel("True value")
    ax.set_ylabel("Predicted value")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return save_figure(fig, output_dir, name, dpi=dpi,
                       save_png=save_png, save_pdf=save_pdf)
