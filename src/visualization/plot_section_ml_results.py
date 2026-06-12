"""
Plots for section-level ML quality prediction (script 16).

Figures saved under results/figures/section_ml/:
  - regression_pred_vs_true_<target>_<model>_<input_set>.png  (+ .pdf)
  - classification_confusion_matrix_<model>_<input_set>.png   (+ .pdf)
  - feature_importance_<target>_<model>_<input_set>.png       (+ .pdf)
  - input_set_performance_comparison.png                      (+ .pdf)

matplotlib only (NO seaborn), default colours, English labels. The pred-vs-true,
confusion-matrix and feature-importance plotters are reused from
``visualization.plot_ml_quality``; this module adds the input-set comparison
chart and the section-specific filename conventions used by script 16.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visualization.plot_curves import save_figure
from visualization.plot_ml_quality import (
    plot_confusion_matrix, plot_feature_importance, plot_regression_prediction,
)

__all__ = [
    "plot_confusion_matrix",
    "plot_feature_importance",
    "plot_regression_prediction",
    "plot_input_set_comparison",
]


def plot_input_set_comparison(rows, output_dir, metric_key, metric_label,
                              name="input_set_performance_comparison",
                              title=None, dpi=200,
                              save_png=True, save_pdf=True, figsize=(9, 5)) -> list:
    """Grouped bar chart comparing input sets across models for one metric.

    Args:
        rows: iterable of dicts, each with keys 'input_set', 'model' and the
            chosen ``metric_key`` (e.g. 'f1' or 'r2'). Rows for a single task.
        output_dir: figures directory.
        metric_key: dict key to read the metric value from each row.
        metric_label: y-axis label (e.g. "Macro F1" or "R^2").
        title: optional title.

    Returns:
        List of written file paths (empty if there is nothing to plot).
    """
    rows = [r for r in rows if r.get(metric_key) is not None]
    if not rows:
        return []

    input_sets = sorted({str(r["input_set"]) for r in rows})
    models = sorted({str(r["model"]) for r in rows})

    # value[input_set][model] -> metric (NaN if absent)
    table = {iset: {m: np.nan for m in models} for iset in input_sets}
    for r in rows:
        table[str(r["input_set"])][str(r["model"])] = float(r[metric_key])

    n_groups = len(input_sets)
    n_models = len(models)
    x = np.arange(n_groups)
    width = 0.8 / max(1, n_models)

    fig, ax = plt.subplots(figsize=figsize)
    for j, m in enumerate(models):
        vals = [table[iset][m] for iset in input_sets]
        ax.bar(x + j * width, vals, width, label=m)

    ax.set_xticks(x + width * (n_models - 1) / 2.0)
    ax.set_xticklabels(input_sets)
    ax.set_xlabel("Input feature set")
    ax.set_ylabel(metric_label)
    ax.set_title(title or f"Input-set comparison ({metric_label})")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return save_figure(fig, output_dir, name, dpi=dpi,
                       save_png=save_png, save_pdf=save_pdf)


def confusion_matrix_from_predictions(y_true, y_pred, labels=None):
    """Build a confusion matrix (list-of-lists) + labels from prediction columns.

    Kept here so script 16 can rebuild a CM from the classification predictions
    table without re-training.
    """
    from sklearn.metrics import confusion_matrix
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return cm.tolist(), labels
