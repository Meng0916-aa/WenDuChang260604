"""
Bar charts comparing metrics across models or magnetic-field groups.

matplotlib only (no seaborn), default colours, English labels.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Re-export the shared saver so callers can use one import.
from visualization.plot_curves import save_figure  # noqa: F401


def plot_metric_bars(labels, metric_values, metric_name: str = "RMSE",
                     title: str = None, ylabel: str = None, figsize=(7, 4)):
    """
    Draw a bar chart of one metric across categories.

    Args:
        labels: category names (e.g. ["with_B", "without_B"] or model names).
        metric_values: numeric values aligned with `labels`.
        metric_name: used in the default title / y-label.
        title: optional explicit title (English).
        ylabel: optional explicit y-axis label.

    Returns:
        The matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    x = list(range(len(labels)))
    bars = ax.bar(x, metric_values)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel(ylabel or metric_name)
    ax.set_title(title or f"{metric_name} by group")
    ax.grid(True, axis="y", alpha=0.3)

    # annotate values
    for rect, val in zip(bars, metric_values):
        ax.text(rect.get_x() + rect.get_width() / 2.0,
                rect.get_height(),
                f"{val:.3g}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    return fig


def plot_grouped_metric_bars(group_labels, metrics_per_group: dict,
                             metric_keys=("rmse", "mae", "waveform_similarity"),
                             title: str = "Metrics by group", figsize=(9, 4)):
    """
    Grouped bar chart: clusters of bars (one per metric) for each group.

    Args:
        group_labels: list of group names.
        metrics_per_group: {group_label: {metric_key: value, ...}}.
        metric_keys: which metrics to plot.

    Returns:
        The matplotlib Figure.
    """
    n_groups = len(group_labels)
    n_metrics = len(metric_keys)
    width = 0.8 / max(1, n_metrics)

    fig, ax = plt.subplots(figsize=figsize)
    base_x = list(range(n_groups))
    for mi, key in enumerate(metric_keys):
        vals = [float(metrics_per_group[g].get(key, 0.0)) for g in group_labels]
        offs = [x + mi * width for x in base_x]
        ax.bar(offs, vals, width=width, label=key)

    ax.set_xticks([x + (n_metrics - 1) * width / 2.0 for x in base_x])
    ax.set_xticklabels(group_labels)
    ax.set_ylabel("Metric value")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig
