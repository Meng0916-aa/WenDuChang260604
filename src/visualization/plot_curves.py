"""
Plot true vs. predicted thermal-cycle curves and the prediction error.

matplotlib only (no seaborn), default colour cycle, English labels for
later use in a paper. Figures are saved as both PNG and PDF.
"""

import os

import matplotlib
matplotlib.use("Agg")          # headless backend
import matplotlib.pyplot as plt


def save_figure(fig, out_dir: str, name: str, dpi: int = 200,
                save_png: bool = True, save_pdf: bool = True) -> list:
    """Save a figure as PNG and/or PDF; return the written paths."""
    os.makedirs(out_dir, exist_ok=True)
    written = []
    if save_png:
        p = os.path.join(out_dir, f"{name}.png")
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        written.append(p)
    if save_pdf:
        p = os.path.join(out_dir, f"{name}.pdf")
        fig.savefig(p, bbox_inches="tight")
        written.append(p)
    plt.close(fig)
    return written


def plot_prediction_curve(y_true, y_pred, title: str = "Thermal Cycle Prediction",
                          figsize=(8, 4)):
    """
    Plot one sample's ground-truth, predicted, and error curves.

    Args:
        y_true: 1-D array (pred_len,) of ground-truth temperature (Celsius).
        y_pred: 1-D array (pred_len,) of predicted temperature (Celsius).
        title: figure title (English).

    Returns:
        The matplotlib Figure (caller saves via save_figure).
    """
    steps = list(range(len(y_true)))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(steps, y_true, marker="o", markersize=3, label="Ground truth")
    ax1.plot(steps, y_pred, marker="s", markersize=3, label="Prediction")
    ax1.set_ylabel("Temperature (deg C)")
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    error = [float(p) - float(t) for p, t in zip(y_pred, y_true)]
    ax2.plot(steps, error, marker="x", markersize=3)
    ax2.axhline(0.0, linewidth=0.8)
    ax2.set_ylabel("Error (deg C)")
    ax2.set_xlabel("Prediction step")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig
