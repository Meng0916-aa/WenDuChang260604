"""
Plots for temporal feature analysis (script 09).

- A thermal-cycle curve together with its dT/dt derivative.
- An overview of key temporal features across experiments (bar charts).

matplotlib only (no seaborn), default colours, English labels for paper use.
Figures are saved as both PNG and PDF. If the data comes from SIM_* files,
"SIMULATED" is added to titles / file names so plots are never mistaken for
real experimental results.
"""

import os
import csv

import matplotlib
matplotlib.use("Agg")          # headless backend
import matplotlib.pyplot as plt

# Reuse the shared PNG+PDF saver.
from visualization.plot_curves import save_figure


def plot_curve_with_derivative(time_axis, curve, derivative, output_path,
                               title: str = "Thermal Cycle and dT/dt",
                               dpi: int = 200, save_png: bool = True,
                               save_pdf: bool = True, figsize=(8, 5)) -> list:
    """
    Plot a thermal-cycle curve (top) and its dT/dt derivative (bottom).

    Args:
        time_axis: 1-D time (seconds).
        curve: 1-D temperature (Celsius).
        derivative: 1-D dT/dt (Celsius/second).
        output_path: path WITHOUT extension OR with one; the stem + parent
                     dir are used (PNG and PDF are written alongside).
        title: figure title (English).

    Returns:
        List of written file paths.
    """
    out_dir = os.path.dirname(output_path) or "."
    name = os.path.splitext(os.path.basename(output_path))[0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                   gridspec_kw={"height_ratios": [3, 2]})
    ax1.plot(time_axis, curve)
    ax1.set_ylabel("Temperature (deg C)")
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)

    ax2.plot(time_axis, derivative)
    ax2.axhline(0.0, linewidth=0.8)
    ax2.set_ylabel("dT/dt (deg C/s)")
    ax2.set_xlabel("Time (s)")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return save_figure(fig, out_dir, name, dpi=dpi,
                       save_png=save_png, save_pdf=save_pdf)


# Features charted by the overview, with axis labels.
_OVERVIEW_FEATURES = [
    ("tmax_peak_temperature", "Peak temperature (deg C)"),
    ("tmax_max_cooling_rate", "Max cooling rate (deg C/s)"),
    ("tmax_dwell_time_above_threshold", "Dwell time above threshold (s)"),
    ("tmax_fluctuation_index", "Fluctuation index"),
]


def _read_feature_csv(feature_csv: str):
    """Return (rows, simulated_flag)."""
    with open(feature_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    simulated = any(str(r.get("simulated", "")).lower() in ("true", "1", "yes")
                    for r in rows)
    return rows, simulated


def plot_temporal_feature_overview(feature_csv: str, output_dir: str,
                                   dpi: int = 200, save_png: bool = True,
                                   save_pdf: bool = True, figsize=(9, 6)) -> list:
    """
    Read temporal_features.csv and draw bar charts of key features per
    experiment: peak_temperature, max_cooling_rate, dwell_time_above_threshold,
    fluctuation_index (using the tmax curve).

    Returns:
        List of written file paths (empty if the CSV has no rows).
    """
    rows, simulated = _read_feature_csv(feature_csv)
    if not rows:
        return []

    tag = " (SIMULATED)" if simulated else ""
    suffix = "_SIMULATED" if simulated else ""
    labels = [r.get("experiment_id", str(i)) for i, r in enumerate(rows)]

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.ravel()
    x = list(range(len(labels)))

    for ax, (key, ylabel) in zip(axes, _OVERVIEW_FEATURES):
        vals = []
        for r in rows:
            try:
                vals.append(float(r.get(key, "nan")))
            except (TypeError, ValueError):
                vals.append(float("nan"))
        ax.bar(x, vals)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(key, fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"Temporal Feature Overview{tag}")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return save_figure(fig, output_dir, f"temporal_feature_overview{suffix}",
                       dpi=dpi, save_png=save_png, save_pdf=save_pdf)
