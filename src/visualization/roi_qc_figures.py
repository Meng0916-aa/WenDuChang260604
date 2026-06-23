"""
ROI-strategy QC / data-audit figures (Python / matplotlib).

These are DATA-QUALITY AUDIT figures for the 57-track unified-ROI evaluation,
NOT publication figures. They follow the nature-figure Python conventions
(Arial sans, editable text, restrained style, unified font/line width) but write
to results/qc/roi/ at QC dpi.

Conventions enforced here:
  * image panels keep the ORIGINAL array orientation (origin at top-left, row
    index increasing downward) and annotate that image-UP is the scan direction;
  * the thermal colormap is fixed to the VALID band 300-1800 C — it is NEVER
    scaled by the 6553.5 C sentinel. above_range / hard_saturation pixels are
    shown as overlay markers (location only), never as real temperatures;
  * read-only on the temperature matrices (mmap); no matrix is modified.

No new fonts and no new dependencies are installed.
"""

import os
import csv

import numpy as np

import matplotlib
matplotlib.use("Agg")           # headless
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from processing.hot_region_mask import classify_temperature_states

# --- nature-figure Python publication style (unified font / size / line width) ---
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})

VALID_MIN, VALID_MAX = 300.0, 1800.0      # thermal colormap clip (never 6553.5)
HARD_SAT = 6500.0
_THERMAL_CMAP = "inferno"
_C_OLD = "#1f77b4"        # legacy ROI
_C_CAND = "#2ca02c"       # candidate global ROI
_C_700 = "#d62728"        # 700 C envelope
_C_800 = "#ff7f0e"        # 800 C core
_C_ABOVE = "#00e5ff"      # above-range marker (location only)
_C_HARD = "#ff00ff"       # hard-saturation marker (location only)


def _abs(root, path):
    return path if os.path.isabs(path) else os.path.join(root, path)


def _grid(n):
    """Rows, cols for an n-panel near-square contact sheet (57 -> 8x8)."""
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    return rows, cols


def _rect_patch(rect, color, lw=0.8, ls="-"):
    top, left, bottom_excl, right_excl = rect
    return Rectangle((left - 0.5, top - 0.5), right_excl - left, bottom_excl - top,
                     fill=False, edgecolor=color, linewidth=lw, linestyle=ls)


def _scan_arrow(ax, h, w):
    """Annotate that image-UP is the scan direction (row index decreasing)."""
    ax.annotate("", xy=(0.06 * w, 0.10 * h), xytext=(0.06 * w, 0.40 * h),
                arrowprops=dict(arrowstyle="-|>", color="white", lw=0.8))


def _save(fig, out_dir, name, also_pdf=False, dpi=200):
    fig.savefig(os.path.join(out_dir, f"{name}.png"), dpi=dpi, bbox_inches="tight")
    if also_pdf:
        fig.savefig(os.path.join(out_dir, f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# per-track temporal-max projection (read-only)
# ---------------------------------------------------------------------------

def _projection(matrix_root, sid, max_frames=None):
    """Temporal max projection over effective frames (frames[1:]); read-only."""
    path = os.path.join(matrix_root, f"{sid}.npy")
    arr = np.load(path, mmap_mode="r")
    n = arr.shape[0]
    hi = n if max_frames is None else min(n, 1 + int(max_frames))
    proj = np.asarray(arr[1:hi], dtype=np.float32).max(axis=0)
    del arr
    return proj


# ---------------------------------------------------------------------------
# image contact sheets
# ---------------------------------------------------------------------------

def _overlay_contact_sheet(order, matrix_root, out_dir, name, rect=None,
                           title="", max_frames=None, also_pdf=False):
    n = len(order)
    rows, cols = _grid(n)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.7, rows * 1.55))
    axes = np.atleast_1d(axes).ravel()
    for k, sid in enumerate(order):
        ax = axes[k]
        proj = _projection(matrix_root, sid, max_frames)
        h, w = proj.shape
        ax.imshow(proj, cmap=_THERMAL_CMAP, vmin=VALID_MIN, vmax=VALID_MAX,
                  interpolation="nearest")
        # above-range / hard-saturation locations (markers only, not temperatures)
        st = classify_temperature_states(proj, VALID_MIN, VALID_MAX, HARD_SAT)
        ys, xs = np.nonzero(st["above_range"])
        if xs.size:
            ax.scatter(xs, ys, s=0.4, c=_C_ABOVE, marker=".", linewidths=0)
        ys, xs = np.nonzero(st["hard_saturation"])
        if xs.size:
            ax.scatter(xs, ys, s=0.6, c=_C_HARD, marker=".", linewidths=0)
        if rect is not None:
            ax.add_patch(_rect_patch(rect, _C_CAND if name.startswith("candidate") else _C_OLD, lw=0.8))
        _scan_arrow(ax, h, w)
        ax.set_title(sid, fontsize=5.5, pad=1.5)
        ax.set_xticks([]); ax.set_yticks([])
    for k in range(n, len(axes)):
        axes[k].axis("off")
    fig.suptitle(title + "  (image-up = scan direction; colormap 300-1800 C)", fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    _save(fig, out_dir, name, also_pdf=also_pdf)


def _mask_contact_sheet(order, per_track, out_dir, name, key, color, title):
    n = len(order)
    rows, cols = _grid(n)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.7, rows * 1.55))
    axes = np.atleast_1d(axes).ravel()
    for k, sid in enumerate(order):
        ax = axes[k]
        m = per_track[sid][key]
        ax.imshow(m, cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
        h, w = m.shape
        _scan_arrow(ax, h, w)
        ax.set_title(sid, fontsize=5.5, pad=1.5, color=color)
        ax.set_xticks([]); ax.set_yticks([])
    for k in range(n, len(axes)):
        axes[k].axis("off")
    fig.suptitle(title, fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    _save(fig, out_dir, name)


def _centroid_contact_sheet(order, per_track, out_dir, name, h, w):
    n = len(order)
    rows, cols = _grid(n)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.7, rows * 1.55))
    axes = np.atleast_1d(axes).ravel()
    for k, sid in enumerate(order):
        ax = axes[k]
        cents = [c for c in per_track[sid]["centroids"] if c is not None]
        if cents:
            xs = [c[0] for c in cents]; ys = [c[1] for c in cents]
            ax.plot(xs, ys, "-", color=_C_700, lw=0.7)
            ax.scatter([xs[0]], [ys[0]], s=6, c="k", marker="o", zorder=3)   # start
            ax.scatter([xs[-1]], [ys[-1]], s=8, c=_C_CAND, marker="^", zorder=3)  # end (up)
        ax.set_xlim(0, w); ax.set_ylim(h, 0)     # image orientation: y down
        ax.set_title(sid, fontsize=5.5, pad=1.5)
        ax.set_xticks([]); ax.set_yticks([])
    for k in range(n, len(axes)):
        axes[k].axis("off")
    fig.suptitle("Melt-pool centroid trajectory (image-up = scan direction; ^ = last frame)",
                 fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    _save(fig, out_dir, name)


# ---------------------------------------------------------------------------
# quantitative grids
# ---------------------------------------------------------------------------

def _track_table(track_rows):
    return {r["sample_id"]: r for r in track_rows}


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _bbox_position_distribution(order, per_track, out_dir, name, h, w, old_roi, cand):
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    for sid in order:
        b = per_track[sid]["track_bbox700"]
        if b is None:
            continue
        ax.add_patch(_rect_patch(b, "#888888", lw=0.4, ls="-"))
    ax.add_patch(_rect_patch(old_roi, _C_OLD, lw=1.4))
    ax.add_patch(_rect_patch(cand, _C_CAND, lw=1.4, ls="--"))
    ax.set_xlim(0, w); ax.set_ylim(h, 0)
    ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")
    ax.set_title("Per-track 700 C bbox (grey) vs legacy (blue) / candidate (green)")
    ax.plot([], [], color=_C_OLD, label="legacy ROI")
    ax.plot([], [], color=_C_CAND, ls="--", label="candidate ROI")
    ax.legend(loc="lower right", fontsize=6)
    _save(fig, out_dir, name, also_pdf=True)


def _bbox_size_distribution(track_rows, out_dir, name):
    ws = [_num(r["bbox700_width_px"]) for r in track_rows]
    hs = [_num(r["bbox700_height_px"]) for r in track_rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 3.0))
    a1.scatter(ws, hs, s=12, c=_C_700, alpha=0.8)
    a1.set_xlabel("700 C bbox width (px)"); a1.set_ylabel("700 C bbox height (px)")
    a1.set_title("bbox size per track")
    a2.hist([w for w in ws if not np.isnan(w)], bins=12, color=_C_700, alpha=0.7, label="width")
    a2.hist([h for h in hs if not np.isnan(h)], bins=12, color=_C_800, alpha=0.7, label="height")
    a2.set_xlabel("px"); a2.set_ylabel("count"); a2.set_title("bbox size distribution")
    a2.legend(fontsize=6)
    fig.tight_layout()
    _save(fig, out_dir, name)


def _bar_by_sample(track_rows, out_dir, name, keys, labels, colors, ylabel,
                   title, hline=None, hline_label=None):
    sids = [r["sample_id"] for r in track_rows]
    x = np.arange(len(sids))
    fig, ax = plt.subplots(figsize=(max(7.0, len(sids) * 0.16), 3.2))
    nb = len(keys)
    width = 0.8 / nb
    for i, (k, lab, col) in enumerate(zip(keys, labels, colors)):
        vals = [_num(r.get(k)) for r in track_rows]
        ax.bar(x + (i - (nb - 1) / 2) * width, vals, width=width, color=col, label=lab)
    if hline is not None:
        ax.axhline(hline, color="k", lw=0.8, ls="--", label=hline_label)
    ax.set_xticks(x); ax.set_xticklabels(sids, rotation=90, fontsize=4.5)
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(fontsize=6, ncol=min(4, nb + (1 if hline is not None else 0)))
    fig.tight_layout()
    _save(fig, out_dir, name)


def _tracking_window_curve(win_rows, out_dir, name):
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    sizes = [w["window_height_px"] * w["window_width_px"] for w in win_rows]
    cov = [w["coverage_rate"] for w in win_rows]
    labels = [f'{w["window_height_px"]}x{w["window_width_px"]}' for w in win_rows]
    order = np.argsort(sizes)
    sizes = [sizes[i] for i in order]; cov = [cov[i] for i in order]
    labels = [labels[i] for i in order]
    ax.plot(sizes, cov, "-o", color=_C_CAND, lw=1.0, ms=4)
    for s, c, l in zip(sizes, cov, labels):
        ax.annotate(l, (s, c), fontsize=5.5, xytext=(0, 4), textcoords="offset points",
                    ha="center")
    ax.axhline(0.99, color="k", lw=0.8, ls="--", label="99% target")
    ax.set_xlabel("tracking window area (px)"); ax.set_ylabel("full-coverage rate")
    ax.set_title("Tracking-window coverage vs size")
    ax.legend(fontsize=6)
    fig.tight_layout()
    _save(fig, out_dir, name)


def _strategy_comparison(summary, out_dir, name):
    cand = summary.get("recommended_global_roi") or {}
    frac = _num(cand.get("fraction_of_full_frame"))
    cov700 = _num(summary.get("global_roi_min_coverage_700"))
    cov800 = _num(summary.get("global_roi_min_coverage_800"))
    twcov = _num(summary.get("tracking_window_coverage_rate"))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.2))
    # left: ROI footprint of the three strategies (fraction of full frame)
    names = ["A: global ROI", "B: global+window", "C: full frame"]
    win = summary.get("recommended_tracking_window", {})
    win_frac = (_num(win.get("height_px")) * _num(win.get("width_px"))) / (
        summary["full_frame_dimensions"][0] * summary["full_frame_dimensions"][1])
    fracs = [frac, win_frac, 1.0]
    cols = [_C_CAND, _C_800, "#999999"]
    a1.bar(names, fracs, color=cols)
    a1.set_ylabel("analysis footprint (fraction of full frame)")
    a1.set_title("ROI footprint by strategy")
    a1.tick_params(axis="x", labelrotation=20, labelsize=6)
    # right: coverage guarantees
    a2.bar(["700 env", "800 core", "track win"], [cov700, cov800, twcov],
           color=[_C_700, _C_800, _C_CAND])
    a2.set_ylim(0.9, 1.005); a2.set_ylabel("min coverage / rate")
    a2.set_title(f"recommended: {summary.get('recommended_strategy', '?')}")
    fig.tight_layout()
    _save(fig, out_dir, name, also_pdf=True)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def generate_all_figures(results, matrix_root, master_csv, calibration_config,
                         out_dir, envelope_threshold=700.0, core_threshold=800.0,
                         min_component_area=9, max_frames=None):
    """Generate all ROI-strategy QC figures into out_dir (results/qc/roi)."""
    os.makedirs(out_dir, exist_ok=True)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    matrix_root = _abs(root, matrix_root)

    per_track = results["per_track"]
    order = list(per_track.keys())
    track_rows = results["track_rows"]
    summary = results["summary"]
    old_roi = results["old_roi"]
    cand_rect = results["candidate_rect"]
    H, W = summary["full_frame_dimensions"]

    # 1-2. image overlay contact sheets (thermal projection + ROI box)
    _overlay_contact_sheet(order, matrix_root, out_dir,
                           "old_roi_overlay_contact_sheet", rect=old_roi,
                           title="Legacy ROI over temporal-max projection",
                           max_frames=max_frames, also_pdf=True)
    _overlay_contact_sheet(order, matrix_root, out_dir,
                           "candidate_global_roi_overlay_contact_sheet", rect=cand_rect,
                           title="Candidate global ROI over temporal-max projection",
                           max_frames=max_frames, also_pdf=True)
    # 3-4. cleaned mask projections
    _mask_contact_sheet(order, per_track, out_dir,
                        "cleaned_700C_projection_contact_sheet", "union700", _C_700,
                        "Cleaned 700 C envelope union (per track)")
    _mask_contact_sheet(order, per_track, out_dir,
                        "cleaned_800C_projection_contact_sheet", "union800", _C_800,
                        "Cleaned 800 C core union (per track)")
    # 5. centroid trajectories
    _centroid_contact_sheet(order, per_track, out_dir,
                            "centroid_trajectory_contact_sheet", H, W)
    # 6-7. bbox distributions
    _bbox_position_distribution(order, per_track, out_dir,
                                "bbox_position_distribution", H, W, old_roi, cand_rect)
    _bbox_size_distribution(track_rows, out_dir, "bbox_size_distribution")
    # 8. vertical travel by sample
    _bar_by_sample(track_rows, out_dir, "centroid_vertical_travel_by_sample",
                   ["centroid_vertical_travel_px"], ["vertical travel (px)"], [_C_700],
                   "px", "Melt-pool centroid vertical travel by sample")
    # 9. coverage by sample
    _bar_by_sample(track_rows, out_dir, "roi_coverage_by_sample",
                   ["old_roi_coverage_700", "old_roi_coverage_800",
                    "candidate_roi_coverage_700", "candidate_roi_coverage_800"],
                   ["old 700", "old 800", "cand 700", "cand 800"],
                   [_C_OLD, "#7fb3d5", _C_CAND, "#82c99a"],
                   "coverage", "ROI coverage by sample", hline=0.999, hline_label="99.9%")
    # 10. edge distance by sample
    _bar_by_sample(track_rows, out_dir, "roi_edge_distance_by_sample",
                   ["old_roi_min_edge_distance_px", "candidate_roi_min_edge_distance_px"],
                   ["legacy ROI", "candidate ROI"], [_C_OLD, _C_CAND],
                   "min edge distance (px)", "Main-body distance to ROI edge by sample",
                   hline=10, hline_label="10 px safety")
    # 11. tracking-window coverage curve
    _tracking_window_curve(results["win_rows"], out_dir, "tracking_window_coverage_curve")
    # 12. strategy comparison
    _strategy_comparison(summary, out_dir, "roi_strategy_comparison")

    return out_dir
