"""
Conversion quality-control for per-track temperature matrices.

For one converted single track (<sample_id>.npy, float32 Celsius, N x 512 x 640)
this produces visual + numeric QC under results/qc/conversion/<sample_id>/:

    first_frame.png
    middle_frame.png
    max_temperature_frame.png
    temporal_max_projection.png
    frame_peak_temperature_curve.png
    conversion_qc.json

All plots use real Celsius units. Because frame_rate_fps is unknown/blank, the
time axis is the FRAME INDEX (never seconds). Read-only on the source data.
"""

import os
import json

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless; no display
import matplotlib.pyplot as plt


def _title(sample_id, scan_speed, n, extra=""):
    sp = f"{scan_speed} mm/min" if scan_speed not in (None, "") else "speed n/a"
    base = f"{sample_id} | {sp} | {n} frames"
    return f"{base} | {extra}" if extra else base


def _save_frame(frame, path, title):
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(frame, cmap="inferno")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Temperature (C)")
    ax.set_title(title)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _save_curve(per_frame_max, path, title):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(per_frame_max.size), per_frame_max, lw=1.2)
    ax.set_title(title)
    ax.set_xlabel("frame index")          # NOT seconds — fps is unknown
    ax.set_ylabel("peak temperature (C)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run_conversion_qc(meta, row, qc_root, fmt):
    """Generate QC plots + conversion_qc.json for one converted track.

    Args:
        meta: the meta dict returned by convert_track (has output_file etc.).
        row:  the master-csv row (has scan_speed_mm_min, xtherm_count, ...).
        qc_root: base output dir (results/qc/conversion).
        fmt: format dict (uses 'valid_max' as the over-range/saturation guard).

    Returns a flat dict with all pilot-summary columns.
    """
    sample_id = meta["sample_id"]
    npy_path = meta["output_file"]
    scan_speed = row.get("scan_speed_mm_min", "")

    if not os.path.isabs(qc_root):
        qc_root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            qc_root)
    out_dir = os.path.join(qc_root, sample_id)
    os.makedirs(out_dir, exist_ok=True)

    data = np.load(npy_path)
    n, h, w = data.shape
    per_frame_max = data.reshape(n, -1).max(axis=1)
    max_frame_idx = int(np.argmax(per_frame_max))

    finite = data[np.isfinite(data)]
    t_min = float(finite.min()) if finite.size else float("nan")
    t_max = float(finite.max()) if finite.size else float("nan")
    t_mean = float(finite.mean()) if finite.size else float("nan")
    zero_ratio = float(np.count_nonzero(data == 0.0) / data.size)
    nan_count = int(np.count_nonzero(np.isnan(data)))
    inf_count = int(np.count_nonzero(np.isinf(data)))

    valid_max = float(fmt.get("valid_max", 3000.0))
    warnings = []
    possible_saturation = bool(np.isfinite(t_max) and t_max >= valid_max)
    if possible_saturation:
        warnings.append(f"max {t_max:.1f} C >= valid_max {valid_max:.0f} C (over-range/saturation?)")
    possible_corruption = bool(nan_count or inf_count or zero_ratio > 0.95
                               or (n, h, w) != tuple(meta.get("output_shape", (n, h, w))))
    if nan_count or inf_count:
        warnings.append(f"non-finite values: nan={nan_count} inf={inf_count}")
    if zero_ratio > 0.95:
        warnings.append(f"zero_pixel_ratio {zero_ratio:.3f} very high")

    qc_status = "fail" if possible_corruption else ("warn" if possible_saturation else "pass")

    # --- plots ---
    _save_frame(data[0], os.path.join(out_dir, "first_frame.png"),
                _title(sample_id, scan_speed, n, "first frame (idx 0)"))
    _save_frame(data[n // 2], os.path.join(out_dir, "middle_frame.png"),
                _title(sample_id, scan_speed, n, f"middle frame (idx {n // 2})"))
    _save_frame(data[max_frame_idx], os.path.join(out_dir, "max_temperature_frame.png"),
                _title(sample_id, scan_speed, n, f"max-temp frame (idx {max_frame_idx})"))
    _save_frame(data.max(axis=0), os.path.join(out_dir, "temporal_max_projection.png"),
                _title(sample_id, scan_speed, n, "temporal max projection"))
    _save_curve(per_frame_max, os.path.join(out_dir, "frame_peak_temperature_curve.png"),
                _title(sample_id, scan_speed, n, "per-frame peak temperature"))

    qc_json = {
        "sample_id": sample_id,
        "frame_count": n,
        "shape": [n, h, w],
        "dtype": str(data.dtype),
        "min_temperature_C": t_min,
        "max_temperature_C": t_max,
        "mean_temperature_C": t_mean,
        "zero_pixel_ratio": zero_ratio,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "max_temperature_frame_index": max_frame_idx,
        "possible_saturation": possible_saturation,
        "possible_corruption": possible_corruption,
        "qc_status": qc_status,
        "warnings": warnings,
    }
    with open(os.path.join(out_dir, "conversion_qc.json"), "w", encoding="utf-8") as f:
        json.dump(qc_json, f, indent=2, ensure_ascii=False)

    del data
    # Flat row for the pilot summary CSV.
    return {
        "sample_id": sample_id,
        "condition_id": meta.get("condition_id", row.get("condition_id", "")),
        "track_id": meta.get("track_id", row.get("track_id", "")),
        "scan_speed_mm_min": scan_speed,
        "xtherm_count": row.get("xtherm_count", ""),
        "output_frame_count": n,
        "output_shape": [n, h, w],
        "output_dtype": "float32",
        "min_temperature_C": t_min,
        "max_temperature_C": t_max,
        "mean_temperature_C": t_mean,
        "zero_pixel_ratio": zero_ratio,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "possible_saturation": possible_saturation,
        "possible_corruption": possible_corruption,
        "conversion_status": meta.get("conversion_status", ""),
        "qc_status": qc_status,
        "notes": "; ".join(warnings),
        "qc_dir": out_dir,
    }
