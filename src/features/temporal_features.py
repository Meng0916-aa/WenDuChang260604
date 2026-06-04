"""
Temporal feature analysis for thermal-cycle curves.

Given a 1-D thermal-cycle curve (e.g. Tmax(t) from script 04), extract
time-series descriptors used for the Chapter-3 results analysis:
peak temperature/time, heating/cooling rates, dwell time above a threshold,
area under the curve, and a fluctuation index.

All temperatures are degrees Celsius; the time axis is seconds (derived from
the camera frame rate). matplotlib/torch are NOT imported here — this module
is pure numpy so every function is callable in isolation and unit-testable.
"""

import os
import glob

import numpy as np

# Curves produced by script 04 / src/preprocess/thermal_cycle.py
_CURVE_COLUMNS = ("tmax", "center_average", "hot_zone_average")


# ---------------------------------------------------------------------------
# Basic signal helpers
# ---------------------------------------------------------------------------

def compute_time_axis(num_frames: int, frame_rate_fps: float) -> np.ndarray:
    """
    Build a time axis in seconds for a sequence of frames.

    Args:
        num_frames: number of frames (N >= 0).
        frame_rate_fps: frames per second (> 0).

    Returns:
        float32 array (num_frames,) = [0, 1/fps, 2/fps, ...].
    """
    if frame_rate_fps <= 0:
        raise ValueError(f"frame_rate_fps must be > 0, got {frame_rate_fps}")
    n = int(num_frames)
    if n <= 0:
        return np.zeros((0,), dtype=np.float32)
    return (np.arange(n, dtype=np.float32) / float(frame_rate_fps)).astype(np.float32)


def smooth_curve(curve: np.ndarray, window_size: int) -> np.ndarray:
    """
    Simple centered moving-average smoothing.

    window_size <= 1 returns the curve unchanged (as float32). Edges are
    handled with 'edge' padding so the output length equals the input length.

    Args:
        curve: 1-D array.
        window_size: averaging window in samples.

    Returns:
        float32 array, same length as `curve`.
    """
    arr = np.asarray(curve, dtype=np.float32).ravel()
    w = int(window_size)
    if w <= 1 or arr.size == 0:
        return arr.copy()
    w = min(w, arr.size)
    pad = w // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(w, dtype=np.float32) / float(w)
    smoothed = np.convolve(padded, kernel, mode="same")
    # trim back to the original length
    return smoothed[pad:pad + arr.size].astype(np.float32)


def compute_derivative(curve: np.ndarray, time_axis: np.ndarray) -> np.ndarray:
    """
    Compute dT/dt in Celsius per second using central differences.

    Args:
        curve: 1-D temperature array (Celsius).
        time_axis: 1-D time array (seconds), same length as curve.

    Returns:
        float32 array (same length) of dT/dt. Returns zeros for length < 2.
    """
    arr = np.asarray(curve, dtype=np.float64).ravel()
    t = np.asarray(time_axis, dtype=np.float64).ravel()
    if arr.size != t.size:
        raise ValueError(
            f"curve ({arr.size}) and time_axis ({t.size}) length mismatch")
    if arr.size < 2:
        return np.zeros_like(arr, dtype=np.float32)
    return np.gradient(arr, t).astype(np.float32)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_temporal_features(curve: np.ndarray,
                              frame_rate_fps: float,
                              threshold: float = None,
                              smooth_window: int = 1) -> dict:
    """
    Extract temporal descriptors from one thermal-cycle curve.

    Args:
        curve: 1-D temperature array (Celsius).
        frame_rate_fps: camera frame rate (fps) -> defines the time axis.
        threshold: Celsius level for dwell-time. If None, dwell_time is 0.0.
        smooth_window: moving-average window applied before rate/derivative
                       features (<=1 disables smoothing).

    Returns:
        Dict of features (all floats). Keys:
          peak_temperature, peak_time, mean_temperature, std_temperature,
          min_temperature, max_heating_rate, max_cooling_rate,
          mean_cooling_rate, temperature_auc, dwell_time_above_threshold,
          fluctuation_index.
    """
    raw = np.asarray(curve, dtype=np.float32).ravel()
    n = raw.size
    if n == 0:
        raise ValueError("curve is empty")

    time_axis = compute_time_axis(n, frame_rate_fps)
    dt = 1.0 / float(frame_rate_fps)

    # Smooth for rate-based features; keep raw for amplitude features.
    sm = smooth_curve(raw, smooth_window)
    deriv = compute_derivative(sm, time_axis)            # Celsius / s

    peak_idx = int(np.argmax(raw))
    peak_temperature = float(raw[peak_idx])
    peak_time = float(time_axis[peak_idx])

    # Heating = positive dT/dt, cooling = negative dT/dt.
    max_heating_rate = float(np.max(deriv)) if deriv.size else 0.0
    max_cooling_rate = float(np.min(deriv)) if deriv.size else 0.0   # most negative
    cooling = deriv[deriv < 0.0]
    mean_cooling_rate = float(cooling.mean()) if cooling.size else 0.0

    # Area under the curve (trapezoidal) in Celsius*seconds.
    # np.trapz was renamed to np.trapezoid in NumPy 2.0; support both.
    _trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    temperature_auc = float(_trapezoid(raw.astype(np.float64),
                                       time_axis.astype(np.float64)))

    # Dwell time above threshold (seconds): count frames >= threshold * dt.
    if threshold is None:
        dwell_time = 0.0
    else:
        dwell_time = float(np.count_nonzero(raw >= float(threshold)) * dt)

    # Fluctuation index: std of the high-frequency residual (raw - smoothed),
    # normalized by mean temperature. Always >= 0.
    mean_temperature = float(raw.mean())
    residual = raw - sm
    denom = abs(mean_temperature) if abs(mean_temperature) > 1e-6 else 1.0
    fluctuation_index = float(np.std(residual) / denom)

    return {
        "peak_temperature": peak_temperature,
        "peak_time": peak_time,
        "mean_temperature": mean_temperature,
        "std_temperature": float(raw.std()),
        "min_temperature": float(raw.min()),
        "max_heating_rate": max_heating_rate,
        "max_cooling_rate": max_cooling_rate,
        "mean_cooling_rate": mean_cooling_rate,
        "temperature_auc": temperature_auc,
        "dwell_time_above_threshold": dwell_time,
        "fluctuation_index": fluctuation_index,
    }


# ---------------------------------------------------------------------------
# CSV-level helpers
# ---------------------------------------------------------------------------

def _temporal_cfg(config: dict) -> dict:
    """Pull the temporal_analysis block with sensible fallbacks."""
    return config.get("temporal_analysis", {}) or {}


def extract_features_from_thermal_cycle_csv(csv_path: str, config: dict) -> dict:
    """
    Read a thermal-cycle CSV and extract temporal features for each curve.

    The CSV is the output of script 04 with columns:
        frame, tmax, center_average, hot_zone_average

    Args:
        csv_path: path to one thermal-cycle CSV.
        config: full project config (uses config['temporal_analysis']).

    Returns:
        Dict mapping "<curve>_<feature>" -> value, plus 'experiment_id',
        'num_frames', and 'simulated' (True if filename starts with SIM_).
    """
    tcfg = _temporal_cfg(config)
    fps = float(tcfg.get("frame_rate_fps", 1000))
    smooth_window = int(tcfg.get("smooth_window", 1))
    threshold = tcfg.get("dwell_threshold", None)
    threshold = None if threshold is None else float(threshold)

    arr = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=np.float32)
    names = arr.dtype.names or ()

    exp_id = os.path.splitext(os.path.basename(csv_path))[0]
    out = {
        "experiment_id": exp_id,
        "simulated": exp_id.startswith("SIM_"),
    }

    num_frames = None
    for col in _CURVE_COLUMNS:
        if col not in names:
            continue
        curve = np.atleast_1d(np.asarray(arr[col], dtype=np.float32))
        num_frames = curve.size
        feats = extract_temporal_features(
            curve, frame_rate_fps=fps, threshold=threshold,
            smooth_window=smooth_window)
        for k, v in feats.items():
            out[f"{col}_{k}"] = v

    out["num_frames"] = int(num_frames) if num_frames is not None else 0
    return out


def batch_extract_temporal_features(input_dir: str, output_csv: str,
                                    config: dict) -> list:
    """
    Process every thermal-cycle CSV in a directory and write one combined CSV.

    Args:
        input_dir: directory of thermal-cycle CSVs (script 04 output).
        output_csv: path to write the combined feature table.
        config: full project config.

    Returns:
        List of per-experiment feature dicts (empty if no CSVs found).
    """
    import csv as _csv

    csv_files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    rows = [extract_features_from_thermal_cycle_csv(p, config) for p in csv_files]
    if not rows:
        return rows

    # Union of keys across rows, stable order: id/meta first, then sorted rest.
    lead = ["experiment_id", "num_frames", "simulated"]
    rest = sorted({k for r in rows for k in r} - set(lead))
    fieldnames = lead + rest

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    return rows
