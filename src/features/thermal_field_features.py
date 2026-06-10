"""
Thermal-field feature extraction for ML-based cladding quality assessment.

Given a per-experiment ROI temperature matrix of shape N x H x W (N frames,
H height, W width, float32 Celsius), this module computes spatial temperature-
field descriptors per frame and aggregates them into ONE feature row per
experiment.

Why one row per experiment (not per frame): the real dataset is small and
adjacent frames within a run are highly correlated. Treating each frame as an
independent sample would leak information between train/test folds and grossly
inflate apparent performance. So every experiment contributes a single sample.

Extracted physical features (per experiment):
  T_max     peak_temperature            hottest melt-pool state
  T_avg     mean_temperature            overall melt-pool thermal state
  A_T       high-temperature area       melting extent
  G_T       temperature gradient        solidification driving force / stress
  R_c       cooling rate                microstructure refinement / solidification
  W_HAZ     heat-affected-zone width    heat-diffusion range
  sigma_T   temperature fluctuation     melt-pool thermal stability
  D_T       thermal-center offset       temperature-field symmetry

Pure NumPy; no torch / matplotlib imports, so each function is unit-testable.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Single-frame primitives
# ---------------------------------------------------------------------------

def compute_high_temperature_mask(frame: np.ndarray, threshold: float) -> np.ndarray:
    """
    Boolean mask of pixels at/above a temperature threshold.

    Args:
        frame: 2-D array (H, W) in Celsius.
        threshold: Celsius level.

    Returns:
        bool array (H, W), True where frame >= threshold.
    """
    f = np.asarray(frame, dtype=np.float32)
    if f.ndim != 2:
        raise ValueError(f"frame must be 2-D (H, W), got shape {f.shape}")
    return f >= float(threshold)


def compute_high_temperature_area(frame: np.ndarray, threshold: float,
                                  pixel_size_mm: float = None) -> float:
    """
    Area of the high-temperature zone.

    Args:
        frame: 2-D (H, W) Celsius.
        threshold: Celsius level.
        pixel_size_mm: mm per pixel. If None, the area is in pixels;
                       otherwise in mm^2 (pixel_count * pixel_size_mm**2).

    Returns:
        Area as float (pixels, or mm^2).
    """
    mask = compute_high_temperature_mask(frame, threshold)
    count = int(np.count_nonzero(mask))
    if pixel_size_mm is None:
        return float(count)
    return float(count) * float(pixel_size_mm) ** 2


def compute_temperature_gradient(frame: np.ndarray,
                                 pixel_size_mm: float = None) -> dict:
    """
    Spatial temperature-gradient magnitude statistics.

    The gradient is computed with central differences; the magnitude is
    sqrt(gy^2 + gx^2). Units are Celsius/pixel, or Celsius/mm if pixel_size_mm
    is given.

    Args:
        frame: 2-D (H, W) Celsius.
        pixel_size_mm: mm per pixel (None -> per-pixel spacing).

    Returns:
        Dict with mean_gradient, max_gradient, std_gradient (floats).
    """
    f = np.asarray(frame, dtype=np.float64)
    if f.ndim != 2:
        raise ValueError(f"frame must be 2-D (H, W), got shape {f.shape}")
    if f.shape[0] < 2 or f.shape[1] < 2:
        return {"mean_gradient": 0.0, "max_gradient": 0.0, "std_gradient": 0.0}

    spacing = 1.0 if pixel_size_mm is None else float(pixel_size_mm)
    gy, gx = np.gradient(f, spacing, spacing)
    mag = np.sqrt(gy ** 2 + gx ** 2)
    return {
        "mean_gradient": float(mag.mean()),
        "max_gradient": float(mag.max()),
        "std_gradient": float(mag.std()),
    }


def compute_thermal_center(frame: np.ndarray, threshold: float = None) -> tuple:
    """
    Temperature-weighted spatial center (centroid).

    Args:
        frame: 2-D (H, W) Celsius.
        threshold: if given, only pixels >= threshold contribute (weights are
                   set to 0 elsewhere). If None, the whole frame is used.

    Returns:
        (center_x, center_y) in pixel coordinates (x = column, y = row).
        Falls back to the geometric center if total weight is ~0.
    """
    f = np.asarray(frame, dtype=np.float64)
    if f.ndim != 2:
        raise ValueError(f"frame must be 2-D (H, W), got shape {f.shape}")
    h, w = f.shape

    weights = f.copy()
    if threshold is not None:
        weights = np.where(f >= float(threshold), f, 0.0)
    # Guard against negative temperatures dominating the centroid.
    weights = np.clip(weights, 0.0, None)

    total = weights.sum()
    if total <= 1e-12:
        return (float((w - 1) / 2.0), float((h - 1) / 2.0))

    ys, xs = np.indices((h, w))
    cx = float((weights * xs).sum() / total)
    cy = float((weights * ys).sum() / total)
    return (cx, cy)


def compute_center_offset(center_x: float, center_y: float,
                          roi_width: int, roi_height: int,
                          pixel_size_mm: float = None) -> float:
    """
    Euclidean distance from the thermal center to the ROI geometric center.

    Args:
        center_x, center_y: thermal center (pixels).
        roi_width, roi_height: ROI size in pixels (W, H).
        pixel_size_mm: mm per pixel (None -> distance in pixels).

    Returns:
        Offset distance (pixels, or mm).
    """
    gx = (roi_width - 1) / 2.0
    gy = (roi_height - 1) / 2.0
    dist = float(np.hypot(center_x - gx, center_y - gy))
    if pixel_size_mm is None:
        return dist
    return dist * float(pixel_size_mm)


def compute_haz_width(frame: np.ndarray, threshold: float, axis: str = "x",
                      pixel_size_mm: float = None) -> float:
    """
    Heat-affected-zone width: the spatial extent of the region >= threshold.

    Along axis="x" (default) this is the column span (max_col - min_col + 1)
    of pixels at/above the threshold; along axis="y" it is the row span.

    Args:
        frame: 2-D (H, W) Celsius.
        threshold: Celsius level (e.g. haz_threshold).
        axis: "x" (width across columns) or "y" (height across rows).
        pixel_size_mm: mm per pixel (None -> width in pixels).

    Returns:
        Width (pixels, or mm). 0.0 if no pixel is above threshold.
    """
    mask = compute_high_temperature_mask(frame, threshold)
    if not mask.any():
        return 0.0

    if axis == "x":
        cols = np.where(mask.any(axis=0))[0]
        span = int(cols.max() - cols.min() + 1)
    elif axis == "y":
        rows = np.where(mask.any(axis=1))[0]
        span = int(rows.max() - rows.min() + 1)
    else:
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")

    if pixel_size_mm is None:
        return float(span)
    return float(span) * float(pixel_size_mm)


# ---------------------------------------------------------------------------
# Per-frame feature bundle
# ---------------------------------------------------------------------------

def _tf_cfg(config: dict) -> dict:
    return config.get("thermal_field_features", {}) or {}


def extract_frame_features(frame: np.ndarray, config: dict) -> dict:
    """
    Extract spatial features from a single frame.

    Uses thresholds / pixel size from config['thermal_field_features'].

    Returns dict: tmax, tavg, high_temp_area, mean_gradient, max_gradient,
    haz_width, thermal_center_x, thermal_center_y, center_offset.
    """
    f = np.asarray(frame, dtype=np.float32)
    if f.ndim != 2:
        raise ValueError(f"frame must be 2-D (H, W), got shape {f.shape}")

    cfg = _tf_cfg(config)
    high_t = float(cfg.get("high_temp_threshold", 800.0))
    haz_t = float(cfg.get("haz_threshold", 500.0))
    center_t = cfg.get("center_threshold", high_t)
    center_t = None if center_t is None else float(center_t)
    px = cfg.get("pixel_size_mm", None)
    px = None if px is None else float(px)

    h, w = f.shape
    grad = compute_temperature_gradient(f, pixel_size_mm=px)
    cx, cy = compute_thermal_center(f, threshold=center_t)
    offset = compute_center_offset(cx, cy, w, h, pixel_size_mm=px)

    return {
        "tmax": float(f.max()),
        "tavg": float(f.mean()),
        "high_temp_area": compute_high_temperature_area(f, high_t, px),
        "mean_gradient": grad["mean_gradient"],
        "max_gradient": grad["max_gradient"],
        "haz_width": compute_haz_width(f, haz_t, axis="x", pixel_size_mm=px),
        "thermal_center_x": cx,
        "thermal_center_y": cy,
        "center_offset": offset,
    }


# ---------------------------------------------------------------------------
# Experiment-level aggregation (ONE row per experiment)
# ---------------------------------------------------------------------------

def extract_experiment_features(roi_temperature: np.ndarray,
                                frame_rate_fps: float,
                                config: dict) -> dict:
    """
    Aggregate per-frame features into ONE feature row for an experiment.

    Args:
        roi_temperature: N x H x W float32 Celsius ROI matrix.
        frame_rate_fps: camera frame rate (defines the time axis, seconds).
        config: full project config (uses config['thermal_field_features']).

    Returns:
        Dict of aggregated features (floats). Keys:
          peak_temperature, mean_temperature,
          max_high_temp_area, mean_high_temp_area,
          max_gradient, mean_gradient,
          max_cooling_rate, mean_cooling_rate,
          haz_width_max, haz_width_mean,
          temperature_fluctuation,
          center_offset_mean, center_offset_max, center_offset_std,
          dwell_time_above_threshold, temperature_auc.

    Raises:
        ValueError if roi_temperature is not 3-D (N x H x W).
    """
    data = np.asarray(roi_temperature, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(
            f"expected N x H x W (3-D) ROI matrix, got ndim={data.ndim}, "
            f"shape={data.shape}")
    n_frames = data.shape[0]
    if n_frames == 0:
        raise ValueError("ROI matrix has zero frames")
    if frame_rate_fps <= 0:
        raise ValueError(f"frame_rate_fps must be > 0, got {frame_rate_fps}")

    cfg = _tf_cfg(config)
    high_t = float(cfg.get("high_temp_threshold", 800.0))
    smooth_window = int(cfg.get("smooth_window", 1))
    dt = 1.0 / float(frame_rate_fps)

    # Per-frame features.
    per_frame = [extract_frame_features(data[i], config) for i in range(n_frames)]

    def col(key):
        return np.array([pf[key] for pf in per_frame], dtype=np.float64)

    tmax_curve = col("tmax")                 # peak temperature per frame
    area_curve = col("high_temp_area")
    grad_max_curve = col("max_gradient")
    grad_mean_curve = col("mean_gradient")
    haz_curve = col("haz_width")
    offset_curve = col("center_offset")

    # --- Cooling rate from the per-frame peak-temperature curve (dT/dt) ---
    time_axis = np.arange(n_frames, dtype=np.float64) * dt
    sm = _smooth(tmax_curve, smooth_window)
    if n_frames >= 2:
        deriv = np.gradient(sm, time_axis)
    else:
        deriv = np.zeros(1, dtype=np.float64)
    cooling = deriv[deriv < 0.0]
    max_cooling_rate = float(deriv.min()) if deriv.size else 0.0      # most negative
    mean_cooling_rate = float(cooling.mean()) if cooling.size else 0.0

    # --- Temperature fluctuation (sigma_T): std of high-freq residual ---
    residual = tmax_curve - sm
    temperature_fluctuation = float(np.std(residual))

    # --- Dwell time above the high-temperature threshold ---
    dwell = float(np.count_nonzero(tmax_curve >= high_t) * dt)

    # --- Area under the peak-temperature curve (Celsius * seconds) ---
    trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    temperature_auc = float(trapz(tmax_curve, time_axis)) if n_frames >= 2 else 0.0

    return {
        "peak_temperature": float(tmax_curve.max()),
        "mean_temperature": float(np.mean(col("tavg"))),
        "max_high_temp_area": float(area_curve.max()),
        "mean_high_temp_area": float(area_curve.mean()),
        "max_gradient": float(grad_max_curve.max()),
        "mean_gradient": float(grad_mean_curve.mean()),
        "max_cooling_rate": max_cooling_rate,
        "mean_cooling_rate": mean_cooling_rate,
        "haz_width_max": float(haz_curve.max()),
        "haz_width_mean": float(haz_curve.mean()),
        "temperature_fluctuation": temperature_fluctuation,
        "center_offset_mean": float(offset_curve.mean()),
        "center_offset_max": float(offset_curve.max()),
        "center_offset_std": float(offset_curve.std()),
        "dwell_time_above_threshold": dwell,
        "temperature_auc": temperature_auc,
    }


def _smooth(curve: np.ndarray, window_size: int) -> np.ndarray:
    """Centered moving average; window <= 1 returns the curve unchanged."""
    arr = np.asarray(curve, dtype=np.float64).ravel()
    w = int(window_size)
    if w <= 1 or arr.size == 0:
        return arr.copy()
    w = min(w, arr.size)
    pad = w // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(w, dtype=np.float64) / float(w)
    return np.convolve(padded, kernel, mode="same")[pad:pad + arr.size]
