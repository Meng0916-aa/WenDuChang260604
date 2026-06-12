"""
Local (cross-section) thermal-field feature extraction.

The section-level ML main line treats each CROSS-SECTION POSITION as one ML
sample (not a frame, and not a whole experiment). For a given section located
at ``section_position_mm`` along a cladding track of length
``travel_distance_mm``, this module:

  1. Maps the section position to a CENTER FRAME of the ROI temperature matrix:
         frame_center = round(section_position_mm / travel_distance_mm * (N-1))
  2. Takes a symmetric frame window of half-width ``window_half_frames``:
         frame_start = max(0,   frame_center - window_half_frames)
         frame_end   = min(N-1, frame_center + window_half_frames)
  3. Extracts LOCAL thermal-field descriptors from just that window.

The descriptors are computed by REUSING
``features.thermal_field_features.extract_experiment_features`` on the windowed
sub-array, so thresholds (high_temp / haz), ``pixel_size_mm`` and
``frame_rate_fps`` semantics stay identical to the experiment-level pipeline.
Keys are renamed with a ``local_`` prefix.

Pure NumPy (+ the reused feature module); no torch / matplotlib, so every
function is unit-testable in isolation.
"""

import warnings

import numpy as np

from features.thermal_field_features import extract_experiment_features

# Below this many frames in the local window, cooling-rate / fluctuation
# descriptors get unreliable -> warn (but still compute what we can).
MIN_WINDOW_FRAMES = 3

# Map experiment-level aggregate keys -> local_* keys for the section sample.
_FEATURE_KEY_MAP = {
    "peak_temperature": "local_peak_temperature",
    "mean_temperature": "local_mean_temperature",
    "temperature_fluctuation": "local_temperature_fluctuation",
    "mean_high_temp_area": "local_high_temp_area_mean",
    "max_high_temp_area": "local_high_temp_area_max",
    "haz_width_mean": "local_haz_width_mean",
    "haz_width_max": "local_haz_width_max",
    "mean_gradient": "local_gradient_mean",
    "max_gradient": "local_gradient_max",
    "mean_cooling_rate": "local_cooling_rate_mean",
    "center_offset_mean": "local_center_offset_mean",
    "center_offset_max": "local_center_offset_max",
    "center_offset_std": "local_center_offset_std",
    "dwell_time_above_threshold": "local_dwell_time",
    "temperature_auc": "local_temperature_auc",
}

# The full set of local_* feature keys this module produces (excluding the
# frame bookkeeping fields). Useful for column ordering / validation.
LOCAL_FEATURE_KEYS = list(_FEATURE_KEY_MAP.values()) + ["local_cooling_rate_max_abs"]


def compute_frame_window(section_position_mm, travel_distance_mm, n_frames,
                         window_half_frames):
    """Map a section position to a (center, start, end) frame window.

    Args:
        section_position_mm: distance of the section from the track start (mm),
            in [0, travel_distance_mm].
        travel_distance_mm: effective track length (mm), must be > 0.
        n_frames: number of frames N in the ROI matrix, must be >= 1.
        window_half_frames: symmetric half-window in frames, must be >= 0.

    Returns:
        (frame_center, frame_start, frame_end) integer frame indices
        (inclusive), all within [0, n_frames-1].

    Raises:
        ValueError if travel_distance_mm <= 0, n_frames < 1, half < 0, or the
        section position is outside [0, travel_distance_mm].
    """
    travel = float(travel_distance_mm)
    pos = float(section_position_mm)
    half = int(window_half_frames)
    n = int(n_frames)

    if travel <= 0.0:
        raise ValueError(f"travel_distance_mm must be > 0, got {travel}")
    if n < 1:
        raise ValueError(f"n_frames must be >= 1, got {n}")
    if half < 0:
        raise ValueError(f"window_half_frames must be >= 0, got {half}")
    if pos < 0.0 or pos > travel:
        raise ValueError(
            f"section_position_mm={pos} is outside the track "
            f"[0, {travel}] mm — check section_position_mm / travel_distance_mm")

    frac = pos / travel
    frame_center = int(round(frac * (n - 1)))
    frame_center = max(0, min(n - 1, frame_center))
    frame_start = max(0, frame_center - half)
    frame_end = min(n - 1, frame_center + half)
    return frame_center, frame_start, frame_end


def extract_local_section_features(roi_temperature, section_position_mm,
                                   travel_distance_mm, window_half_frames,
                                   frame_rate_fps, config):
    """Extract LOCAL thermal-field features for one cross-section sample.

    Args:
        roi_temperature: N x H x W float32 Celsius ROI matrix.
        section_position_mm: section position along the track (mm).
        travel_distance_mm: effective track length (mm).
        window_half_frames: symmetric half-window (frames).
        frame_rate_fps: effective frame rate for this track (fps, > 0).
        config: full project config (uses config['thermal_field_features']).

    Returns:
        Dict of local_* features plus frame_center/frame_start/frame_end/
        window_frame_count (ints).

    Raises:
        ValueError if roi_temperature is not 3-D or the section position is out
        of range (see compute_frame_window).
    """
    data = np.asarray(roi_temperature, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(
            f"expected N x H x W (3-D) ROI matrix, got ndim={data.ndim}, "
            f"shape={data.shape}")
    n_frames = data.shape[0]

    frame_center, frame_start, frame_end = compute_frame_window(
        section_position_mm, travel_distance_mm, n_frames, window_half_frames)

    window = data[frame_start:frame_end + 1]
    win_count = int(window.shape[0])
    if win_count < MIN_WINDOW_FRAMES:
        warnings.warn(
            f"local window has only {win_count} frame(s) "
            f"(< {MIN_WINDOW_FRAMES}) for section_position_mm="
            f"{section_position_mm}; cooling-rate / fluctuation features may be "
            f"unreliable. Computing the available features anyway.",
            stacklevel=2)

    agg = extract_experiment_features(window, frame_rate_fps=frame_rate_fps,
                                      config=config)

    feats = {local_key: float(agg[src_key])
             for src_key, local_key in _FEATURE_KEY_MAP.items()}
    # Cooling rate magnitude: agg['max_cooling_rate'] is the most-negative
    # dT/dt (strongest cooling); report its absolute value.
    feats["local_cooling_rate_max_abs"] = float(abs(agg["max_cooling_rate"]))

    feats["frame_center"] = int(frame_center)
    feats["frame_start"] = int(frame_start)
    feats["frame_end"] = int(frame_end)
    feats["window_frame_count"] = win_count
    return feats
