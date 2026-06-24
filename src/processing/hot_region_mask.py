"""
Main melt-pool hot-region localization + temperature-state classification.

Operates on a single 2-D frame (H x W, float32 Celsius). The RAW INPUT IS NEVER
MODIFIED — every function only reads the frame and returns boolean masks or
scalars. This module supports the *ROI-strategy evaluation* phase; it does not
extract quantitative temperature features.

Temperature states (camera valid range 300-1800 C; hard saturation 6500 C):
  below_range     : T < valid_min          (not quantitative; below 700/800 so
                                             never enters the hot-region masks)
  valid           : valid_min <= T <= valid_max
  above_range     : valid_max < T < hard_saturation   (NOT a real temperature)
  hard_saturation : T >= hard_saturation    (uint16 ceiling 6553.5 / invalid)

Spatial-localization rule (geometry only, see docs):
  A 700/800 threshold mask numerically INCLUDES above_range and hard_saturation
  pixels (their values exceed the threshold). Connected components are formed
  (8-connectivity), tiny ones (< ``min_component_area``) dropped, and a component
  is "genuine" iff it contains at least one in-range *valid* pixel >= threshold.
  Components made ONLY of above_range / hard_saturation pixels are isolated
  sentinel hotspots (splatter) and are EXCLUDED. The largest genuine component
  is the MAIN body; it is hole-filled so an embedded hard-saturation core never
  produces a hole. The fill acts on the MASK only — the temperature matrix is
  never altered.

Pure NumPy + scipy.ndimage; no torch / matplotlib and NO dependency on the
``features`` package — the shared primitives live in
``src/processing/temperature_mask.py`` (the single canonical home), so this
module composes them without any circular dependency.
"""

import numpy as np

from processing.temperature_mask import (
    DEFAULT_VALID_MIN_C, DEFAULT_VALID_MAX_C, DEFAULT_HARD_SAT_C,
    DEFAULT_MIN_COMPONENT_AREA_PX,
    build_threshold_mask, classify_temperature_state, label_components,
    remove_small_components, retain_main_component, handle_connected_above_range,
)

# Back-compat alias (older callers/tests import the plural name from here).
classify_temperature_states = classify_temperature_state


def _check_2d(frame):
    f = np.asarray(frame, dtype=np.float32)
    if f.ndim != 2:
        raise ValueError(f"frame must be 2-D (H, W), got shape {f.shape}")
    return f


# ---------------------------------------------------------------------------
# Main hot-region extraction at one threshold (composes processing primitives)
# ---------------------------------------------------------------------------

def extract_main_hot_region(frame, threshold,
                            valid_min=DEFAULT_VALID_MIN_C,
                            valid_max=DEFAULT_VALID_MAX_C,
                            hard_sat=DEFAULT_HARD_SAT_C,
                            min_component_area=DEFAULT_MIN_COMPONENT_AREA_PX,
                            connectivity=8, fill_holes=True):
    """Localize the MAIN hot region at one temperature threshold.

    Args:
        frame: 2-D (H, W) Celsius.
        threshold: envelope (700) or core (800) Celsius level.
        valid_min/valid_max/hard_sat: temperature-state thresholds.
        min_component_area: components smaller than this (px) are dropped.
        connectivity: 8 (default) or 4.
        fill_holes: fill interior holes of the main component (mask only).

    Returns:
        (main_mask, info). ``main_mask`` is a bool (H, W) of the largest genuine
        component (hole-filled). ``info`` carries n_components, n_genuine,
        isolated_component_count, main_area_px, kept_mask (all genuine
        components), and the above_range_connected / above_range_isolated /
        hard_saturation pixel counts.
    """
    f = _check_2d(frame)
    states = classify_temperature_state(f, valid_min, valid_max, hard_sat)
    ge = build_threshold_mask(f, threshold)   # T >= threshold (incl. sentinels)

    main_mask, info = retain_main_component(
        ge, states["valid"], min_area=min_component_area,
        connectivity=connectivity, fill_holes=fill_holes)

    above_connected, above_isolated = handle_connected_above_range(
        states["above_range"], info["kept_mask"])
    info["above_range_connected_count"] = above_connected
    info["above_range_isolated_count"] = above_isolated
    info["hard_saturation_count"] = int(np.count_nonzero(states["hard_saturation"]))
    return main_mask, info


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def mask_bbox(mask):
    """Bounding box of True pixels as (top, left, bottom_excl, right_excl).

    Half-open in Python convention: rows [top, bottom_excl), cols [left,
    right_excl). Returns None when the mask is empty.
    """
    m = np.asarray(mask, dtype=bool)
    rows = np.where(m.any(axis=1))[0]
    cols = np.where(m.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return None
    return (int(rows.min()), int(cols.min()),
            int(rows.max()) + 1, int(cols.max()) + 1)


def mask_centroid(mask):
    """Geometric centroid (cx, cy) = (mean col, mean row), or None if empty."""
    m = np.asarray(mask, dtype=bool)
    ys, xs = np.nonzero(m)
    if xs.size == 0:
        return None
    return (float(xs.mean()), float(ys.mean()))


def bbox_size(bbox):
    """(width_px, height_px) for a (top, left, bottom_excl, right_excl) bbox."""
    if bbox is None:
        return (0, 0)
    top, left, bottom_excl, right_excl = bbox
    return (int(right_excl - left), int(bottom_excl - top))


def bbox_touches_frame(bbox, height, width):
    """Which image edges the bbox touches: dict top/bottom/left/right -> bool."""
    if bbox is None:
        return {"top": False, "bottom": False, "left": False, "right": False}
    top, left, bottom_excl, right_excl = bbox
    return {
        "top": top <= 0,
        "bottom": bottom_excl >= int(height),
        "left": left <= 0,
        "right": right_excl >= int(width),
    }


# ---------------------------------------------------------------------------
# One-call per-frame analysis
# ---------------------------------------------------------------------------

def analyze_frame(frame, envelope_threshold=700.0, core_threshold=800.0,
                  valid_min=DEFAULT_VALID_MIN_C, valid_max=DEFAULT_VALID_MAX_C,
                  hard_sat=DEFAULT_HARD_SAT_C,
                  min_component_area=DEFAULT_MIN_COMPONENT_AREA_PX,
                  connectivity=8):
    """Analyze one frame: 700 envelope + 800 core main regions, bboxes, centroid.

    The centroid is the geometric centroid of the 700 envelope main body (used
    later to center the tracking window). Returns a dict of masks + scalars.
    """
    f = _check_2d(frame)
    env_mask, env_info = extract_main_hot_region(
        f, envelope_threshold, valid_min, valid_max, hard_sat,
        min_component_area, connectivity)
    core_mask, core_info = extract_main_hot_region(
        f, core_threshold, valid_min, valid_max, hard_sat,
        min_component_area, connectivity)

    bbox700 = mask_bbox(env_mask)
    bbox800 = mask_bbox(core_mask)
    centroid = mask_centroid(env_mask)
    return {
        "envelope_mask": env_mask,
        "core_mask": core_mask,
        "bbox700": bbox700,
        "bbox800": bbox800,
        "centroid": centroid,
        "area_700_px": int(np.count_nonzero(env_mask)),
        "area_800_px": int(np.count_nonzero(core_mask)),
        "above_range_connected_count": env_info["above_range_connected_count"],
        "above_range_isolated_count": env_info["above_range_isolated_count"],
        "hard_saturation_count": env_info["hard_saturation_count"],
        "has_envelope": bbox700 is not None,
        "has_core": bbox800 is not None,
    }
