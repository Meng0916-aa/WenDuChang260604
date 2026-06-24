"""
Canonical temperature-state / hot-region mask primitives (shared processing).

This is the SINGLE source of truth for threshold masks, temperature-state
classification, connected-component cleanup and main-region retention used by
the ROI pipeline. It depends ONLY on numpy + scipy.ndimage — never on the
``features`` package — so there is no circular dependency. The legacy
``features.thermal_field_features`` module imports ``build_threshold_mask`` from
here (the reverse direction) instead of carrying its own implementation.

The raw frame is never modified: every function reads the frame and returns
boolean masks / scalars.

Temperature states (camera valid band 300-1800 C; hard saturation 6500 C):
  below_range     : T < valid_min
  valid           : valid_min <= T <= valid_max
  above_range     : valid_max < T < hard_saturation   (NOT a real temperature)
  hard_saturation : T >= hard_saturation               (uint16 ceiling / invalid)
"""

import numpy as np
from scipy import ndimage

# Formal defaults (configs/physical_calibration.yaml -> temperature_calibration).
DEFAULT_VALID_MIN_C = 300.0
DEFAULT_VALID_MAX_C = 1800.0
DEFAULT_HARD_SAT_C = 6500.0
DEFAULT_MIN_COMPONENT_AREA_PX = 9

_STRUCT_8 = ndimage.generate_binary_structure(2, 2)   # 8-connectivity
_STRUCT_4 = ndimage.generate_binary_structure(2, 1)   # 4-connectivity


def _struct(connectivity):
    if int(connectivity) == 8:
        return _STRUCT_8
    if int(connectivity) == 4:
        return _STRUCT_4
    raise ValueError(f"connectivity must be 4 or 8, got {connectivity!r}")


def _check_2d(frame):
    f = np.asarray(frame, dtype=np.float32)
    if f.ndim != 2:
        raise ValueError(f"frame must be 2-D (H, W), got shape {f.shape}")
    return f


# ---------------------------------------------------------------------------
# Threshold mask  (THE canonical implementation; features delegates here)
# ---------------------------------------------------------------------------

def build_threshold_mask(frame, threshold):
    """Boolean mask of pixels at/above ``threshold`` (T >= threshold).

    This is the one and only implementation of the high-temperature threshold
    mask in the project; other modules call it rather than re-deriving it.
    """
    f = _check_2d(frame)
    return f >= float(threshold)


# ---------------------------------------------------------------------------
# Temperature-state classification
# ---------------------------------------------------------------------------

def classify_temperature_state(frame, valid_min=DEFAULT_VALID_MIN_C,
                               valid_max=DEFAULT_VALID_MAX_C,
                               hard_sat=DEFAULT_HARD_SAT_C):
    """Classify each pixel into the four mutually-exclusive states.

    Returns a dict of bool (H, W) masks: below_range / valid / above_range /
    hard_saturation. The raw frame is not modified.
    """
    f = _check_2d(frame)
    vmin, vmax, hard = float(valid_min), float(valid_max), float(hard_sat)
    below = f < vmin
    valid = (f >= vmin) & (f <= vmax)
    above = (f > vmax) & (f < hard)
    hardm = f >= hard
    return {"below_range": below, "valid": valid,
            "above_range": above, "hard_saturation": hardm}


# ---------------------------------------------------------------------------
# Connected components
# ---------------------------------------------------------------------------

def label_components(mask, connectivity=8):
    """Label connected components of a boolean mask. Returns (labels, n)."""
    m = np.asarray(mask, dtype=bool)
    labels, n = ndimage.label(m, structure=_struct(connectivity))
    return labels, int(n)


def remove_small_components(mask, min_area, connectivity=8):
    """Drop connected components smaller than ``min_area`` pixels."""
    m = np.asarray(mask, dtype=bool)
    labels, n = label_components(m, connectivity)
    if n == 0:
        return np.zeros_like(m, dtype=bool)
    counts = np.bincount(labels.ravel())
    keep = np.zeros_like(m, dtype=bool)
    for lab in range(1, n + 1):
        if counts[lab] >= int(min_area):
            keep |= (labels == lab)
    return keep


def handle_hard_saturation_holes(main_mask):
    """Fill interior holes of the main region (mask only; raw never modified).

    An embedded hard-saturation / sub-threshold pixel must not punch a hole in
    the main body's localization mask. ``T >= threshold`` already includes
    hard-saturation pixels (they exceed the threshold numerically), so this is a
    belt-and-braces fill for any remaining interior hole.
    """
    return ndimage.binary_fill_holes(np.asarray(main_mask, dtype=bool))


def handle_connected_above_range(above_range_mask, kept_mask):
    """Split above-range pixels into (connected_to_body, isolated).

    above-range pixels that fall inside a genuine kept component are retained
    for geometry (``connected``); the rest are isolated splatter (``isolated``).
    """
    above = np.asarray(above_range_mask, dtype=bool)
    kept = np.asarray(kept_mask, dtype=bool)
    connected = int(np.count_nonzero(above & kept))
    isolated = int(np.count_nonzero(above) - connected)
    return connected, isolated


def retain_main_component(ge_mask, valid_mask, min_area=DEFAULT_MIN_COMPONENT_AREA_PX,
                          connectivity=8, fill_holes=True):
    """Keep the largest GENUINE component of a threshold mask.

    A component is *genuine* iff it contains at least one in-range *valid* pixel
    that is also at/above the threshold (``valid_mask & ge_mask``); pure
    above-range / hard-saturation components are isolated splatter and excluded.
    The largest genuine component is the main body, hole-filled.

    Returns (main_mask, info) where info has kept_mask (all genuine components),
    n_components, n_genuine, isolated_component_count, main_area_px.
    """
    ge = np.asarray(ge_mask, dtype=bool)
    valid = np.asarray(valid_mask, dtype=bool)
    labels, n = label_components(ge, connectivity)

    main_mask = np.zeros_like(ge, dtype=bool)
    kept_mask = np.zeros_like(ge, dtype=bool)
    n_genuine = 0
    isolated_components = 0

    if n > 0:
        counts = np.bincount(labels.ravel())
        valid_ge = valid & ge
        best_lab, best_area = 0, 0
        for lab in range(1, n + 1):
            area = int(counts[lab])
            if area < int(min_area):
                continue
            comp = (labels == lab)
            if not bool(np.any(valid_ge & comp)):
                isolated_components += 1
                continue
            n_genuine += 1
            kept_mask |= comp
            if area > best_area:
                best_area, best_lab = area, lab
        if best_lab > 0:
            main_mask = (labels == best_lab)
            if fill_holes:
                main_mask = handle_hard_saturation_holes(main_mask)

    return main_mask, {
        "kept_mask": kept_mask,
        "n_components": int(n),
        "n_genuine": int(n_genuine),
        "isolated_component_count": int(isolated_components),
        "main_area_px": int(np.count_nonzero(main_mask)),
    }
