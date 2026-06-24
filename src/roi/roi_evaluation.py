"""
ROI-strategy evaluation geometry.

Pure, unit-testable helpers for the unified-ROI evaluation phase: effective-frame
bookkeeping (frames[1:]), bounding-box unions, ROI coverage / safety-margin math,
fixed global-ROI candidate construction (margin + rounding + 512x640 clamp), and
fixed-size tracking-window sizing / coverage.

Rectangle convention everywhere: ``(top, left, bottom_excl, right_excl)`` —
half-open in Python style, so rows ``[top, bottom_excl)`` and cols
``[left, right_excl)``. This module never loads or modifies any matrix; it only
operates on masks, rectangles and scalar lists handed to it.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Effective frames (frames[1:]; the first frame is the camera startup frame)
# ---------------------------------------------------------------------------

def effective_frame_indices(n_frames):
    """Indices of the effective frames = frames[1:] (startup frame excluded)."""
    n = int(n_frames)
    if n <= 1:
        return []
    return list(range(1, n))


def effective_frame_info(n_frames):
    """Bookkeeping dict for one track's effective-frame split."""
    n = int(n_frames)
    eff = max(0, n - 1)
    return {
        "original_frame_count": n,
        "effective_frame_count": eff,
        "excluded_frame_count": 1 if n >= 1 else 0,
        "effective_start_index_0_based": 1,
    }


# ---------------------------------------------------------------------------
# Bounding-box algebra  (top, left, bottom_excl, right_excl)
# ---------------------------------------------------------------------------

def bbox_union(bboxes):
    """Union of bounding boxes (None entries ignored). None if all empty."""
    boxes = [b for b in bboxes if b is not None]
    if not boxes:
        return None
    tops, lefts, bottoms, rights = zip(*boxes)
    return (int(min(tops)), int(min(lefts)), int(max(bottoms)), int(max(rights)))


def bbox_width_height(bbox):
    """(width_px, height_px) for a half-open bbox; (0, 0) if None."""
    if bbox is None:
        return (0, 0)
    top, left, bottom_excl, right_excl = bbox
    return (int(right_excl - left), int(bottom_excl - top))


def rect_touches_frame(rect, height, width):
    """Which image edges a rect touches: dict top/bottom/left/right -> bool."""
    if rect is None:
        return {"top": False, "bottom": False, "left": False, "right": False}
    top, left, bottom_excl, right_excl = rect
    return {
        "top": top <= 0,
        "bottom": bottom_excl >= int(height),
        "left": left <= 0,
        "right": right_excl >= int(width),
    }


# ---------------------------------------------------------------------------
# Coverage and safety margin of a candidate ROI
# ---------------------------------------------------------------------------

def coverage_fraction(mask, rect):
    """Fraction of True mask pixels that fall inside ``rect``.

    Returns NaN when the mask is empty (no hot pixels to cover). ``rect`` is
    half-open (top, left, bottom_excl, right_excl).
    """
    m = np.asarray(mask, dtype=bool)
    total = int(m.sum())
    if total == 0:
        return float("nan")
    top, left, bottom_excl, right_excl = rect
    top = max(0, int(top)); left = max(0, int(left))
    bottom_excl = min(m.shape[0], int(bottom_excl))
    right_excl = min(m.shape[1], int(right_excl))
    if bottom_excl <= top or right_excl <= left:
        return 0.0
    inside = int(m[top:bottom_excl, left:right_excl].sum())
    return inside / total


def min_edge_distance(bbox, rect):
    """Minimum pixel gap between a hot bbox and the inner ROI boundary.

    Positive = the bbox sits at least this many pixels inside every ROI edge.
    Zero = touching; negative = the bbox extends beyond the ROI (not covered).
    Returns None when ``bbox`` is None (no hot region).
    """
    if bbox is None:
        return None
    btop, bleft, bbot_excl, bright_excl = bbox
    rtop, rleft, rbot_excl, rright_excl = rect
    # inclusive maxima
    b_max_row, b_max_col = bbot_excl - 1, bright_excl - 1
    r_max_row, r_max_col = rbot_excl - 1, rright_excl - 1
    d_top = btop - rtop
    d_left = bleft - rleft
    d_bottom = r_max_row - b_max_row
    d_right = r_max_col - b_max_col
    return int(min(d_top, d_left, d_bottom, d_right))


# ---------------------------------------------------------------------------
# Fixed global-ROI candidate construction
# ---------------------------------------------------------------------------

def clamp_rect(rect, height, width):
    """Clamp a rect into [0, height] x [0, width] (half-open)."""
    top, left, bottom_excl, right_excl = rect
    top = max(0, min(int(top), int(height)))
    left = max(0, min(int(left), int(width)))
    bottom_excl = max(0, min(int(bottom_excl), int(height)))
    right_excl = max(0, min(int(right_excl), int(width)))
    return (top, left, bottom_excl, right_excl)


def expand_rect(rect, margin, height, width):
    """Grow a rect by ``margin`` px on every side, clamped to the frame."""
    top, left, bottom_excl, right_excl = rect
    m = int(margin)
    return clamp_rect((top - m, left - m, bottom_excl + m, right_excl + m),
                      height, width)


def round_rect_up(rect, multiple, height, width):
    """Round a rect's width/height UP to a multiple, only GROWING the box.

    Grows toward bottom/right first; if the frame edge is reached, grows toward
    top/left. Never shrinks (coverage can only increase) and never exceeds the
    frame. If the full frame dimension is not a multiple, the box may stay below
    the rounded size after clamping (it is then frame-limited).
    """
    mult = int(multiple)
    if mult <= 1:
        return clamp_rect(rect, height, width)
    top, left, bottom_excl, right_excl = clamp_rect(rect, height, width)
    h, w = bottom_excl - top, right_excl - left

    def _grow(lo, hi, size, target, limit):
        # grow [lo, hi) up to `target`, first extending hi, then lo, within [0, limit]
        need = target - size
        if need <= 0:
            return lo, hi
        grow_hi = min(need, limit - hi)
        hi += grow_hi
        need -= grow_hi
        if need > 0:
            grow_lo = min(need, lo)
            lo -= grow_lo
        return lo, hi

    target_h = ((h + mult - 1) // mult) * mult
    target_w = ((w + mult - 1) // mult) * mult
    top, bottom_excl = _grow(top, bottom_excl, h, target_h, int(height))
    left, right_excl = _grow(left, right_excl, w, target_w, int(width))
    return (top, left, bottom_excl, right_excl)


def build_global_roi_candidate(global_bbox700, margin, height, width,
                               round_to=8):
    """Build the fixed global-ROI candidate from the 57-track 700 union bbox.

    Returns a dict with the candidate rect plus derived size / area metrics.
    """
    if global_bbox700 is None:
        return None
    expanded = expand_rect(global_bbox700, margin, height, width)
    rounded = round_rect_up(expanded, round_to, height, width)
    top, left, bottom_excl, right_excl = rounded
    cw, ch = right_excl - left, bottom_excl - top
    area_px = int(cw * ch)
    full = int(height) * int(width)
    return {
        "candidate_top": int(top),
        "candidate_left": int(left),
        "candidate_bottom_exclusive": int(bottom_excl),
        "candidate_right_exclusive": int(right_excl),
        "candidate_height": int(ch),
        "candidate_width": int(cw),
        "candidate_area_px": area_px,
        "candidate_fraction_of_full_frame": area_px / full if full else float("nan"),
        "rect": (int(top), int(left), int(bottom_excl), int(right_excl)),
    }


# ---------------------------------------------------------------------------
# Fixed-size tracking window
# ---------------------------------------------------------------------------

def size_percentiles(values, percentiles=(95, 99)):
    """Percentiles of a list of sizes; empty -> 0 for each requested pct."""
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if arr.size == 0:
        return {int(p): 0.0 for p in percentiles}
    return {int(p): float(np.percentile(arr, p)) for p in percentiles}


def _ceil_to(value, multiple):
    v = int(np.ceil(value))
    m = int(multiple)
    if m <= 1:
        return v
    return ((v + m - 1) // m) * m


def tracking_window_size(bbox_widths, bbox_heights, percentile=99,
                         edge_safety_margin=10, round_to=8,
                         height=512, width=640):
    """LEGACY (percentile-of-bbox-size) window sizing — kept for comparison only.

    This sizes from the p-th percentile bbox dimension plus a per-side margin and
    centers on the temperature centroid. Because the centroid is NOT the bbox
    geometric center, an asymmetric main region can still be clipped even when
    the window is larger than the bbox. Use ``tracking_window_from_extents`` for
    the formal (100%-coverage) window. Returns (win_height_px, win_width_px).
    """
    pw = size_percentiles(bbox_widths, (percentile,))[int(percentile)]
    ph = size_percentiles(bbox_heights, (percentile,))[int(percentile)]
    win_w = _ceil_to(pw + 2 * int(edge_safety_margin), round_to)
    win_h = _ceil_to(ph + 2 * int(edge_safety_margin), round_to)
    win_w = min(int(win_w), int(width))
    win_h = min(int(win_h), int(height))
    return int(win_h), int(win_w)


# ---------------------------------------------------------------------------
# Extent-based tracking window (formal; guarantees containment about the centroid)
# ---------------------------------------------------------------------------

def compute_extents(bbox, centroid):
    """Four-direction extents from the (float) centroid to the bbox edges.

    bbox is half-open (top, left, bottom_excl, right_excl); centroid is (cx, cy)
    in pixels. Returns a dict with float ``left``/``right``/``top``/``bottom``
    extents, or None when bbox/centroid is missing. (Per the spec:
    right = right_excl - cx, bottom = bottom_excl - cy.)
    """
    if bbox is None or centroid is None:
        return None
    top, left, bottom_excl, right_excl = bbox
    cx, cy = centroid
    return {
        "left": float(cx - left),
        "right": float(right_excl - cx),
        "top": float(cy - top),
        "bottom": float(bottom_excl - cy),
    }


def extent_stats(extents):
    """Per-direction min / p95 / p99 / max over a list of extent dicts."""
    out = {}
    for key in ("left", "right", "top", "bottom"):
        vals = np.asarray([e[key] for e in extents if e is not None], dtype=np.float64)
        if vals.size == 0:
            out[key] = {"min": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
        else:
            out[key] = {
                "min": float(vals.min()),
                "p95": float(np.percentile(vals, 95)),
                "p99": float(np.percentile(vals, 99)),
                "max": float(vals.max()),
            }
    return out


def tracking_window_from_extents(frame_bboxes, frame_centroids, round_to=8,
                                 height=512, width=640):
    """Size a fixed window that contains EVERY frame's main bbox about its centroid.

    Placement (``place_window``) centers an even-sized window on round(centroid).
    For containment with that placement, each half must cover the worst-case
    placement-aware need: ``round(cx) - left`` and ``right_excl - round(cx)``
    (and likewise vertically). The window is the same fixed size for all frames;
    only the center moves (no scaling, no rotation). Returns a dict with
    win_height_px/win_width_px, the required half sizes, the float extent stats,
    and ``fits_in_frame`` (False if 100% coverage would exceed the frame).
    """
    half_w_need = 0
    half_h_need = 0
    exts = []
    for bbox, cen in zip(frame_bboxes, frame_centroids):
        e = compute_extents(bbox, cen)
        if e is None:
            continue
        exts.append(e)
        top, left, bottom_excl, right_excl = bbox
        cx, cy = cen
        rcx, rcy = int(round(cx)), int(round(cy))
        half_w_need = max(half_w_need, rcx - left, right_excl - rcx)
        half_h_need = max(half_h_need, rcy - top, bottom_excl - rcy)

    win_w_ideal = _ceil_to(2 * half_w_need, round_to)
    win_h_ideal = _ceil_to(2 * half_h_need, round_to)
    win_w = min(int(win_w_ideal), int(width))
    win_h = min(int(win_h_ideal), int(height))
    fits = (win_w_ideal <= int(width)) and (win_h_ideal <= int(height))
    return {
        "win_height_px": int(win_h),
        "win_width_px": int(win_w),
        "required_half_width_px": int(half_w_need),
        "required_half_height_px": int(half_h_need),
        "fits_in_frame": bool(fits),
        "extent_stats": extent_stats(exts),
    }


def classify_tracking_window(win_h, win_w, cand_area_px, fits_in_frame,
                             full_coverage, large_fraction=0.85):
    """Decide whether a 100%-coverage tracking window is recommendable.

    Returns one of:
      ``auxiliary_only``            window doesn't fit the frame, or its area is
                                    >= ``large_fraction`` of the global ROI (it has
                                    lost local-analysis value);
      ``rejected_clips_main_region``window still clips the cleaned main region;
      ``candidate_full_coverage``   fits, small enough, and covers 100%.
    """
    win_area = int(win_h) * int(win_w)
    too_large = (not fits_in_frame) or (
        bool(cand_area_px) and win_area >= float(large_fraction) * float(cand_area_px))
    if too_large:
        return "auxiliary_only"
    if not full_coverage:
        return "rejected_clips_main_region"
    return "candidate_full_coverage"


def place_window(center_xy, win_h, win_w, height, width):
    """Place a fixed-size window centered on ``center_xy`` = (cx, cy), shifted to
    stay fully inside the frame. Returns (rect, edge_adjusted: bool).

    No scaling and no rotation — only a translation of a fixed-size box.
    """
    cx, cy = center_xy
    top0 = int(round(cy)) - win_h // 2
    left0 = int(round(cx)) - win_w // 2
    top = max(0, min(top0, int(height) - win_h))
    left = max(0, min(left0, int(width) - win_w))
    edge_adjusted = bool(top != top0 or left != left0)
    return (top, left, top + win_h, left + win_w), edge_adjusted


def bbox_inside(inner, outer):
    """True iff bbox ``inner`` is fully contained in rect ``outer`` (half-open)."""
    if inner is None:
        return True
    itop, ileft, ibot, iright = inner
    otop, oleft, obot, oright = outer
    return (itop >= otop and ileft >= oleft and ibot <= obot and iright <= oright)


def evaluate_tracking_window(frame_bboxes, frame_centroids, win_h, win_w,
                             height=512, width=640):
    """Coverage of a fixed-size moving window over many (bbox, centroid) frames.

    For each frame the window is centered on the centroid (then shifted to stay
    inside the frame) and the frame's 700 bbox must fit inside it. Frames with no
    bbox/centroid are skipped. Returns counts + coverage_rate.
    """
    total = 0
    fully = 0
    clipped = 0
    edge_adjusted = 0
    for bbox, centroid in zip(frame_bboxes, frame_centroids):
        if bbox is None or centroid is None:
            continue
        total += 1
        rect, adj = place_window(centroid, win_h, win_w, height, width)
        if adj:
            edge_adjusted += 1
        if bbox_inside(bbox, rect):
            fully += 1
        else:
            clipped += 1
    rate = (fully / total) if total else float("nan")
    return {
        "window_height_px": int(win_h),
        "window_width_px": int(win_w),
        "total_frames": int(total),
        "fully_covered_frame_count": int(fully),
        "clipped_frame_count": int(clipped),
        "edge_adjusted_frame_count": int(edge_adjusted),
        "coverage_rate": rate,
    }
