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
    """Fixed window size covering the p-th percentile bbox plus a margin/side.

    The same width/height are used for ALL tracks and frames; only the window
    center moves. Returns (win_height_px, win_width_px), clamped to the frame.
    """
    pw = size_percentiles(bbox_widths, (percentile,))[int(percentile)]
    ph = size_percentiles(bbox_heights, (percentile,))[int(percentile)]
    win_w = _ceil_to(pw + 2 * int(edge_safety_margin), round_to)
    win_h = _ceil_to(ph + 2 * int(edge_safety_margin), round_to)
    win_w = min(int(win_w), int(width))
    win_h = min(int(win_h), int(height))
    return int(win_h), int(win_w)


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
