"""
Region of Interest (ROI) extraction.

Crops a temperature field array of shape (N, H, W) to a rectangular
sub-region defined by pixel coordinates.
"""

import numpy as np


def crop_roi(data: np.ndarray, bounds: tuple) -> np.ndarray:
    """
    Crop a temperature field to a rectangular ROI.

    Args:
        data: float32 array of shape (N, H, W).
        bounds: (x1, y1, x2, y2) in pixel coordinates.
                x1, y1 = top-left corner; x2, y2 = bottom-right corner (inclusive).

    Returns:
        Cropped array of shape (N, y2-y1+1, x2-x1+1).
    """
    x1, y1, x2, y2 = bounds
    data = np.asarray(data, dtype=np.float32)

    if data.ndim != 3:
        raise ValueError(f"Expected 3D array (N, H, W), got shape {data.shape}")

    # bounds are (x, y) = (column, row)
    roi = data[:, y1:y2 + 1, x1:x2 + 1].copy()
    return roi


def get_center_roi(data: np.ndarray, half_width: int, half_height: int) -> np.ndarray:
    """
    Crop a centered ROI around the spatial center of each frame.

    Args:
        data: float32 array of shape (N, H, W).
        half_width: half the desired ROI width in pixels.
        half_height: half the desired ROI height in pixels.

    Returns:
        Cropped array centered on (H/2, W/2).
    """
    _, h, w = data.shape
    cx, cy = w // 2, h // 2
    x1 = max(0, cx - half_width)
    x2 = min(w - 1, cx + half_width)
    y1 = max(0, cy - half_height)
    y2 = min(h - 1, cy + half_height)
    return crop_roi(data, (x1, y1, x2, y2))


def validate_bounds(data: np.ndarray, bounds: tuple) -> bool:
    """
    Check that ROI bounds are within the data's spatial dimensions.

    Args:
        data: array of shape (N, H, W).
        bounds: (x1, y1, x2, y2).

    Returns:
        True if bounds are valid, False otherwise.
    """
    _, h, w = data.shape
    x1, y1, x2, y2 = bounds
    return 0 <= x1 < x2 < w and 0 <= y1 < y2 < h
