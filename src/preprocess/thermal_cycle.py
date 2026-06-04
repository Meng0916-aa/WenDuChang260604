"""
Thermal cycle curve extraction.

From a temperature field (N, H, W), extract time-series curves:

1. Tmax(t)       — maximum temperature in each frame.
2. center_avg(t) — average temperature in a circular region around the
                    spatial center.
3. hot_zone_avg(t) — average temperature over all pixels above a threshold.

All output is float32 Celsius.
"""

import numpy as np


def extract_tmax(data: np.ndarray) -> np.ndarray:
    """
    Extract the maximum temperature per frame.

    Args:
        data: float32 array of shape (N, H, W).

    Returns:
        1D array of shape (N,) — Tmax for each frame.
    """
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D array (N, H, W), got shape {data.shape}")
    return np.max(data, axis=(1, 2)).astype(np.float32)


def extract_center_average(data: np.ndarray, radius: int = 5) -> np.ndarray:
    """
    Average temperature within a circular ROI around the frame center.

    Args:
        data: float32 array of shape (N, H, W).
        radius: radius of the circular region in pixels.

    Returns:
        1D array of shape (N,).
    """
    data = np.asarray(data, dtype=np.float32)
    n_frames, h, w = data.shape
    cy, cx = h // 2, w // 2

    # Build a circular mask once
    yy, xx = np.ogrid[:h, :w]
    mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius ** 2

    result = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        result[i] = data[i, mask].mean()

    return result


def extract_hot_zone_average(data: np.ndarray,
                              threshold: float = 800.0) -> np.ndarray:
    """
    Average temperature of all pixels above a threshold in each frame.

    Args:
        data: float32 array of shape (N, H, W) in Celsius.
        threshold: temperature threshold in Celsius.

    Returns:
        1D array of shape (N,). Frames with no pixels above threshold
        return 0.0.
    """
    data = np.asarray(data, dtype=np.float32)
    n_frames = data.shape[0]

    result = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        frame = data[i]
        hot = frame[frame >= threshold]
        if hot.size > 0:
            result[i] = hot.mean()

    return result


def extract_all_cycles(data: np.ndarray,
                       center_radius: int = 5,
                       hot_threshold: float = 800.0) -> dict:
    """
    Extract all three thermal cycle curves in one call.

    Args:
        data: float32 array of shape (N, H, W) in Celsius.
        center_radius: radius for center-average extraction.
        hot_threshold: Celsius threshold for hot-zone extraction.

    Returns:
        Dict with keys:
          'tmax'            — array (N,)
          'center_average'   — array (N,)
          'hot_zone_average' — array (N,)
    """
    return {
        "tmax": extract_tmax(data),
        "center_average": extract_center_average(data, radius=center_radius),
        "hot_zone_average": extract_hot_zone_average(data, threshold=hot_threshold),
    }


def save_thermal_cycles(cycles: dict, output_path: str) -> None:
    """
    Save thermal cycle curves as a CSV file.

    Columns: frame, tmax, center_average, hot_zone_average

    Args:
        cycles: dict from extract_all_cycles().
        output_path: path to output .csv file.
    """
    import os
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    n = len(cycles["tmax"])
    header = "frame,tmax,center_average,hot_zone_average"
    rows = [
        f"{i},{cycles['tmax'][i]:.4f},"
        f"{cycles['center_average'][i]:.4f},"
        f"{cycles['hot_zone_average'][i]:.4f}"
        for i in range(n)
    ]
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        f.write("\n".join(rows) + "\n")
