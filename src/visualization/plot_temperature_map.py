"""
Heatmap of a single temperature-field frame.

matplotlib only (no seaborn). Uses the default colormap unless the caller
passes one. English labels for paper use.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Re-export the shared saver.
from visualization.plot_curves import save_figure  # noqa: F401


def plot_temperature_frame(frame: np.ndarray,
                           title: str = "Temperature Field (single frame)",
                           cmap: str = "jet", figsize=(6, 5)):
    """
    Render one (H, W) temperature frame as a heatmap with a colour bar.

    Args:
        frame: 2-D array (H, W) in Celsius. If a 3-D (N, H, W) array is given,
               the first frame is used.
        title: figure title (English).
        cmap: matplotlib colormap name.

    Returns:
        The matplotlib Figure.
    """
    frame = np.asarray(frame, dtype=np.float32)
    if frame.ndim == 3:
        frame = frame[0]
    if frame.ndim != 2:
        raise ValueError(f"Expected (H, W) or (N, H, W), got {frame.shape}")

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(frame, cmap=cmap, origin="upper", aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Temperature (deg C)")
    ax.set_title(title)
    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    fig.tight_layout()
    return fig
