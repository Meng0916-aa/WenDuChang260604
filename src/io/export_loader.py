"""
Export loader for temperature field data.

Reads temperature matrices from .npy, .csv, and .h5 export formats
and returns them as float32 Celsius arrays of shape (N, H, W).

Also provides the raw_value → Celsius conversion function.
"""

import os
import numpy as np


# ---------------------------------------------------------------------------
# Temperature scale conversion
# ---------------------------------------------------------------------------

def raw_to_celsius(raw: np.ndarray, scale: float = 0.1) -> np.ndarray:
    """
    Convert raw digital counts to Celsius.

        T_celsius = raw_value * scale

    Default scale = 0.1 (i.e. raw_value / 10.0).

    Args:
        raw: array of raw digital count values.
        scale: multiplicative scale factor.

    Returns:
        Temperature array in degrees Celsius (float32).
    """
    return (np.asarray(raw, dtype=np.float32) * scale).astype(np.float32)


# ---------------------------------------------------------------------------
# File loaders
# ---------------------------------------------------------------------------

def load_npy(filepath: str, as_celsius: bool = True, scale: float = 0.1) -> np.ndarray:
    """
    Load a temperature matrix from a .npy file.

    Args:
        filepath: path to .npy file.
        as_celsius: if True and data range looks like raw counts,
                    apply raw_to_celsius conversion.
        scale: temperature scale factor (used when as_celsius=True).

    Returns:
        float32 array of shape (N, H, W).
    """
    data = np.load(filepath).astype(np.float32)

    if data.ndim == 2:
        # Single frame → expand to (1, H, W)
        data = data[np.newaxis, ...]

    if data.ndim != 3:
        raise ValueError(
            f"Expected 2D or 3D temperature data, got shape {data.shape}"
        )

    if as_celsius and data.max() > 100.0:
        # Heuristic: if max > 100, likely still raw counts
        data = raw_to_celsius(data, scale=scale)

    return data


def load_csv(filepath: str, delimiter: str = ",", as_celsius: bool = True,
             scale: float = 0.1) -> np.ndarray:
    """
    Load a temperature matrix from a .csv file.

    The CSV is expected to contain one frame per row (flattened H×W),
    or a single frame as a 2D matrix.

    Args:
        filepath: path to .csv file.
        delimiter: column delimiter.
        as_celsius: apply raw_to_celsius if data appears to be raw counts.
        scale: temperature scale factor.

    Returns:
        float32 array of shape (N, H, W).
    """
    data = np.loadtxt(filepath, delimiter=delimiter, dtype=np.float32)

    if data.ndim == 2:
        # Assume shape (N, H*W) or (H, W)
        if data.shape[1] > 100:
            # Heuristic: columns > 100 → flat frames (N, H*W)
            # Try to infer square-ish spatial dims
            n_frames = data.shape[0]
            h = w = int(np.sqrt(data.shape[1]))
            if h * w != data.shape[1]:
                raise ValueError(
                    f"Cannot infer spatial dims from {data.shape[1]} columns. "
                    "Use load_csv_with_shape()."
                )
            data = data.reshape(n_frames, h, w)
        # else: already (H, W) → treated as single frame below

    if data.ndim == 2:
        data = data[np.newaxis, ...]

    if as_celsius and data.max() > 100.0:
        data = raw_to_celsius(data, scale=scale)

    return data


def load_h5(filepath: str, dataset_key: str = "temperature",
            as_celsius: bool = True, scale: float = 0.1) -> np.ndarray:
    """
    Load a temperature matrix from an HDF5 (.h5) file.

    Args:
        filepath: path to .h5 file.
        dataset_key: HDF5 dataset name.
        as_celsius: apply raw_to_celsius if data appears to be raw counts.
        scale: temperature scale factor.

    Returns:
        float32 array of shape (N, H, W).
    """
    import h5py
    with h5py.File(filepath, "r") as f:
        data = np.asarray(f[dataset_key], dtype=np.float32)

    if data.ndim == 2:
        data = data[np.newaxis, ...]
    if data.ndim != 3:
        raise ValueError(
            f"Expected 2D or 3D temperature data from HDF5, got shape {data.shape}"
        )

    if as_celsius and data.max() > 100.0:
        data = raw_to_celsius(data, scale=scale)

    return data


# ---------------------------------------------------------------------------
# Convenience: auto-detect format
# ---------------------------------------------------------------------------

def load_temperature(filepath: str, **kwargs) -> np.ndarray:
    """
    Load a temperature matrix, auto-detecting the format from extension.

    Supported extensions: .npy, .csv, .h5/.hdf5.

    Args:
        filepath: path to the data file.
        **kwargs: forwarded to the specific loader.

    Returns:
        float32 array of shape (N, H, W).
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".npy":
        return load_npy(filepath, **kwargs)
    elif ext == ".csv":
        return load_csv(filepath, **kwargs)
    elif ext in (".h5", ".hdf5"):
        return load_h5(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


# ---------------------------------------------------------------------------
# Data validation
# ---------------------------------------------------------------------------

def check_data_shape(filepath: str) -> dict:
    """
    Quick inspection of a temperature matrix file.

    Returns a dict with keys: shape, dtype, min, max, mean, std.
    """
    data = load_temperature(filepath)
    return {
        "shape": data.shape,
        "dtype": str(data.dtype),
        "min": float(data.min()),
        "max": float(data.max()),
        "mean": float(data.mean()),
        "std": float(data.std()),
    }
