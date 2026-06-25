"""
Verified binary .xtherm parse core — SINGLE SOURCE OF TRUTH.

The WeldStudio temperature-matrix .xtherm layout was verified empirically on
real exports (2026-06):

    file size = header_bytes + width * height * itemsize
              = 56 + 640 * 512 * 2
              = 655416 bytes
    payload   = little-endian uint16 raw counts
    T_celsius = raw * scale_C_per_count + offset_C

Both ``scripts/02b_convert_xtherm_binary_to_npy.py`` (legacy-compatible
single-folder conversion) and ``scripts/02c_batch_convert_tracks.py`` (formal
per-track batch conversion) import this module, so there is exactly one parser.

The authoritative formal binary-format values are loaded by the entry scripts
from ``configs/xtherm_format.yaml``. This module receives explicit parameters
and never reads configuration files itself. Source files are opened read-only
and are never written, moved, renamed, or deleted.
"""

from __future__ import annotations

import glob
import os
import re
from collections.abc import Mapping

import numpy as np


class XthermSizeError(ValueError):
    """A .xtherm file size differs from the verified frame size."""


def build_numpy_dtype(dtype_name: str, endian: str) -> np.dtype:
    """Map a configured dtype and byte order to a concrete NumPy dtype."""
    if endian not in {"little", "big"}:
        raise ValueError(
            f"endian must be 'little' or 'big', got {endian!r}"
        )
    prefix = "<" if endian == "little" else ">"
    return np.dtype(dtype_name).newbyteorder(prefix)


def natural_sort_key(name: str) -> list[tuple[int, object]]:
    """Natural-sort key so numeric filename components sort numerically."""
    parts = re.split(r"(\d+)", str(name))
    key: list[tuple[int, object]] = []
    for token in parts:
        if token.isdigit():
            key.append((0, int(token)))
        elif token:
            key.append((1, token.lower()))
    return key


def list_xtherm_files(
    input_dir: str | os.PathLike[str],
    recursive: bool = True,
) -> list[str]:
    """List XTherm frames under ``input_dir`` in deterministic natural order."""
    input_dir = os.fspath(input_dir)
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"input_dir not found: {input_dir}")

    found: list[str] = []
    if recursive:
        for pattern in ("*.xtherm", "*.XTHERM"):
            found.extend(
                glob.glob(
                    os.path.join(input_dir, "**", pattern),
                    recursive=True,
                )
            )
    else:
        for pattern in ("*.xtherm", "*.XTHERM"):
            found.extend(glob.glob(os.path.join(input_dir, pattern)))

    return sorted(
        set(found),
        key=lambda path: natural_sort_key(
            os.path.relpath(path, input_dir).replace(os.sep, "/")
        ),
    )


def expected_frame_size(
    width: int,
    height: int,
    header_bytes: int,
    np_dtype: np.dtype,
) -> int:
    """Return the exact byte size required for one verified frame."""
    return (
        int(header_bytes)
        + int(width) * int(height) * int(np_dtype.itemsize)
    )


def read_xtherm_frame(
    path: str | os.PathLike[str],
    width: int,
    height: int,
    header_bytes: int,
    np_dtype: np.dtype,
    scale_factor: float,
    offset_C: float = 0.0,
) -> np.ndarray:
    """Read one XTherm frame into an ``(H, W)`` float32 Celsius array."""
    path = os.fspath(path)
    expected = expected_frame_size(
        width,
        height,
        header_bytes,
        np_dtype,
    )
    actual = os.path.getsize(path)
    if actual != expected:
        raise XthermSizeError(
            f"size mismatch for '{os.path.basename(path)}': "
            f"expected {expected} bytes "
            f"(header {header_bytes} + "
            f"{width}x{height}x{np_dtype.itemsize}), "
            f"got {actual} bytes ({path})"
        )

    raw = np.fromfile(
        path,
        dtype=np_dtype,
        count=int(width) * int(height),
        offset=int(header_bytes),
    )
    frame = raw.reshape(int(height), int(width)).astype(
        np.float32,
        copy=False,
    )
    return (
        frame * np.float32(scale_factor)
        + np.float32(offset_C)
    ).astype(np.float32, copy=False)


def convert_xtherm_dir(
    input_dir_or_config,
    *,
    width: int | None = None,
    height: int | None = None,
    header_bytes: int | None = None,
    dtype_name: str | None = None,
    endian: str | None = None,
    scale_factor: float | None = None,
    offset_C: float = 0.0,
    recursive: bool = True,
):
    """Convert a directory of XTherm frames with one shared parser.

    Preferred formal/updated call::

        convert_xtherm_dir(
            input_dir,
            width=640,
            height=512,
            header_bytes=56,
            dtype_name="uint16",
            endian="little",
            scale_factor=0.1,
            offset_C=0.0,
        )

    A mapping with the historical keys is still accepted to preserve legacy
    callers and tests. That compatibility path does not make the mapping an
    authoritative formal-format source.
    """
    if isinstance(input_dir_or_config, Mapping):
        legacy = input_dir_or_config
        input_dir = legacy["input_dir"]
        width = int(legacy["width"])
        height = int(legacy["height"])
        header_bytes = int(legacy["header_bytes"])
        dtype_name = str(legacy["dtype"])
        endian = str(legacy["endian"])
        scale_factor = float(legacy["scale_factor"])
        offset_C = float(legacy.get("offset_C", 0.0))
    else:
        input_dir = os.fspath(input_dir_or_config)

    required = {
        "width": width,
        "height": height,
        "header_bytes": header_bytes,
        "dtype_name": dtype_name,
        "endian": endian,
        "scale_factor": scale_factor,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(
            "Missing required XTherm conversion parameters: "
            + ", ".join(missing)
        )

    width = int(width)
    height = int(height)
    header_bytes = int(header_bytes)
    scale_factor = float(scale_factor)
    offset_C = float(offset_C)
    np_dtype = build_numpy_dtype(str(dtype_name), str(endian))

    files = list_xtherm_files(input_dir, recursive=recursive)
    if not files:
        raise FileNotFoundError(
            f"no .xtherm files found under {input_dir}"
        )

    data = np.empty((len(files), height, width), dtype=np.float32)
    for index, path in enumerate(files):
        data[index] = read_xtherm_frame(
            path,
            width,
            height,
            header_bytes,
            np_dtype,
            scale_factor,
            offset_C,
        )
    return data, files
