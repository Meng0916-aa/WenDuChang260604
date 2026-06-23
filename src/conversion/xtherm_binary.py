"""
Verified binary .xtherm parse core — SINGLE SOURCE OF TRUTH.

The WeldStudio temperature-matrix .xtherm layout was verified empirically on real
exports (2026-06):

    file size = header_bytes + width * height * itemsize   (655416 = 56 + 640*512*2)
    payload   = little-endian uint16 raw counts
    T_celsius = raw * scale_factor                          (scale_factor = 0.1)

Both ``scripts/02b_convert_xtherm_binary_to_npy.py`` (single-folder dataset) and
``scripts/02c_batch_convert_tracks.py`` (per-track batch) import this module so
there is exactly ONE parse algorithm. This module is READ-ONLY with respect to
source files: it only opens them for reading, never writes/moves/deletes.

Format parameters are passed in (sourced from configs) — no magic numbers are
hard-coded into the parsing functions.
"""

import os
import re
import glob

import numpy as np


class XthermSizeError(ValueError):
    """A .xtherm file size != header_bytes + width*height*itemsize."""


def build_numpy_dtype(dtype_name, endian):
    """Map config (dtype, endian) to a concrete numpy dtype, e.g. '<u2'."""
    if endian not in ("little", "big"):
        raise ValueError(
            f"endian must be 'little' or 'big', got {endian!r}")
    prefix = "<" if endian == "little" else ">"
    return np.dtype(dtype_name).newbyteorder(prefix)


def natural_sort_key(name):
    """Natural-sort key: split digits from non-digits so '2' < '10'.

    Returns a list of (type_tag, value) tuples so int and str chunks never
    compare against each other (avoids TypeError on mixed keys).
    """
    parts = re.split(r"(\d+)", str(name))
    key = []
    for tok in parts:
        if tok.isdigit():
            key.append((0, int(tok)))
        elif tok:
            key.append((1, tok.lower()))
    return key


def list_xtherm_files(input_dir, recursive=True):
    """List .xtherm files under input_dir, NATURAL-sorted by relative path.

    recursive=True  -> search subfolders (used by 02b's dataset import).
    recursive=False -> only files directly in input_dir (used by 02c per track).
    Read-only: files are only listed by name, never opened here.
    """
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"input_dir not found: {input_dir}")
    found = []
    if recursive:
        for pat in ("*.xtherm", "*.XTHERM"):
            found.extend(glob.glob(os.path.join(input_dir, "**", pat),
                                   recursive=True))
    else:
        for pat in ("*.xtherm", "*.XTHERM"):
            found.extend(glob.glob(os.path.join(input_dir, pat)))
    uniq = set(found)
    return sorted(
        uniq,
        key=lambda p: natural_sort_key(
            os.path.relpath(p, input_dir).replace(os.sep, "/")),
    )


def expected_frame_size(width, height, header_bytes, np_dtype):
    """Bytes a single valid .xtherm frame must have."""
    return int(header_bytes) + int(width) * int(height) * np_dtype.itemsize


def read_xtherm_frame(path, width, height, header_bytes, np_dtype, scale_factor):
    """Read ONE .xtherm file -> (H, W) float32 Celsius. Never writes.

    Raises XthermSizeError if the file size does not match the verified layout.
    """
    expected = expected_frame_size(width, height, header_bytes, np_dtype)
    actual = os.path.getsize(path)
    if actual != expected:
        raise XthermSizeError(
            f"size mismatch for '{os.path.basename(path)}': "
            f"expected {expected} bytes "
            f"(header {header_bytes} + {width}x{height}x{np_dtype.itemsize}), "
            f"got {actual} bytes ({path})")
    raw = np.fromfile(path, dtype=np_dtype, count=int(width) * int(height),
                      offset=int(header_bytes))
    frame = raw.reshape(int(height), int(width)).astype(np.float32)
    return frame * np.float32(scale_factor)


def convert_xtherm_dir(xb):
    """Convert all .xtherm files described by config section ``xb`` (02b flow).

    Returns (data, files): data is N x H x W float32 Celsius; files is the
    natural-sorted list of source paths. Kept here so 02b and its tests have a
    single shared implementation.
    """
    width = int(xb["width"])
    height = int(xb["height"])
    header_bytes = int(xb["header_bytes"])
    scale_factor = float(xb["scale_factor"])
    np_dtype = build_numpy_dtype(xb["dtype"], xb["endian"])

    files = list_xtherm_files(xb["input_dir"], recursive=True)
    if not files:
        raise FileNotFoundError(
            f"no .xtherm files found under {xb['input_dir']}")

    data = np.empty((len(files), height, width), dtype=np.float32)
    for i, path in enumerate(files):
        data[i] = read_xtherm_frame(path, width, height, header_bytes,
                                    np_dtype, scale_factor)
    return data, files
