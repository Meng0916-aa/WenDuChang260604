"""
02_convert_exported_to_npy.py

Read exported temperature matrices from data/exported/{npy,csv,h5} and write
standardized float32 Celsius N x H x W .npy files to data/processed/matrix.

Celsius handling is controlled by config (NOT by a magic heuristic):
  data.exported_is_celsius = true   -> values are already Celsius; DO NOT scale
  data.exported_is_celsius = false  -> values are raw counts; multiply by
                                       data.temperature_scale (default 0.1)

Shape contract (enforced explicitly, no silent reshape):
  - final array must have data.expected_ndim dimensions (default 3 => N x H x W)
  - a single H x W frame (ndim == 2) is auto-expanded to (1, H, W)
  - anything else raises a clear error naming the offending file and shape

Usage:
    python scripts/02_convert_exported_to_npy.py --config configs/default.yaml
"""

import os
import sys
import glob
import argparse

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from utils.config import load_config

# NOTE: the export loader lives in src/io/, but `io` collides with Python's
# stdlib `io`. Load it directly from its file path to avoid shadowing.
import importlib.util as _ilu
_EXPORT_LOADER_PATH = os.path.join(_ROOT, "src", "io", "export_loader.py")
_spec = _ilu.spec_from_file_location("export_loader", _EXPORT_LOADER_PATH)
_export_loader = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_export_loader)
load_npy = _export_loader.load_npy
load_csv = _export_loader.load_csv
load_h5 = _export_loader.load_h5
raw_to_celsius = _export_loader.raw_to_celsius


class ShapeContractError(ValueError):
    """Raised when an exported array does not match the N x H x W contract."""


def _gather(paths):
    """Collect (filepath, kind) for every exported matrix file."""
    files = []
    for f in sorted(glob.glob(os.path.join(paths["exported_npy"], "*.npy"))):
        files.append((f, "npy"))
    for f in sorted(glob.glob(os.path.join(paths["exported_csv"], "*.csv"))):
        files.append((f, "csv"))
    for ext in ("*.h5", "*.hdf5"):
        for f in sorted(glob.glob(os.path.join(paths["exported_h5"], ext))):
            files.append((f, "h5"))
    return files


def _load_raw(filepath, kind):
    """Load WITHOUT auto Celsius conversion (we scale explicitly below)."""
    if kind == "npy":
        return load_npy(filepath, as_celsius=False)
    if kind == "csv":
        return load_csv(filepath, as_celsius=False)
    return load_h5(filepath, as_celsius=False)


def _validate_shape(data: np.ndarray, expected_ndim: int, filepath: str) -> np.ndarray:
    """
    Enforce the N x H x W contract explicitly; never reshape silently.

    A 2-D (H, W) single frame is expanded to (1, H, W). Any other ndim, or a
    degenerate spatial dimension, raises ShapeContractError naming the file.
    """
    name = os.path.basename(filepath)

    if data.ndim == 2 and expected_ndim == 3:
        data = data[np.newaxis, ...]   # single frame -> (1, H, W)

    if data.ndim != expected_ndim:
        raise ShapeContractError(
            f"{name}: expected {expected_ndim}-D (N x H x W) data but got "
            f"ndim={data.ndim}, shape={data.shape}. Re-export as N x H x W "
            f"(or a single H x W frame). No silent reshape is performed."
        )

    n, h, w = data.shape
    if h < 2 or w < 2:
        raise ShapeContractError(
            f"{name}: spatial dims look wrong (H={h}, W={w}) for shape "
            f"{data.shape}. Check the axis order — it must be N x H x W "
            f"(frames first), not H x W x N or similar."
        )
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config["paths"]
    data_cfg = config["data"]
    scale = float(data_cfg.get("temperature_scale", 0.1))
    already_celsius = bool(data_cfg.get("exported_is_celsius", False))
    expected_ndim = int(data_cfg.get("expected_ndim", 3))
    axis_order = str(data_cfg.get("expected_axis_order", "NHW"))

    out_dir = paths["processed_matrix"]
    os.makedirs(out_dir, exist_ok=True)

    files = _gather(paths)
    if not files:
        print("[02] No exported npy/csv/h5 files found under data/exported/. "
              "Nothing to convert.")
        return

    print(f"[02] input: {len(files)} exported file(s)")
    print(f"[02] exported_is_celsius={already_celsius} "
          f"scale={scale if not already_celsius else 'n/a'} "
          f"expected={expected_ndim}-D ({axis_order})")

    n_done, n_failed = 0, 0
    for filepath, kind in files:
        name = os.path.basename(filepath)
        try:
            data = _load_raw(filepath, kind).astype(np.float32)
            data = _validate_shape(data, expected_ndim, filepath)
        except ShapeContractError as e:
            n_failed += 1
            print(f"      ERROR {name}: {e}")
            continue
        except Exception as e:                       # noqa: BLE001 (report + continue)
            n_failed += 1
            print(f"      ERROR {name}: failed to load ({type(e).__name__}: {e})")
            continue

        if not already_celsius:
            data = raw_to_celsius(data, scale=scale)   # raw -> Celsius
        # else: already Celsius — do NOT divide again

        stem = os.path.splitext(name)[0]
        out_path = os.path.join(out_dir, f"{stem}.npy")
        np.save(out_path, data.astype(np.float32))
        n_done += 1
        print(f"      {kind:3s} {name:30s} -> {os.path.basename(out_path)}  "
              f"shape={data.shape} min={data.min():.1f} max={data.max():.1f} C")

    print(f"[02] converted {n_done} file(s) -> {out_dir} "
          f"(float32 Celsius, N x H x W)")
    if n_failed:
        print(f"[02] WARNING: {n_failed} file(s) FAILED the shape/format contract "
              f"and were skipped (see ERROR lines above). Nothing was deleted.")
        sys.exit(1)


if __name__ == "__main__":
    main()
