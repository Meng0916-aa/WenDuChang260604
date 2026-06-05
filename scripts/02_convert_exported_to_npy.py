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


def _validate_shape(data: np.ndarray, filepath: str,
                    expected_ndim: int = 3,
                    expected_frame_axis: int = 0,
                    expected_height=None,
                    expected_width=None,
                    min_frames: int = 2) -> np.ndarray:
    """
    Enforce the N x H x W contract explicitly; NEVER reshape or transpose.

    Rules:
      - A 2-D (H, W) single frame is expanded to (1, H, W); this counts as a
        legitimate single frame and is exempt from the min_frames check.
      - Any other ndim raises ShapeContractError.
      - Frames must be on axis 0 (expected_frame_axis must be 0).
      - If expected_height / expected_width are set, axes 1 / 2 must match.
      - A genuine multi-frame array with N < min_frames is rejected: the most
        likely cause is wrong axis order (e.g. H x W x N), which must be
        re-exported as N x H x W rather than silently transposed.
    """
    name = os.path.basename(filepath)

    if int(expected_frame_axis) != 0:
        raise ShapeContractError(
            f"{name}: data.expected_frame_axis={expected_frame_axis} is not "
            f"supported. Frames must be axis 0 (N x H x W). Re-export with "
            f"frames first."
        )

    was_single_frame = (data.ndim == 2)
    if was_single_frame and expected_ndim == 3:
        data = data[np.newaxis, ...]   # (H, W) -> (1, H, W)

    if data.ndim != expected_ndim:
        raise ShapeContractError(
            f"{name}: expected N x H x W ({expected_ndim}-D) but got "
            f"ndim={data.ndim}, shape={data.shape}. Re-export as N x H x W "
            f"(or a single H x W frame). No silent reshape/transpose is done."
        )

    n, h, w = data.shape

    # Degenerate spatial dims almost always mean a wrong axis order.
    if h < 2 or w < 2:
        raise ShapeContractError(
            f"{name}: spatial dims look wrong (H={h}, W={w}) for shape "
            f"{data.shape}. Expected N x H x W, got shape={data.shape}. "
            f"If your software exported H x W x N, convert it to N x H x W "
            f"before importing (this script will NOT transpose for you)."
        )

    # Explicit H / W checks when configured.
    if expected_height is not None and h != int(expected_height):
        raise ShapeContractError(
            f"{name}: height mismatch — expected H={int(expected_height)} but "
            f"got H={h}. Expected N x H x W, got shape={data.shape}."
        )
    if expected_width is not None and w != int(expected_width):
        raise ShapeContractError(
            f"{name}: width mismatch — expected W={int(expected_width)} but "
            f"got W={w}. Expected N x H x W, got shape={data.shape}."
        )

    # Too few frames on axis 0. A genuine single frame (N == 1) is allowed —
    # note the loaders may already have expanded a 2-D (H, W) frame to (1,H,W),
    # so N == 1 is treated as the single-frame case regardless of input ndim.
    is_single_frame = was_single_frame or (n == 1)
    if n == 0 or (n < int(min_frames) and not is_single_frame):
        raise ShapeContractError(
            f"{name}: only N={n} frame(s) on axis 0 (min_frames={int(min_frames)}). "
            f"Expected N x H x W, got shape={data.shape}. A small axis-0 size "
            f"usually means the axis order is wrong (e.g. H x W x N) — re-export "
            f"as N x H x W. (A true single H x W frame, N=1, is accepted.)"
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
    expected_frame_axis = int(data_cfg.get("expected_frame_axis", 0))
    expected_height = data_cfg.get("expected_height", None)
    expected_width = data_cfg.get("expected_width", None)
    min_frames = int(data_cfg.get("min_frames", 2))

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
          f"expected={expected_ndim}-D ({axis_order}) "
          f"H={expected_height} W={expected_width} min_frames={min_frames}")

    n_done, n_failed = 0, 0
    for filepath, kind in files:
        name = os.path.basename(filepath)
        try:
            data = _load_raw(filepath, kind).astype(np.float32)
            data = _validate_shape(
                data, filepath,
                expected_ndim=expected_ndim,
                expected_frame_axis=expected_frame_axis,
                expected_height=expected_height,
                expected_width=expected_width,
                min_frames=min_frames,
            )
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
