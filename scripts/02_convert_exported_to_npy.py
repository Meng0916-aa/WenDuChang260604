"""
02_convert_exported_to_npy.py

Read exported temperature matrices from data/exported/{npy,csv,h5} and write
standardized float32 Celsius N x H x W .npy files to data/processed/matrix.

Celsius handling is controlled by config (NOT by a magic heuristic):
  data.exported_is_celsius = true   -> values are already Celsius; DO NOT scale
  data.exported_is_celsius = false  -> values are raw counts; multiply by
                                       data.temperature_scale (default 0.1)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config["paths"]
    data_cfg = config["data"]
    scale = float(data_cfg.get("temperature_scale", 0.1))
    already_celsius = bool(data_cfg.get("exported_is_celsius", False))

    out_dir = paths["processed_matrix"]
    os.makedirs(out_dir, exist_ok=True)

    files = _gather(paths)
    if not files:
        print("[02] No exported npy/csv/h5 files found under data/exported/. "
              "Nothing to convert.")
        return

    print(f"[02] input: {len(files)} exported file(s)")
    print(f"[02] exported_is_celsius={already_celsius} "
          f"scale={scale if not already_celsius else 'n/a'}")

    n_done = 0
    for filepath, kind in files:
        data = _load_raw(filepath, kind).astype(np.float32)
        if not already_celsius:
            data = raw_to_celsius(data, scale=scale)   # raw -> Celsius
        # else: already Celsius — do NOT divide again

        stem = os.path.splitext(os.path.basename(filepath))[0]
        out_path = os.path.join(out_dir, f"{stem}.npy")
        np.save(out_path, data.astype(np.float32))
        n_done += 1
        print(f"      {kind:3s} {os.path.basename(filepath):30s} "
              f"-> {os.path.basename(out_path)}  shape={data.shape} "
              f"min={data.min():.1f} max={data.max():.1f} C")

    print(f"[02] converted {n_done} file(s) -> {out_dir} (float32 Celsius, N x H x W)")


if __name__ == "__main__":
    main()
