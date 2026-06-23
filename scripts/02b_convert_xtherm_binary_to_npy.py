"""
02b_convert_xtherm_binary_to_npy.py

Convert binary .xtherm files (WeldStudio temperature-matrix export) into a
single stacked N x H x W float32 Celsius .npy plus a JSON metadata file.

The binary layout was verified empirically on real exports (2026-06):
  file size = header_bytes + width * height * 2   (655416 = 56 + 640*512*2)
  payload   = little-endian uint16 raw counts; T_celsius = raw * 0.1
  (big-endian reads give ~6500 C on the first frame -> wrong)

All format parameters come from the `xtherm_binary` section of the YAML
config. This script is READ-ONLY with respect to data/raw_xtherm: it never
deletes, moves, or modifies any original .xtherm file.

Output (local only, never committed to git):
  data/exported/npy/dataset.npy        N x H x W float32 Celsius
  data/exported/npy/dataset_meta.json  conversion metadata

Usage:
    python scripts/02b_convert_xtherm_binary_to_npy.py --config configs/default.yaml

NOTE: the output is already Celsius, so `data.exported_is_celsius` must be
true before running script 02 on it (otherwise 02 divides by 10 again).
"""

import os
import sys
import glob
import json
import argparse

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from utils.config import load_config

# Single source of truth for the verified .xtherm parse algorithm. These names
# are re-exported so existing callers/tests of this script keep working while the
# actual implementation lives in one place (shared with scripts/02c).
from conversion.xtherm_binary import (  # noqa: F401  (re-export)
    XthermSizeError,
    build_numpy_dtype,
    list_xtherm_files,
    read_xtherm_frame,
    convert_xtherm_dir,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    xb = config["xtherm_binary"]
    valid_max = float(config["data"]["valid_range"][1])
    zero_note_threshold = float(xb["zero_ratio_note_threshold"])

    print("[02b] Binary .xtherm -> npy conversion "
          "(read-only on raw files; format from configs)")
    print("-" * 60)

    data, files = convert_xtherm_dir(xb)
    first_file = os.path.basename(files[0])
    last_file = os.path.basename(files[-1])

    t_min = float(data.min())
    t_max = float(data.max())
    t_mean = float(data.mean())
    zero_ratio = float(np.count_nonzero(data == 0.0) / data.size)

    out_npy = xb["output_npy"]
    out_meta = xb["output_meta"]
    os.makedirs(os.path.dirname(out_npy), exist_ok=True)
    os.makedirs(os.path.dirname(out_meta), exist_ok=True)
    np.save(out_npy, data)

    meta = {
        "sample_id": xb["sample_id"],
        "source_dir": xb["input_dir"],
        "frame_count": len(files),
        "first_file": first_file,
        "last_file": last_file,
        "width": int(xb["width"]),
        "height": int(xb["height"]),
        "header_bytes": int(xb["header_bytes"]),
        "dtype": xb["dtype"],
        "endian": xb["endian"],
        "scale_factor": float(xb["scale_factor"]),
        "output_shape": list(data.shape),
        "unit": "Celsius",
        "min_temperature": t_min,
        "max_temperature": t_max,
        "mean_temperature": t_mean,
        "zero_pixel_ratio": zero_ratio,
        "created_by": "scripts/02b_convert_xtherm_binary_to_npy.py",
    }
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[02b] loaded frames    : {len(files)}")
    print(f"[02b] first filename   : {first_file}")
    print(f"[02b] last filename    : {last_file}")
    print(f"[02b] output shape     : {data.shape} (N x H x W, float32 Celsius)")
    print(f"[02b] min/max/mean temp: {t_min:.1f} / {t_max:.1f} / {t_mean:.2f} C")
    print(f"[02b] zero pixel ratio : {zero_ratio:.4f}")
    print(f"[02b] output_npy       : {out_npy}")
    print(f"[02b] output_meta      : {out_meta}")

    if t_max > valid_max:
        print(f"[02b] WARNING: max temperature {t_max:.1f} C exceeds "
              f"{valid_max:.0f} C - check dtype/endian/scale_factor settings.")
    if zero_ratio > zero_note_threshold:
        print(f"[02b] NOTE: zero pixel ratio {zero_ratio:.4f} > "
              f"{zero_note_threshold} - edge zeros are likely invalid pixels "
              f"or background; not treated as an error.")

    print("-" * 60)
    print("[02b] Raw .xtherm files were only READ - never deleted, moved, "
          "or modified.")
    print("[02b] Output is ALREADY Celsius: keep data.exported_is_celsius "
          "true so script 02 does not divide by 10 again.")


if __name__ == "__main__":
    main()
