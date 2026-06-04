"""
03_extract_roi.py

Read standardized temperature matrices (N x H x W float32 Celsius) from
data/processed/matrix, crop the configured ROI, and save to data/processed/roi.

If roi.enabled is false, the frames are copied through unchanged.

Usage:
    python scripts/03_extract_roi.py --config configs/default.yaml
"""

import os
import sys
import glob
import argparse

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from utils.config import load_config
from preprocess.roi import crop_roi, validate_bounds


def _resolve_bounds(rcfg: dict) -> tuple:
    """
    Resolve ROI to (x1, y1, x2, y2) inclusive pixel bounds (x=col, y=row).

    Preferred: top/left/height/width. Falls back to the legacy `bounds`
    list when height or width is missing/None.
    """
    top = rcfg.get("top")
    left = rcfg.get("left")
    height = rcfg.get("height")
    width = rcfg.get("width")

    if height is not None and width is not None and top is not None and left is not None:
        x1 = int(left)
        y1 = int(top)
        x2 = int(left) + int(width) - 1
        y2 = int(top) + int(height) - 1
        return (x1, y1, x2, y2)

    return tuple(rcfg.get("bounds", [0, 0, 0, 0]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config["paths"]
    rcfg = config["roi"]

    in_dir = paths["processed_matrix"]
    out_dir = paths["processed_roi"]
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(in_dir, "*.npy")))
    if not files:
        print(f"[03] No matrices found under {in_dir}. Run 02 first.")
        return

    enabled = bool(rcfg.get("enabled", True))
    bounds = _resolve_bounds(rcfg)
    print(f"[03] input: {len(files)} matrix file(s) from {in_dir}")
    print(f"[03] roi.enabled={enabled} bounds(x1,y1,x2,y2)={bounds}")

    n_done = 0
    for filepath in files:
        data = np.load(filepath).astype(np.float32)
        if enabled:
            if not validate_bounds(data, bounds):
                print(f"      SKIP {os.path.basename(filepath)} "
                      f"(bounds {bounds} invalid for shape {data.shape})")
                continue
            roi = crop_roi(data, bounds)
        else:
            roi = data

        stem = os.path.splitext(os.path.basename(filepath))[0]
        out_path = os.path.join(out_dir, f"{stem}.npy")
        np.save(out_path, roi.astype(np.float32))
        n_done += 1
        print(f"      {os.path.basename(filepath):30s} {data.shape} "
              f"-> {roi.shape}  {os.path.basename(out_path)}")

    print(f"[03] wrote {n_done} ROI file(s) -> {out_dir}")


if __name__ == "__main__":
    main()
