"""
01_check_raw_data.py

Sanity-check the data directories WITHOUT parsing any .xtherm binary.
Reports whether the expected directories exist and counts exported
(.npy/.csv/.h5) and processed files.

This script never reads the internal structure of .xtherm files — that is
deferred to a future reader based on the Xiris WeldSDK / official export.

Usage:
    python scripts/01_check_raw_data.py --config configs/default.yaml
"""

import os
import sys
import glob
import argparse

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from utils.config import load_config


def _count(directory, patterns):
    if not os.path.isdir(directory):
        return None
    total = 0
    for pat in patterns:
        total += len(glob.glob(os.path.join(directory, pat)))
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config["paths"]

    print("[01] Data directory check (no .xtherm parsing)")
    print("-" * 56)

    checks = [
        ("raw_xtherm", paths["raw_xtherm"], ["*.xtherm", "*.XTHERM"]),
        ("exported_npy", paths["exported_npy"], ["*.npy"]),
        ("exported_csv", paths["exported_csv"], ["*.csv"]),
        ("exported_h5", paths["exported_h5"], ["*.h5", "*.hdf5"]),
        ("processed_matrix", paths["processed_matrix"], ["*.npy"]),
        ("processed_roi", paths["processed_roi"], ["*.npy"]),
        ("processed_thermal_cycle", paths["processed_thermal_cycle"], ["*.csv"]),
    ]

    exported_total = 0
    for name, directory, patterns in checks:
        exists = os.path.isdir(directory)
        n = _count(directory, patterns)
        status = "OK   " if exists else "MISS "
        count_str = "-" if n is None else str(n)
        print(f"  [{status}] {name:24s} files={count_str:>4}  ({directory})")
        if name.startswith("exported_") and n:
            exported_total += n

    print("-" * 56)
    print(f"[01] exported temperature-matrix files total: {exported_total}")
    if exported_total == 0:
        print("[01] No exported npy/csv/h5 found. Provide exported matrices, or "
              "rely on simulation in script 05 for a code-chain test.")
    print("[01] Reminder: raw .xtherm files are never deleted/modified; their "
          "binary structure is NOT parsed here.")


if __name__ == "__main__":
    main()
