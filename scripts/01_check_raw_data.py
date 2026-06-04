"""
01_check_raw_data.py

Sanity-check the data directories WITHOUT parsing any .xtherm binary, and
flag leftover SIMULATED files before real data is imported. This script is
read-only: it NEVER deletes, moves, or modifies any file.

Checks:
  - raw_xtherm file count (.xtherm) — counted, never parsed.
  - exported file counts under data/exported/{npy,csv,h5}.
  - processed file counts under data/processed/{matrix,roi,thermal_cycle}.
  - WARNING if data/processed/thermal_cycle contains SIM_*.csv (likely
    leftover simulated data from a code-chain test).

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


def _list(directory, patterns):
    """Return matching file paths in a directory, or None if it doesn't exist."""
    if not os.path.isdir(directory):
        return None
    out = []
    for pat in patterns:
        out.extend(glob.glob(os.path.join(directory, pat)))
    return sorted(out)


def _report(label, directory, patterns):
    files = _list(directory, patterns)
    exists = files is not None
    status = "OK   " if exists else "MISS "
    count = "-" if files is None else str(len(files))
    print(f"  [{status}] {label:24s} files={count:>4}  ({directory})")
    return files or []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config["paths"]

    print("[01] Data directory check (read-only; no .xtherm parsing)")
    print("-" * 60)

    # 1. Raw xtherm (counted, never parsed)
    print("[01] raw_xtherm:")
    raw = _report("raw_xtherm", paths["raw_xtherm"], ["*.xtherm", "*.XTHERM"])

    # 2. Exported matrices
    print("[01] exported (temperature matrices):")
    exp_npy = _report("exported_npy", paths["exported_npy"], ["*.npy"])
    exp_csv = _report("exported_csv", paths["exported_csv"], ["*.csv"])
    exp_h5 = _report("exported_h5", paths["exported_h5"], ["*.h5", "*.hdf5"])
    exported_total = len(exp_npy) + len(exp_csv) + len(exp_h5)

    # 3. Processed outputs (existing/old files)
    print("[01] processed (existing outputs):")
    proc_matrix = _report("processed_matrix", paths["processed_matrix"], ["*.npy"])
    proc_roi = _report("processed_roi", paths["processed_roi"], ["*.npy"])
    proc_cycle = _report("processed_thermal_cycle",
                         paths["processed_thermal_cycle"], ["*.csv"])

    print("-" * 60)
    print(f"[01] raw_xtherm files          : {len(raw)}")
    print(f"[01] exported matrix files     : {exported_total} "
          f"(npy={len(exp_npy)} csv={len(exp_csv)} h5={len(exp_h5)})")
    print(f"[01] processed matrix/roi/cycle: "
          f"{len(proc_matrix)}/{len(proc_roi)}/{len(proc_cycle)}")

    # 4. Warn on leftover SIMULATED data
    sim_files = [f for f in proc_cycle
                 if os.path.basename(f).startswith("SIM_")]
    if sim_files:
        print("-" * 60)
        print(f"[01] WARNING: {len(sim_files)} SIM_*.csv file(s) found in "
              f"{paths['processed_thermal_cycle']}.")
        print("[01] These are likely leftover SIMULATED data from a code-chain "
              "test and are NOT real experiments.")
        print("[01] Remove or archive them MANUALLY before importing real data "
              "(this script never deletes anything).")
        for f in sim_files[:10]:
            print(f"        {os.path.basename(f)}")
        if len(sim_files) > 10:
            print(f"        ... and {len(sim_files) - 10} more")

    # 5. Guidance
    print("-" * 60)
    if exported_total == 0 and len(raw) == 0:
        print("[01] No raw or exported data yet. Export temperature matrices from "
              "WeldStudio into data/exported/{npy,csv,h5} (see "
              "docs/real_data_import.md), then run 02->08.")
    elif exported_total == 0:
        print("[01] Raw .xtherm present but no exported matrices. Export to "
              "data/exported/{npy,csv,h5} before running 02.")
    else:
        print("[01] Exported matrices found. You can run 02->08.")
    print("[01] Reminder: raw .xtherm files are never deleted/modified; their "
          "binary structure is NOT parsed here.")


if __name__ == "__main__":
    main()
