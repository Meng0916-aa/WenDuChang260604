"""
01_check_raw_data.py

Sanity-check the data directories WITHOUT parsing any .xtherm binary, and
flag leftover SIMULATED files before real data is imported. This script is
read-only: it NEVER deletes, moves, or modifies any file.

Checks:
  - raw_xtherm file count (.xtherm) — counted RECURSIVELY (subfolders such as
    data/raw_xtherm/dataset/ are included), never parsed. Per-subfolder counts
    and first/last frame filenames are reported.
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


def _scan_raw_xtherm(directory):
    """Recursively collect .xtherm files under directory, grouped by subfolder.

    Returns (all_files, groups) where groups maps a subfolder path relative to
    directory ("." = top level) to its sorted file list. Returns (None, None)
    if the directory does not exist. Read-only: files are only listed by name,
    never opened or parsed.
    """
    if not os.path.isdir(directory):
        return None, None
    found = []
    for pat in ("*.xtherm", "*.XTHERM"):
        found.extend(glob.glob(os.path.join(directory, "**", pat),
                               recursive=True))
    all_files = sorted(set(found))
    groups = {}
    for f in all_files:
        rel_dir = os.path.relpath(os.path.dirname(f), directory)
        groups.setdefault(rel_dir, []).append(f)
    return all_files, groups


def _subfolders_without_xtherm(directory, groups):
    """Immediate subfolders of directory containing no .xtherm anywhere below."""
    empty = []
    for entry in sorted(os.listdir(directory)):
        sub = os.path.join(directory, entry)
        if not os.path.isdir(sub):
            continue
        has_files = any(g == entry or g.startswith(entry + os.sep)
                        for g in groups)
        if not has_files:
            empty.append(entry)
    return empty


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

    # 1. Raw xtherm (counted recursively, never parsed)
    print("[01] raw_xtherm (recursive, subfolders included):")
    raw, raw_groups = _scan_raw_xtherm(paths["raw_xtherm"])
    if raw is None:
        print(f"  [MISS ] {'raw_xtherm':24s} files=   -  "
              f"({paths['raw_xtherm']})")
        raw = []
    else:
        print(f"  [OK   ] {'raw_xtherm':24s} files={len(raw):>4}  "
              f"({paths['raw_xtherm']})")
        for sub in sorted(raw_groups):
            files = raw_groups[sub]
            names = [os.path.basename(f) for f in files]
            label = "(top level)" if sub == "." else sub
            print(f"    {label}: {len(files)} files")
            print(f"    {label}: first={names[0]}, last={names[-1]}")
        empty_subs = _subfolders_without_xtherm(paths["raw_xtherm"],
                                                raw_groups)
        for sub in empty_subs:
            print(f"    NOTE: subfolder '{sub}' contains no .xtherm files.")
        if not raw and not empty_subs:
            print("    (no .xtherm files and no subfolders found)")

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
    print(f"[01] raw_xtherm files          : {len(raw)} (recursive)")
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
