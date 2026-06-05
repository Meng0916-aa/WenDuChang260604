"""
05_build_window_dataset.py

Build sliding-window samples for thermal-cycle prediction and persist them
as a single .npz bundle (train / val / test splits, with per-sample
experiment ids and magnetic-group labels) under paths.processed_samples.

Input:
  - Real thermal-cycle CSVs under paths.processed_thermal_cycle, if present
    (columns: frame, tmax, center_average, hot_zone_average).
  - Otherwise, if simulation.enabled, a small SIMULATED dataset is generated
    so the downstream training / evaluation code chain can be exercised.
    SIMULATED data is clearly labelled and is NOT experimental data.

Output:
  - <paths.processed_samples>/<dataset.samples_file>  (.npz bundle)

All parameters come from the YAML config.

Usage:
    python scripts/05_build_window_dataset.py --config configs/default.yaml
"""

import os
import sys
import glob
import argparse

import numpy as np

# Make src/ importable
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from utils.config import load_config
from utils.seed import set_seed
from preprocess.thermal_cycle import save_thermal_cycles
from datasets.window_dataset import (
    build_windows, split_experiment_indices,
)

_CURVE_COLUMNS = ("tmax", "center_average", "hot_zone_average")


# ---------------------------------------------------------------------------
# Simulated data (fallback only)
# ---------------------------------------------------------------------------

def _simulate_thermal_cycles(config: dict) -> list:
    """Generate SIMULATED thermal-cycle curves; save SIM_***.csv. NOT real data."""
    sim = config.get("simulation", {})
    n_exp = int(sim.get("n_experiments", 6))
    n_frames = int(sim.get("n_frames", 400))
    seed = int(sim.get("seed", 42))
    out_dir = config["paths"]["processed_thermal_cycle"]
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.RandomState(seed)
    t = np.linspace(0.0, 1.0, n_frames, dtype=np.float32)

    experiments = []
    for i in range(n_exp):
        peak = 1200.0 + 300.0 * rng.rand()
        peak_pos = 0.25 + 0.1 * rng.rand()
        decay = 3.0 + 2.0 * rng.rand()
        rise = np.clip(t / peak_pos, 0.0, 1.0)
        fall = np.exp(-decay * np.clip(t - peak_pos, 0.0, None))
        base = peak * rise * fall + 200.0
        tmax = (base + rng.normal(0.0, 8.0, n_frames)).astype(np.float32)
        cycles = {
            "tmax": tmax,
            "center_average": (tmax * 0.85 + rng.normal(0, 5, n_frames)).astype(np.float32),
            "hot_zone_average": (tmax * 0.92 + rng.normal(0, 5, n_frames)).astype(np.float32),
        }
        exp_id = f"SIM_{i:03d}"
        save_thermal_cycles(cycles, os.path.join(out_dir, f"{exp_id}.csv"))
        experiments.append((exp_id, cycles))

    print(f"[05] SIMULATED dataset generated: {n_exp} experiments x "
          f"{n_frames} frames -> {out_dir}")
    print("[05] WARNING: SIMULATED curves are placeholders, NOT experimental data.")
    return experiments


# ---------------------------------------------------------------------------
# Real CSV loading
# ---------------------------------------------------------------------------

class MixedDataError(RuntimeError):
    """Raised when SIM_ and real thermal-cycle CSVs coexist in one directory."""


def _classify_cycle_csvs(in_dir: str) -> tuple:
    """
    List thermal-cycle CSVs and split them into (real, simulated) by filename.

    A file whose basename starts with 'SIM_' is simulated; everything else is
    treated as a real experiment.

    Returns:
        (real_paths, sim_paths) — both sorted lists.
    """
    csv_files = sorted(glob.glob(os.path.join(in_dir, "*.csv")))
    sim = [p for p in csv_files if os.path.basename(p).startswith("SIM_")]
    real = [p for p in csv_files if not os.path.basename(p).startswith("SIM_")]
    return real, sim


def _load_cycle_csvs(paths: list) -> list:
    """Load given thermal-cycle CSV paths into [(exp_id, cycles_dict), ...]."""
    experiments = []
    for path in paths:
        arr = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float32)
        exp_id = os.path.splitext(os.path.basename(path))[0]
        cycles = {col: np.asarray(arr[col], dtype=np.float32)
                  for col in _CURVE_COLUMNS if col in (arr.dtype.names or ())}
        experiments.append((exp_id, cycles))
    return experiments


# ---------------------------------------------------------------------------
# Feature / param / group assembly
# ---------------------------------------------------------------------------

def _select_features(cycles: dict, feature_columns: list) -> np.ndarray:
    """Stack requested curves into a (n_frames, F) float32 array."""
    cols = []
    for col in feature_columns:
        if col not in cycles:
            raise KeyError(
                f"feature_column '{col}' not found. Available: {list(cycles.keys())}")
        cols.append(np.asarray(cycles[col], dtype=np.float32))
    return np.stack(cols, axis=1).astype(np.float32)


def _process_params_for(config: dict, exp_id: str):
    """Return (D,) process-param vector for an experiment, or None if disabled."""
    pp = config.get("process_params", {})
    if not pp.get("enabled", False):
        return None
    by_exp = pp.get("by_experiment", {}) or {}
    if exp_id not in by_exp:
        raise KeyError(f"process_params.enabled but no entry for '{exp_id}'.")
    return np.asarray(by_exp[exp_id], dtype=np.float32)


def _magnetic_group_for(config: dict, exp_id: str) -> str:
    """Map an experiment id to its magnetic group label, or 'unknown'."""
    groups = config.get("magnetic_field_groups", {})
    for key in ("with_magnetic_field", "without_magnetic_field"):
        g = groups.get(key, {}) or {}
        if exp_id in (g.get("experiment_ids", []) or []):
            return g.get("label", key)
    return "unknown"


def _build_split(exp_subset, input_len, pred_len, step, pp_map):
    """
    Build windows for a subset of experiments.

    exp_subset: list of (exp_id, feature_array).
    Returns (X, y, exp_ids_per_sample). Process params (if any) are
    concatenated onto every input time step.
    """
    X_parts, y_parts, ids = [], [], []
    for exp_id, feats in exp_subset:
        Xi, yi = build_windows(feats, input_len, pred_len, step)
        if pp_map is not None:
            p = pp_map[exp_id]
            tiled = np.tile(p[None, None, :], (len(Xi), input_len, 1))
            Xi = np.concatenate([Xi, tiled.astype(np.float32)], axis=-1)
        X_parts.append(Xi)
        y_parts.append(yi)
        ids.extend([exp_id] * len(Xi))
    if not X_parts:
        return (np.empty((0,)), np.empty((0,)), np.array([]))
    return (np.concatenate(X_parts, 0),
            np.concatenate(y_parts, 0),
            np.array(ids))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    dcfg = config["dataset"]
    feature_columns = list(dcfg.get("feature_columns", ["tmax"]))
    input_len = int(dcfg["input_window"])
    pred_len = int(dcfg["predict_window"])
    step = int(dcfg.get("step", 1))

    # 1. Acquire curves. Classify existing CSVs into real vs SIMULATED.
    in_dir = config["paths"]["processed_thermal_cycle"]
    real_paths, sim_paths = _classify_cycle_csvs(in_dir)

    if real_paths and sim_paths:
        # Mixing real and simulated data would silently corrupt results.
        raise MixedDataError(
            f"Found BOTH real and SIMULATED thermal-cycle CSVs in {in_dir}: "
            f"{len(real_paths)} real + {len(sim_paths)} SIM_*.csv. "
            f"Manually archive or remove the SIM_*.csv files before importing "
            f"real data (this script never deletes anything). "
            f"SIM files: {[os.path.basename(p) for p in sim_paths[:5]]}"
            f"{' ...' if len(sim_paths) > 5 else ''}"
        )

    if real_paths:
        experiments = _load_cycle_csvs(real_paths)
        simulated = False
        print(f"[05] input: {len(experiments)} REAL thermal-cycle experiments "
              f"from {in_dir}")
    elif sim_paths:
        experiments = _load_cycle_csvs(sim_paths)
        simulated = True
        print(f"[05] input: {len(experiments)} SIMULATED thermal-cycle "
              f"experiments (SIM_*.csv) from {in_dir}")
        print("[05] WARNING: SIMULATED data — code-chain validation only, "
              "NOT experimental data.")
    elif config.get("simulation", {}).get("enabled", False):
        experiments = _simulate_thermal_cycles(config)
        simulated = True
    else:
        raise FileNotFoundError(
            "No thermal-cycle CSVs found and simulation.enabled is false. "
            f"Populate {in_dir} first.")

    # 2. Assemble per-experiment feature arrays + optional process params
    exp_ids = [e[0] for e in experiments]
    feats = [_select_features(c, feature_columns) for _, c in experiments]
    pp_map = None
    if config.get("process_params", {}).get("enabled", False):
        pp_map = {eid: _process_params_for(config, eid) for eid in exp_ids}

    experiments_feat = list(zip(exp_ids, feats))

    # 3. Split by experiment (explicit, so we can emit per-sample exp ids)
    tr_i, va_i, te_i = split_experiment_indices(
        len(experiments_feat),
        val_ratio=float(dcfg.get("val_ratio", 0.15)),
        test_ratio=float(dcfg.get("test_ratio", 0.15)),
        seed=int(config.get("seed", 42)))

    def subset(idx):
        return [experiments_feat[i] for i in idx]

    Xtr, ytr, idtr = _build_split(subset(tr_i), input_len, pred_len, step, pp_map)
    Xva, yva, idva = _build_split(subset(va_i), input_len, pred_len, step, pp_map)
    Xte, yte, idte = _build_split(subset(te_i), input_len, pred_len, step, pp_map)

    feature_dim = int(Xtr.shape[-1])
    mag_test = np.array([_magnetic_group_for(config, e) for e in idte])

    # 4. Persist bundle
    out_dir = config["paths"]["processed_samples"]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, dcfg.get("samples_file", "window_samples.npz"))
    np.savez(
        out_path,
        X_train=Xtr, y_train=ytr, exp_train=idtr,
        X_val=Xva, y_val=yva, exp_val=idva,
        X_test=Xte, y_test=yte, exp_test=idte,
        mag_group_test=mag_test,
        feature_dim=np.int64(feature_dim),
        feature_columns=np.array(feature_columns),
        input_len=np.int64(input_len),
        pred_len=np.int64(pred_len),
        simulated=np.bool_(simulated),
    )

    print(f"[05] feature_columns={feature_columns} feature_dim={feature_dim} "
          f"(process_params={'on' if pp_map else 'off'})")
    print(f"[05] experiments -> train={len(tr_i)} val={len(va_i)} test={len(te_i)}")
    print(f"[05] samples     -> train={len(Xtr)} val={len(Xva)} test={len(Xte)}")
    print(f"[05] output: {out_path}")
    if simulated:
        print("[05] NOTE: bundle is SIMULATED — for code-chain testing only, "
              "NOT experimental data.")


if __name__ == "__main__":
    main()
