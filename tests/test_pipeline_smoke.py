"""
Smoke test for the minimal pipeline 05 -> 06 -> 07 -> 08.

Runs each script as a subprocess against a TINY temporary config that writes
to a temp directory (so real data/ and results/ are never touched). Uses a
small SIMULATED dataset and 2 training epochs purely to validate the code
chain — the numbers produced are meaningless.

SIMULATED data only — for code validation, not experimental results.
"""

import os
import sys
import csv
import shutil
import tempfile
import subprocess

import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS = os.path.join(_ROOT, "scripts")


def _tiny_config(tmp: str) -> dict:
    """Build a minimal config rooted at a temp directory (absolute paths)."""
    def d(*parts):
        p = os.path.join(tmp, *parts)
        os.makedirs(p, exist_ok=True)
        return p

    return {
        "seed": 0,
        "paths": {
            "raw_xtherm": d("data", "raw_xtherm"),
            "exported_csv": d("data", "exported", "csv"),
            "exported_npy": d("data", "exported", "npy"),
            "exported_h5": d("data", "exported", "h5"),
            "processed_matrix": d("data", "processed", "matrix"),
            "processed_roi": d("data", "processed", "roi"),
            "processed_thermal_cycle": d("data", "processed", "thermal_cycle"),
            "processed_samples": d("data", "processed", "samples"),
            "metadata": d("data", "metadata"),
            "results_figures": d("results", "figures"),
            "results_tables": d("results", "tables"),
            "results_logs": d("results", "logs"),
            "results_checkpoints": d("results", "checkpoints"),
        },
        "data": {"temperature_scale": 0.1, "exported_is_celsius": False},
        "roi": {"enabled": True, "bounds": [0, 0, 10, 10]},
        "thermal_cycle": {"center_average_radius": 3,
                          "hot_zone_threshold_celsius": 800.0},
        "dataset": {
            "input_window": 20, "predict_window": 5, "step": 1,
            "val_ratio": 0.25, "test_ratio": 0.25, "shuffle_train": True,
            "feature_columns": ["tmax"], "samples_file": "window_samples.npz",
        },
        "normalization": {"enabled": True, "method": "standard", "eps": 1e-8},
        "model": {"name": "lstm", "hidden_dim": 16, "num_layers": 1,
                  "dropout": 0.0, "lstm": {"bidirectional": False}},
        "training": {"loss": "mse", "batch_size": 64, "epochs": 2,
                     "learning_rate": 0.01, "weight_decay": 0.0,
                     "patience": 5, "gradient_clip": 1.0, "device": "auto"},
        "evaluation": {"metrics": ["rmse", "mae", "waveform_similarity"],
                       "predictions_file": "lstm_predictions.csv",
                       "metrics_file": "lstm_metrics.csv",
                       "metrics_by_group_file": "lstm_metrics_by_magnetic_group.csv",
                       "group_by_experiment": True},
        "magnetic_field_groups": {
            "with_magnetic_field": {"experiment_ids": [], "label": "with_B"},
            "without_magnetic_field": {"experiment_ids": [], "label": "without_B"},
        },
        "visualization": {"dpi": 80, "save_png": True, "save_pdf": True,
                          "num_curve_samples": 2, "figsize": [6, 3]},
        "simulation": {"enabled": True, "n_experiments": 6, "n_frames": 120,
                       "seed": 0},
    }


def _run(script: str, cfg_path: str):
    env = dict(os.environ)
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    r = subprocess.run(
        [sys.executable, os.path.join(_SCRIPTS, script), "--config", cfg_path],
        capture_output=True, text=True, env=env, cwd=_ROOT,
    )
    if r.returncode != 0:
        raise AssertionError(
            f"{script} failed (code {r.returncode})\nSTDOUT:\n{r.stdout}\n"
            f"STDERR:\n{r.stderr}")
    return r.stdout


def test_pipeline_05_to_08():
    tmp = tempfile.mkdtemp(prefix="wdc_smoke_")
    try:
        cfg = _tiny_config(tmp)
        cfg_path = os.path.join(tmp, "smoke.yaml")
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        _run("05_build_window_dataset.py", cfg_path)
        assert os.path.exists(os.path.join(
            cfg["paths"]["processed_samples"], "window_samples.npz"))

        _run("06_train_model.py", cfg_path)
        assert os.path.exists(os.path.join(
            cfg["paths"]["results_checkpoints"], "best_lstm.pt"))
        assert os.path.exists(os.path.join(
            cfg["paths"]["results_checkpoints"], "normalizer.npz"))
        assert os.path.exists(os.path.join(
            cfg["paths"]["results_logs"], "training_log.csv"))
        assert os.path.exists(os.path.join(
            cfg["paths"]["results_logs"], "used_config.yaml"))

        _run("07_evaluate_model.py", cfg_path)
        metrics_csv = os.path.join(cfg["paths"]["results_tables"], "lstm_metrics.csv")
        preds_csv = os.path.join(cfg["paths"]["results_tables"], "lstm_predictions.csv")
        assert os.path.exists(metrics_csv)
        assert os.path.exists(preds_csv)
        # predictions.csv has the required columns
        with open(preds_csv, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f))
        for col in ("sample_index", "step", "y_true", "y_pred", "error"):
            assert col in header, f"missing column {col}"

        _run("08_plot_results.py", cfg_path)
        figs = os.listdir(cfg["paths"]["results_figures"])
        assert any(x.endswith(".png") for x in figs)
        assert any(x.endswith(".pdf") for x in figs)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_pipeline_05_to_08()
    print("test_pipeline_smoke: OK (simulated/code-validation only)")
