# Magnetic-Field-Assisted Laser Cladding — XTherm Temperature Field Processing & Thermal Cycle Prediction

## Project Purpose

Process temperature field data from a **Xiris VXIR-3000** camera (WeldStudio Pro `.xtherm`
format), convert raw digital counts to degrees Celsius, and build deep-learning models for
multi-step **thermal cycle prediction** under **with / without magnetic field** conditions.

> **Currently runnable model: LSTM baseline only.**
> TCN / Transformer / LSTM-TCN exist as guarded skeletons (`NotImplementedError`) and are
> not yet implemented. See `docs/model_design.md`.

## Environment

This project runs PyTorch **only** in the conda environment named `pytorch`
(PyTorch 2.11.0+cu128, CUDA available). **Do not** run `pip install torch`, and **do not**
install PyTorch into the `base` environment.

On Windows PowerShell, before running any PyTorch script:

```powershell
conda activate pytorch
$env:KMP_DUPLICATE_LIB_OK="TRUE"
```

(The `KMP_DUPLICATE_LIB_OK` variable avoids an OpenMP duplicate-runtime crash on Windows.)

## Data Format

| Stage      | Format               | Shape     | Unit                   |
|------------|----------------------|-----------|------------------------|
| Raw        | `.xtherm`            | binary    | raw digital counts     |
| Exported   | `.npy` / `.csv` / `.h5` | N × H × W | Celsius (`raw / 10`)   |
| Processed  | `.npy`               | N × H × W | float32 Celsius        |
| Thermal cycle | `.csv`            | N rows    | float32 Celsius curves |

- **N** = number of frames, **H, W** = spatial dimensions.
- The internal `.xtherm` binary layout is **not** parsed yet (see `src/io/xtherm_reader.py`).
  The runnable path starts from exported `.npy` / `.csv` / `.h5` matrices or thermal-cycle CSVs.

Full details: `docs/data_format.md`.

## Directory Layout

```
data/
  raw_xtherm/              <- original .xtherm files (NEVER delete/move/modify)
  exported/{csv,npy,h5}/   <- exported temperature matrices
  processed/{matrix,roi,thermal_cycle,samples}/
  metadata/
configs/                   <- YAML configuration (default.yaml)
src/                       <- source modules
scripts/                   <- numbered pipeline scripts
tests/                     <- unit + smoke tests
results/{figures,tables,logs,checkpoints}/
docs/                      <- documentation
```

## Full Run (Windows PowerShell)

```powershell
conda activate pytorch
$env:KMP_DUPLICATE_LIB_OK="TRUE"

python scripts/01_check_raw_data.py --config configs/default.yaml
python scripts/02_convert_exported_to_npy.py --config configs/default.yaml
python scripts/03_extract_roi.py --config configs/default.yaml
python scripts/04_extract_thermal_cycle.py --config configs/default.yaml
python scripts/05_build_window_dataset.py --config configs/default.yaml
python scripts/06_train_model.py --config configs/default.yaml
python scripts/07_evaluate_model.py --config configs/default.yaml
python scripts/08_plot_results.py --config configs/default.yaml
python scripts/09_analyze_temporal_features.py --config configs/default.yaml
```

If no real exported data exists, step **05** generates a small **SIMULATED** dataset so the
training/evaluation/plotting chain (05→08) runs end-to-end. Simulated results only prove the
code chain works — they are **not** experimental conclusions.

## Recommended Workflows

Two independent analyses share the same 01→04 preprocessing:

**A. Temporal feature analysis** — `01 → 02 → 03 → 04 → 09`

```powershell
python scripts/01_check_raw_data.py --config configs/default.yaml
python scripts/02_convert_exported_to_npy.py --config configs/default.yaml
python scripts/03_extract_roi.py --config configs/default.yaml
python scripts/04_extract_thermal_cycle.py --config configs/default.yaml
python scripts/09_analyze_temporal_features.py --config configs/default.yaml
```

Produces `results/tables/temporal_features.csv` and temporal feature figures
(`temporal_feature_overview.*`, `temporal_curve_<id>.*`) under `results/figures/`.
See `docs/temporal_analysis.md`.

**B. LSTM prediction modeling** — `01 → 02 → 03 → 04 → 05 → 06 → 07 → 08`

```powershell
python scripts/01_check_raw_data.py --config configs/default.yaml
python scripts/02_convert_exported_to_npy.py --config configs/default.yaml
python scripts/03_extract_roi.py --config configs/default.yaml
python scripts/04_extract_thermal_cycle.py --config configs/default.yaml
python scripts/05_build_window_dataset.py --config configs/default.yaml
python scripts/06_train_model.py --config configs/default.yaml
python scripts/07_evaluate_model.py --config configs/default.yaml
python scripts/08_plot_results.py --config configs/default.yaml
```

> **SIMULATED data caveat:** if inputs are `SIM_*.csv` (the script-05 fallback), every
> output table and figure is tagged `SIMULATED`. Such results only validate the code chain
> and must **never** be cited as experimental conclusions. Archive `SIM_*.csv` before
> importing real data (see `docs/after_experiment_checklist.md`).

## Pipeline Scripts

| Script | Input | Output |
|--------|-------|--------|
| `01_check_raw_data.py` | data directories | console report (no `.xtherm` parsing) |
| `02_convert_exported_to_npy.py` | `data/exported/{npy,csv,h5}` | `data/processed/matrix/*.npy` (float32 °C) |
| `03_extract_roi.py` | `data/processed/matrix` | `data/processed/roi/*.npy` |
| `04_extract_thermal_cycle.py` | `data/processed/roi` | `data/processed/thermal_cycle/*.csv` |
| `05_build_window_dataset.py` | thermal-cycle CSVs (or SIMULATED) | `data/processed/samples/window_samples.npz` |
| `06_train_model.py` | samples `.npz` | `best_lstm.pt`, `normalizer.npz`, `training_log.csv`, `used_config.yaml` |
| `07_evaluate_model.py` | samples + checkpoint | `lstm_metrics.csv`, `lstm_predictions.csv`, (group metrics) |
| `08_plot_results.py` | tables | figures (`.png` + `.pdf`) |
| `09_analyze_temporal_features.py` | `data/processed/thermal_cycle` | `temporal_features.csv` + temporal feature figures |

## Output Files

```
results/checkpoints/best_lstm.pt                       <- best model
results/checkpoints/normalizer.npz                     <- train-set normalization stats
results/logs/training_log.csv                          <- per-epoch train/val loss
results/logs/used_config.yaml                          <- exact config used for the run
results/tables/lstm_metrics.csv                        <- overall test metrics (Celsius)
results/tables/lstm_predictions.csv                    <- per-sample/step predictions
results/tables/lstm_metrics_by_experiment.csv          <- per-experiment metrics
results/tables/lstm_metrics_by_magnetic_group.csv      <- per-group metrics (if grouping set)
results/tables/temporal_features.csv                   <- temporal features per experiment (script 09)
results/figures/*.png, *.pdf                           <- curves, metric charts, temporal figures
```

All metrics are computed on **inverse-normalized Celsius** values.

## Tests

```powershell
conda activate pytorch
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python -m pytest tests
```

If `pytest` is not installed (do **not** auto-install it), run each test directly, e.g.:

```powershell
python tests/test_temperature_scale.py
python tests/test_window_dataset.py
python tests/test_metrics.py
python tests/test_normalize.py
python tests/test_lstm_forward.py
python tests/test_pipeline_smoke.py
```

Tests use small **SIMULATED** data for code validation only.

## Key Rules

1. **Never** delete, move, or modify `data/raw_xtherm/`.
2. All processed temperature data is **float32 Celsius**, shape **N × H × W**.
3. Train/val/test split by **experiment ID**, never by shuffling adjacent frames.
4. Pseudo-color images are **not** quantitative — use raw temperature matrices only.
5. All parameters come from a YAML config under `configs/`.
6. All generated outputs land under `results/`.
7. Run PyTorch only in the `pytorch` conda env; never `pip install torch`.
