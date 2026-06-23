# Magnetic-Field-Assisted Laser Cladding — XTherm Temperature Field Processing & Thermal Cycle Prediction

## Project Purpose

Process temperature field data from a **Xiris VXIR-3000** camera (WeldStudio Pro `.xtherm`
format), convert raw digital counts to degrees Celsius, extract temperature-field features,
and assess **cladding quality** under **with / without magnetic field** conditions.

> **Formal experiment design — single source of truth:** 19 conditions
> (`C1`, `C2`, `R1`–`R17`), a 3-factor 3-level **Box–Behnken Design**, each with 3 repeated
> single tracks `T1/T2/T3` → **57 independent temperature-field samples**. See
> `docs/actual_experiment_plan.md` and the machine-readable `configs/experiments.yaml`.
> Canonical raw data: `D:/WenDuChang-data-repo/raw_xtherm`.
>
> **Current phase (now): process parameters → single-track temperature-field features.**
> ```
> process parameters → 57 independent single-track temperature fields
>   → per-track thermal-field features (single track = processing unit)
>   → per-condition aggregation over T1/T2/T3: mean / std / CV (condition = aggregation unit)
>   → 19 condition-level thermal responses → response surface & magnetic-field effect
> ```
> The three tracks are repeated experiments — their raw frames are **never** concatenated.
> This phase does **not** slice tracks, measure cross-sections, build quality labels, or run
> quality classification.
>
> **Later phase (not now): temperature-field features + cross-section quality labels → quality prediction.**
> Once cross-section quality labels exist, thermal-field features are joined with them to
> train traditional ML models (Random Forest / SVM / KNN / Logistic Regression). The
> section-level ML scripts **13–16** (and the label-dependent steps of `10–12`) are **kept
> but currently NOT run** — do not delete them. See `docs/ml_quality_assessment.md`.
>
> **LSTM baseline is optional**, kept for when enough data is available for deep sequence
> modeling. TCN / Transformer / LSTM-TCN remain guarded skeletons (`NotImplementedError`)
> and are not implemented. See `docs/model_design.md`.

## Physical calibration & current feature scope

**Spatial calibration (formal, user-confirmed): 150.2 px = 5 mm → `0.0332889481` mm/px**
(`pixel_area_mm2 = 0.0011081541` mm², measured along the **Y/vertical** axis). Single source
of truth: `configs/physical_calibration.yaml`, loaded/validated by
`src/config/physical_calibration.py`. The legacy pilot value `0.03128` mm/px (95.9 px = 3 mm)
is **not** used for formal processing; X is **assumed equal** to Y (X/Y anisotropy not yet
verified). **Image geometry confirmed:** `scan_axis = y`, melt pool moves toward the image
**top** (`image_scan_direction = upward`, array row index decreasing, `physical_to_array_y_sign
= -1`), no rotation/flip. **Frame rate confirmed `52 fps`** (user setting, not `session.xml`);
effective frames = `frames[1:]` (startup frame 1 excluded). **Camera valid range 300–1800 °C**
(outside → masked & reported; raw never modified; >1800 °C is `above_range`, ≥6500 °C is
`hard_saturation`). Working distance **300 mm**, laser spot **1 mm**, defocus **+14 mm**,
powder-feed setpoint **40 g/min** (actual not measured). Emissivity/transmission `not_recorded`
→ results are the **infrared apparent temperature field**, not absolute surface temperature.
Details: `docs/physical_calibration_and_process_parameters_to_confirm.md`.

**Computable now** (apparent-temperature, frame rate available): valid-band statistics (°C);
robust peak **P99.9** (°C) within band; above-range / hard-saturation pixel counts & ratios;
high-temperature **area (mm²)**; 700/800 °C isotherm **width (mm)**; hot-zone bounding-box
(mm); **scan-direction & transverse** gradients (°C/mm); **signed** center offset (mm);
left/right asymmetry; cooling rate (°C/s), dwell time (s), temperature AUC (°C·s), scan
distance per frame (mm), scan duration (s).

**Still required before formal feature extraction** (not run here): unified ROI confirmation,
invalid-pixel masking, valid-range masking, final feature-definition review.

**Unified ROI strategy — evaluated (read-only).** `scripts/03a_evaluate_roi_strategy.py` locates
the main melt-pool hot region across all 57 tracks (`frames[1:]`, 700 °C envelope / 800 °C core,
above-range & hard-saturation handled spatially without touching the raw data) and compares three
analysis options. The legacy ROI (top=200,left=0,h=300,w=600) is **not** usable (700-envelope min
coverage 99.71%); the recommended fixed global ROI is **(175,86)→(495,334) = 320×248 px** (100% core
& envelope coverage), and the recommended strategy is **global ROI + a 192×208 px moving tracking
window**. It writes tables + JSON + 12 QC figures under `results/` and **no ROI matrices**; formal
ROI cropping waits for user confirmation. See `docs/roi_strategy_evaluation.md`.

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
- The WeldStudio `.xtherm` export is a **binary temperature matrix** whose layout has been
  verified empirically: `56-byte header + 640×512 little-endian uint16`, Celsius = raw / 10.
  Script `02b_convert_xtherm_binary_to_npy.py` converts it to a stacked N × H × W `.npy`
  (all parameters in `configs/default.yaml` → `xtherm_binary`). `src/io/xtherm_reader.py`
  remains interface-only; exported `.npy` / `.csv` / `.h5` matrices are still accepted directly.

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

All workflows share the same 01→03 preprocessing.

**A0. Binary `.xtherm` import (real camera data)** —
`raw_xtherm/dataset/*.xtherm → 02b → exported/npy/dataset.npy → 02 → 03 → 10`

```powershell
python scripts/01_check_raw_data.py --config configs/default.yaml
python scripts/02b_convert_xtherm_binary_to_npy.py --config configs/default.yaml
python scripts/02_convert_exported_to_npy.py --config configs/default.yaml
python scripts/03_extract_roi.py --config configs/default.yaml
python scripts/10_extract_thermal_field_features.py --config configs/default.yaml
```

Script `02b` reads the binary `.xtherm` frames (read-only), stacks them into
`data/exported/npy/dataset.npy` (N × H × W float32 **Celsius**) plus
`dataset_meta.json`, then the normal `02 → 03 → 10` chain continues. Because
`02b` output is already Celsius, `data.exported_is_celsius` must stay `true`.

**A. ML quality assessment (MAIN LINE, small samples)** — `01 → 02 → 03 → 10 → 11 → 12`

```powershell
python scripts/01_check_raw_data.py --config configs/default.yaml
python scripts/02_convert_exported_to_npy.py --config configs/default.yaml
python scripts/03_extract_roi.py --config configs/default.yaml
python scripts/10_extract_thermal_field_features.py --config configs/default.yaml
python scripts/11_build_ml_quality_dataset.py --config configs/default.yaml
python scripts/12_train_ml_quality_model.py --config configs/default.yaml
```

Extracts one thermal-field feature row per experiment, merges with substrate
cross-section **quality labels** (`data/metadata/quality_labels.csv`, local), and trains
traditional ML models. Quality labels come from substrate cross-section measurements — see
`docs/quality_label_template.md` and `docs/ml_quality_assessment.md`.

**B. Temporal feature analysis** — `01 → 02 → 03 → 04 → 09`

```powershell
python scripts/01_check_raw_data.py --config configs/default.yaml
python scripts/02_convert_exported_to_npy.py --config configs/default.yaml
python scripts/03_extract_roi.py --config configs/default.yaml
python scripts/04_extract_thermal_cycle.py --config configs/default.yaml
python scripts/09_analyze_temporal_features.py --config configs/default.yaml
```

Produces `results/tables/temporal_features.csv` and temporal feature figures. See
`docs/temporal_analysis.md`.

**C. LSTM deep-learning prediction (OPTIONAL, for larger datasets)** —
`01 → 02 → 03 → 04 → 05 → 06 → 07 → 08`

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

The LSTM baseline is best suited to **larger** datasets; with few experiments prefer
workflow A.

**D. Section-level ML quality prediction** — `01 → 02b → 02 → 03 → 13 → 14 → 15 → 16`
*(LATER PHASE — scripts 13–16 are currently **NOT run**; kept in the repo, not deleted. See the current/later-phase note at the top.)*

```powershell
python scripts/01_check_raw_data.py --config configs/default.yaml
python scripts/02b_convert_xtherm_binary_to_npy.py --config configs/default.yaml
python scripts/02_convert_exported_to_npy.py --config configs/default.yaml
python scripts/03_extract_roi.py --config configs/default.yaml
python scripts/13_extract_local_section_features.py --config configs/default.yaml
python scripts/14_build_section_ml_dataset.py --config configs/default.yaml
python scripts/15_train_section_quality_model.py --config configs/default.yaml
python scripts/16_plot_section_ml_results.py --config configs/default.yaml
```

The ML **sample unit is a cross-section position** (e.g. `R01_T1_S1`), **not** a
temperature-field frame. Each sample = process parameters + **local**
thermal-field features around that section's frame window.

- `13` maps each section position to a local frame window and extracts `local_*`
  features (reusing the `thermal_field_features` math).
- `14` merges local features with cross-section quality labels and derives
  `dilution_rate` / `aspect_ratio` / `wetting_angle_avg` / `wetting_angle_diff`
  and a Good/Bad label.
- `15` trains regression + classification models with **GroupKFold /
  LeaveOneGroupOut on `experiment_id`** — sections of the same experiment are
  **never** split across train and test (no random shuffling, no leakage).
- `16` plots prediction scatter, confusion matrices, feature importance, and an
  input-set comparison.

Three input sets — **process_only**, **thermal_only**, **fused** — are compared
to test whether local thermal-field features improve section-quality prediction.
`section_plan.csv` and `section_quality_labels.csv` are **local** files (see
`docs/section_level_ml_dataset.md` and `docs/section_quality_label_template.md`);
the scripts give clear guidance instead of fabricating data when they are absent.

> **SIMULATED data caveat:** if inputs are `SIM_*` (the script-05 fallback), every output
> table and figure is tagged `SIMULATED`. Such results only validate the code chain and must
> **never** be cited as experimental conclusions. Archive `SIM_*` before importing real data
> (see `docs/after_experiment_checklist.md`).

## Pipeline Scripts

| Script | Input | Output |
|--------|-------|--------|
| `01_check_raw_data.py` | data directories | console report (no `.xtherm` parsing) |
| `02b_convert_xtherm_binary_to_npy.py` | `data/raw_xtherm/dataset/*.xtherm` (binary) | `data/exported/npy/dataset.npy` + `dataset_meta.json` (float32 °C) |
| `02_convert_exported_to_npy.py` | `data/exported/{npy,csv,h5}` | `data/processed/matrix/*.npy` (float32 °C) |
| `03_extract_roi.py` | `data/processed/matrix` | `data/processed/roi/*.npy` |
| `03a_evaluate_roi_strategy.py` | `data/processed/matrix` (read-only) | ROI-strategy tables + JSON + QC figures under `results/` (NO ROI `.npy`) |
| `04_extract_thermal_cycle.py` | `data/processed/roi` | `data/processed/thermal_cycle/*.csv` |
| `05_build_window_dataset.py` | thermal-cycle CSVs (or SIMULATED) | `data/processed/samples/window_samples.npz` |
| `06_train_model.py` | samples `.npz` | `best_lstm.pt`, `normalizer.npz`, `training_log.csv`, `used_config.yaml` |
| `07_evaluate_model.py` | samples + checkpoint | `lstm_metrics.csv`, `lstm_predictions.csv`, (group metrics) |
| `08_plot_results.py` | tables | figures (`.png` + `.pdf`) |
| `09_analyze_temporal_features.py` | `data/processed/thermal_cycle` | `temporal_features.csv` + temporal feature figures |
| `10_extract_thermal_field_features.py` | `data/processed/roi` | `thermal_field_features.csv` (one row/experiment) |
| `11_build_ml_quality_dataset.py` | features + `quality_labels.csv` (local) | `ml_quality_dataset.csv` |
| `12_train_ml_quality_model.py` | `ml_quality_dataset.csv` | `ml_quality_metrics.csv`, `ml_quality_predictions.csv`, `ml_feature_importance.csv` + figures |
| `13_extract_local_section_features.py` | `section_plan.csv` (local) + `data/processed/roi` | `local_section_features.csv` (one row/section) |
| `14_build_section_ml_dataset.py` | local features + `section_quality_labels.csv` (local) | `section_ml_dataset.csv` |
| `15_train_section_quality_model.py` | `section_ml_dataset.csv` | `section_ml_{regression,classification}_{metrics,predictions}.csv`, `section_ml_feature_importance.csv` |
| `16_plot_section_ml_results.py` | section ML tables | `results/figures/section_ml/*.png/.pdf` |

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
