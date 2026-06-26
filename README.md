# Magnetic-Field-Assisted Laser Cladding — XTherm Temperature Field Processing & Thermal Cycle Prediction

## Project Purpose

Process temperature field data from a **Xiris VXIR-3000** camera (WeldStudio Pro `.xtherm`
format), convert raw digital counts to degrees Celsius, extract temperature-field features,
and assess **cladding quality** under **with / without magnetic field** conditions.

> **Formal experiment design — single source of truth:** 19 conditions
> (`C1`, `C2`, `R1`–`R17`), a 3-factor 3-level **Box–Behnken Design**, each with 3 repeated
> single tracks `T1/T2/T3` → **57 single-track temperature-field samples**. See
> `docs/actual_experiment_plan.md` and the machine-readable `configs/experiments.yaml`.
> The 57 tracks are independent processing units, but they are not 57 independent
> plate-level replicates.
> The formal raw-data root is resolved locally through the portable
> path-resolution rules described below; no machine-specific absolute path
> is stored in the repository.
>
> **Current phase (now): process parameters → single-track temperature-field features.**
> ```
> process parameters → 57 single-track temperature fields
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

**Formal feature contract — designed, not executed.**
`configs/thermal_feature_contract.yaml` and `docs/formal_feature_dictionary.md`
define the first 15 Core thermal-field features, QC-only fields, Secondary
candidates, and Rejected features. The contract exists only as a design artifact:
formal feature extraction remains disabled, and no formal feature table has been
generated.

**Unified ROI strategy — evaluated (read-only).**

`scripts/03a_evaluate_roi_strategy.py` locates the cleaned main melt-pool hot region across all 57 tracks using the effective frames `frames[1:]`, with a 700 °C envelope and an 800 °C core. Above-range and hard-saturation pixels are handled spatially without modifying the original temperature matrices.

The legacy fixed ROI (`top=200`, `left=0`, `height=300`, `width=600`) is **not suitable**, because its minimum 700 °C envelope coverage is only 99.71%.

The evaluated fixed global ROI uses the half-open array ranges `rows[175:495]` and `cols[86:334]`. Its dimensions are:

* `global_roi_height_px = 320`
* `global_roi_width_px = 248`

This fixed global ROI provides 100% coverage of both the cleaned 700 °C envelope and the 800 °C core.

The evaluated moving tracking window uses:

* `tracking_window_width_px = 256`
* `tracking_window_height_px = 216`
* `tracking_window_width_mm ≈ 8.522`
* `tracking_window_height_mm ≈ 7.190`

The tracking window provides 100% coverage of both the cleaned 700 °C envelope and the 800 °C core, with zero clipped frames.

The recommended strategy is therefore **a fixed global ROI plus a 256-px-wide × 216-px-high moving tracking window**. The global ROI preserves absolute position and trajectory information, while the tracking window supports local melt-pool morphology and temperature-field analysis.

`configs/roi_strategy.yaml` is now the machine-readable record of this evaluated
strategy. Its presence does **not** activate ROI generation: formal ROI generation
and formal feature extraction remain disabled, and there are still **no formal ROI
matrices** and **no formal temperature-field feature tables**.

The evaluation writes tables, JSON summaries, and QC figures under `results/`, but generates **no formal ROI matrices**.

See `docs/roi_strategy_evaluation.md`.


## Environment

Install the base dependencies with pip:

```powershell
pip install -r requirements.txt
```

**PyTorch is managed separately and is NOT in `requirements.txt`.** This project runs PyTorch
**only** in an existing conda environment named `pytorch` (a CUDA build). Installing it via
`requirements.txt` could overwrite that CUDA build, so:

- **Do not** run `pip install torch`, and **do not** install PyTorch into the `base` environment.
- Install/verify PyTorch yourself only when GPU training is needed, and check
  `torch.cuda.is_available()`.
- The current 57-track formal pipeline (metadata → conversion → conversion QC → ROI strategy
  evaluation) does **not** require PyTorch.

On Windows PowerShell, before running any PyTorch script:

```powershell
conda activate pytorch
$env:KMP_DUPLICATE_LIB_OK="TRUE"
```

(The `KMP_DUPLICATE_LIB_OK` variable avoids an OpenMP duplicate-runtime crash on Windows.)
Full details: `docs/environment_setup.md`.

### Local machine paths (portable)

The raw-data root is **not** hard-coded in the repo. It is resolved by priority: CLI
`--raw-data-root` > env `WENDUCHANG_DATA_ROOT` > `configs/local.yaml` (git-ignored; copy
`configs/local.example.yaml`) > `configs/experiments.yaml` (`null`) > clear error. See
`src/config/path_resolution.py` and `docs/formal_pipeline.md`.

## Data Format

| Stage      | Format               | Shape     | Unit                   |
|------------|----------------------|-----------|------------------------|
| Raw        | `.xtherm`            | binary    | raw digital counts     |
| Exported   | `.npy` / `.csv` / `.h5` | N × H × W | Celsius (`raw / 10`)   |
| Processed  | `.npy`               | N × H × W | float32 Celsius        |
| Thermal cycle | `.csv`            | N rows    | float32 Celsius curves |

- **N** = number of frames, **H, W** = spatial dimensions.
- The WeldStudio `.xtherm` export is a **binary temperature matrix** whose layout has been
  verified empirically: `56-byte header + 640×512 little-endian uint16`, with
  `temperature_C = raw_count × 0.1`.
- `scripts/02b_convert_xtherm_binary_to_npy.py` and
  `scripts/02c_batch_convert_tracks.py` share the verified parser in
  `src/conversion/xtherm_binary.py`; the formal 57-track workflow uses `02c`.
- The historical `xtherm_binary` block in `configs/default.yaml` is retained only for
  legacy path compatibility and is not authoritative.
- `configs/xtherm_format.yaml` is the authoritative source for the verified binary layout,
  temperature scaling, camera-valid temperature range, and conversion-QC thresholds.
- `configs/physical_calibration.yaml` remains the authoritative source for spatial
  calibration, imaging geometry, process metadata, and the physical interpretation used
  during formal feature extraction.
- `src/io/xtherm_reader.py` remains interface-only; exported `.npy`, `.csv`, and `.h5`
  matrices are still accepted by the legacy conversion utility.

Full details: `docs/data_format.md`.

## Directory Layout

```
data/
  raw_xtherm/              <- original .xtherm files (NEVER delete/move/modify)
  exported/{csv,npy,h5}/   <- exported temperature matrices
  processed/{matrix,roi,thermal_cycle,samples}/
  metadata/
configs/                   <- YAML config: formal_pipeline.yaml (ACTIVE) +
                              experiments.yaml + physical_calibration.yaml;
                              default.yaml = LEGACY; local.yaml = machine paths (git-ignored)
src/                       <- source modules
scripts/                   <- numbered pipeline scripts
tests/                     <- unit + smoke tests
results/{figures,tables,logs,checkpoints}/
docs/                      <- documentation
```

## Current formal workflow (the ONLY recommended pipeline)

The active pipeline entry of record is `configs/formal_pipeline.yaml`
(it references `configs/experiments.yaml` + `configs/physical_calibration.yaml`).
`configs/default.yaml` is **legacy** and is not used here. Full detail:
`docs/formal_pipeline.md`.

```text
experiment design + physical metadata
  ├─> 57 single-track .xtherm conversion         (scripts/02c_batch_convert_tracks.py)
  ├─> per-track conversion QC                    (generated by scripts/02c_batch_convert_tracks.py)
  ├─> ROI strategy evaluation                    (scripts/03a_evaluate_roi_strategy.py)
  ├─> machine-readable ROI strategy config       (configs/roi_strategy.yaml)
  ├─> machine-readable feature contract          (configs/thermal_feature_contract.yaml)
  ├─> USER activates ROI/window generation       [decision gate]
  ├─> formal ROI or analysis-window generation   (planned)
  ├─> 57 single-track temperature-field features (planned)
  ├─> T1/T2/T3 in-plate repeat aggregation       (planned)
  └─> 19-condition response-surface analysis      (planned)
```

Track-level conversion and its per-track QC are handled by `scripts/02c_batch_convert_tracks.py`. No independent aggregate conversion-report script is currently incorporated into the version-controlled formal pipeline.


**Current formal entry points** (already completed/evaluated; rerun only for an explicit task):

Use any Python environment that satisfies `requirements.txt`. The existing `pytorch` conda
environment may also be used, but Torch is not required for these three stages.

```powershell
# Optional when using the existing project environment:
# conda activate pytorch
# $env:KMP_DUPLICATE_LIB_OK="TRUE"

# 1. Build the per-track metadata map (local CSV; raw-data root resolved portably)
python scripts/00_build_experiment_master.py

# 2. Convert the 57 single tracks (read-only on raw .xtherm) and generate per-track QC
python scripts/02c_batch_convert_tracks.py --master-csv data/metadata/experiment_master.csv --all --qc

# 3. Evaluate the fixed-ROI and tracking-window strategy (writes no ROI matrices)
python scripts/03a_evaluate_roi_strategy.py
```

**Done:** 19-condition / 57-track metadata · spatial calibration + camera/optics/substrate
metadata · 57 full temperature matrices · conversion QC · ROI strategy evaluation ·
machine-readable ROI strategy configuration · formal thermal-feature contract.

**Not yet executed (planned):** formal ROI matrix generation · formal temperature-field feature
extraction · condition-level response-surface fitting · section-level quality-label modelling.

> No formal ROI matrices and no formal temperature-field features have been generated yet.

## Legacy and future-stage workflows

> **Not part of the current 57-track formal pipeline. Do not run unless explicitly working on
> legacy pilot data or the later section-level quality-prediction stage.** These use the legacy
> `configs/default.yaml` (`config_status.role: legacy_pilot`), the early-test `dataset.npy`
> entry, the old `03_extract_roi.py`, the LSTM baseline, and scripts 13–16. The early-test
> `dataset` path and old ROI are disabled for formal processing.

- **Legacy A0 — binary pilot import:** `01 → 02b (dataset.npy) → 02 → 03 → 10`. Early-test
  `data/raw_xtherm/dataset` only; the formal pipeline converts per-track via `02c` instead.
- **Legacy A — ML quality assessment:** `01 → 02 → 03 → 10 → 11 → 12` (needs cross-section
  quality labels; later stage).
- **Legacy B — temporal feature analysis:** `01 → 02 → 03 → 04 → 09`.
- **Legacy C — LSTM prediction (optional):** `01 → 02 → 03 → 04 → 05 → 06 → 07 → 08` (requires
  the separately-managed `pytorch` env; SIMULATED fallback in step 05 is code-validation only,
  never an experimental conclusion).
- **Future-stage D — section-level ML quality prediction:** `13 → 14 → 15 → 16` (cross-section
  sample unit; GroupKFold/LeaveOneGroupOut on `experiment_id`). Kept in the repo, **not run** in
  the current phase. See `docs/section_level_ml_dataset.md`, `docs/ml_quality_assessment.md`.

> **SIMULATED data caveat:** if inputs are `SIM_*` (the script-05 fallback), every output table
> and figure is tagged `SIMULATED` — code-chain validation only, never an experimental
> conclusion. See `docs/after_experiment_checklist.md`.

## Pipeline Scripts

Only scripts marked **Formal**, **Formal utility**, or **Formal evaluation** belong to the
current 57-track thermal-field pipeline. Scripts marked **Legacy**, **Later-stage**, or
**Future section-level stage** must not be run unless that workflow is explicitly activated.

| Script | Input | Output | Status |
|---|---|---|---|
| `00_build_experiment_master.py` | `configs/experiments.yaml` and `configs/physical_calibration.yaml` | `data/metadata/experiment_master.csv` containing 57 formal single-track records | **Formal** |
| `00b_build_metadata_audit.py` | Formal experiment, calibration, process, and plate metadata | Metadata audit tables under `results/tables/` | **Formal utility** |
| `01_check_raw_data.py` | Legacy raw-data directories | Console integrity report without formal `.xtherm` parsing | **Legacy / optional** |
| `02b_convert_xtherm_binary_to_npy.py` | A legacy or single-directory `.xtherm` dataset | One temperature-matrix `.npy` file and its metadata | **Legacy-compatible utility** |
| `02c_batch_convert_tracks.py` | The 57 formal track directories defined by `experiment_master.csv` | 57 track-level temperature matrices, metadata files, and per-track conversion QC | **Formal** |
| `02_convert_exported_to_npy.py` | `data/exported/{npy,csv,h5}` | `data/processed/matrix/*.npy` | **Legacy / optional** |
| `03_extract_roi.py` | Historical pilot temperature matrices | Historical ROI matrices under `data/processed/roi/` | **Legacy / optional** |
| `03a_evaluate_roi_strategy.py` | The 57 full-frame temperature matrices under `data/processed/matrix/` in read-only mode | ROI-strategy tables, JSON summaries, and QC figures under `results/`; no formal ROI matrices | **Formal evaluation** |
| `04_extract_thermal_cycle.py` | Historical ROI matrices | Thermal-cycle CSV files under `data/processed/thermal_cycle/` | **Legacy / optional** |
| `05_build_window_dataset.py` | Historical thermal-cycle CSV files or simulated data | `data/processed/samples/window_samples.npz` | **Legacy / optional** |
| `06_train_model.py` | Historical window-sample dataset | LSTM model checkpoint, normalization data, and training log | **Legacy / optional** |
| `07_evaluate_model.py` | Historical samples and trained checkpoint | LSTM metrics, predictions, and grouped evaluation results | **Legacy / optional** |
| `08_plot_results.py` | Historical result tables | Historical result figures in PNG and PDF formats | **Legacy / optional** |
| `09_analyze_temporal_features.py` | Historical thermal-cycle data | Temporal-feature tables and exploratory figures | **Legacy / optional** |
| `10_extract_thermal_field_features.py` | Historical ROI matrices | Historical thermal-field feature table with one row per experiment | **Later-stage / not current** |
| `11_build_ml_quality_dataset.py` | Thermal-field features and local `quality_labels.csv` | `ml_quality_dataset.csv` | **Later-stage / not current** |
| `12_train_ml_quality_model.py` | `ml_quality_dataset.csv` | Quality-model metrics, predictions, feature importance, and figures | **Later-stage / not current** |
| `13_extract_local_section_features.py` | Local `section_plan.csv` and temperature-field data | `local_section_features.csv` with one row per section | **Future section-level stage** |
| `14_build_section_ml_dataset.py` | Local section features and `section_quality_labels.csv` | `section_ml_dataset.csv` | **Future section-level stage** |
| `15_train_section_quality_model.py` | `section_ml_dataset.csv` | `section_ml_{regression,classification}_{metrics,predictions}.csv` and `section_ml_feature_importance.csv` | **Future section-level stage** |
| `16_plot_section_ml_results.py` | Section-level machine-learning result tables | Figures under `results/figures/section_ml/` in PNG and PDF formats | **Future section-level stage** |

## Output Files

### Current formal pipeline outputs

```text
data/metadata/experiment_master.csv                    <- local 57-row track metadata map
data/processed/matrix/<sample_id>.npy                  <- full-frame float32 temperature matrix
data/processed/matrix_meta/<sample_id>.json            <- per-track conversion metadata
results/qc/conversion/<sample_id>/                     <- per-track conversion QC
results/tables/<qc_summary>.csv                        <- conversion-QC summary selected by --qc-summary
results/qc/roi/                                        <- ROI-strategy QC figures and JSON
results/tables/roi_bbox_by_track.csv                   <- 57-track ROI geometry summary
results/tables/roi_exception_list.csv                  <- ROI exceptions requiring review
results/tables/roi_repeatability_summary.csv           <- 19-condition spatial repeatability
results/tables/tracking_window_coverage_summary.csv    <- tracking-window coverage evaluation
```

The current default value of `--qc-summary` is
`results/tables/pilot_conversion_qc.csv` for backward compatibility. Aggregate conversion
exception and repeatability reports are not listed as repository-generated formal outputs
until a tracked aggregate-report implementation is finalized.

The formal ROI evaluation is read-only: **no formal ROI matrices and no formal feature table
have been generated yet**. Files under `data/` and `results/` are local and git-ignored.

### Legacy or later-stage outputs

```text
results/checkpoints/best_lstm.pt                       <- legacy optional LSTM model
results/checkpoints/normalizer.npz                     <- legacy train-set normalization stats
results/logs/training_log.csv                          <- legacy per-epoch train/validation loss
results/logs/used_config.yaml                          <- exact legacy configuration used
results/tables/lstm_metrics.csv                        <- legacy LSTM test metrics
results/tables/lstm_predictions.csv                    <- legacy LSTM predictions
results/tables/lstm_metrics_by_experiment.csv          <- legacy per-experiment metrics
results/tables/lstm_metrics_by_magnetic_group.csv      <- legacy per-group metrics
results/tables/temporal_features.csv                   <- legacy temporal features
results/figures/*.png, *.pdf                           <- legacy or later-stage figures
```

Legacy model metrics are computed on inverse-normalized Celsius values. They are not part of
the current 57-track formal analysis unless that workflow is explicitly activated.

## Tests

The current formal metadata, XTherm conversion, conversion-QC, and ROI-evaluation stages do not require PyTorch. However, the complete repository test suite also includes legacy LSTM and window-dataset tests that import Torch. Therefore, run the complete test suite in the existing `pytorch` conda environment:

```powershell
conda activate pytorch
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python -m pytest tests -q
```

The suite contains formal configuration, metadata, conversion, ROI-strategy, legacy-compatibility, and small synthetic-fixture tests. Synthetic inputs are used only for code validation and must never be interpreted as experimental results.

Do not install PyTorch into the `base` environment, and do not add Torch back to `requirements.txt`.


## Key Rules

1. **Never** delete, move, or modify the canonical raw `.xtherm` data.
2. Full-frame processed temperature matrices remain **float32 Celsius**, shape **N × H × W**.
3. `T1/T2/T3` are in-plate repeated single tracks; their raw frames are never concatenated.
4. The first frame is a confirmed startup frame; formal effective frames are `frames[1:]`.
5. Pseudo-color images are not quantitative; all calculations use raw temperature matrices.
6. Formal experiment, calibration, path, and pipeline settings come from the formal YAML
   configuration chain, not from the legacy `configs/default.yaml`.
7. Camera-valid quantitative temperatures are 300–1800 °C. Above-range and hard-saturation
   pixels are masked and reported without modifying raw or processed matrices.
8. The fixed global ROI preserves absolute position; the moving tracking window supports
   local melt-pool analysis and must not erase full-frame coordinates.
9. Generated processed data belong under `data/processed/`; QC, tables, figures, logs, and
   checkpoints belong under `results/`.
10. Train/validation/test splitting for later machine-learning stages must be grouped by
    experiment or plate, never by randomly shuffling adjacent frames.
11. Run PyTorch only in a separately managed compatible environment; never install it through
    `requirements.txt`.
