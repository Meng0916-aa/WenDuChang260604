# Project Rules for Claude

## Data Integrity
- **NEVER** delete, move, or modify files under `data/raw_xtherm/`.
- Pseudo-color images (PNG, JPG, TIFF heat maps) are **NOT** quantitative
  temperature data. Do not use them as model input.

## Temperature Conventions
- All processed temperature data must be **float32** in **degrees Celsius**.
- Raw digital counts → Celsius: `temperature = raw_value / 10.0`.
- Data shape convention: **N × H × W** (N = frames, H = height, W = width).

## Data Splitting
- Train / validation / test splits must be done by **experiment ID (run number)**,
  not by random shuffling of individual frames.
- Adjacent frames from the same experiment must never be split across
  train/val/test.

## Configuration
- Every experiment parameter (paths, ROI bounds, model hyperparameters,
  training settings) must be read from a **YAML configuration file**
  under `configs/`.
- Hard-coded paths or magic numbers are forbidden in source code.

## Outputs
- All results go under `results/`: figures → `results/figures/`,
  tables → `results/tables/`, logs → `results/logs/`,
  checkpoints → `results/checkpoints/`.

## Code Standards
- **Modular** — each module has a single responsibility.
- **Testable** — every function should be callable in isolation.
- **Reproducible** — use `src/utils/seed.py` to fix random seeds.
- All hyperparameters and paths configurable via YAML, not hard-coded.

## Magnetic Field Comparison
- Group experiments into `with_magnetic_field` and `without_magnetic_field`.
- Evaluation produces side-by-side metrics and plots for the two groups.

Python environment rule:
This project uses the conda environment named pytorch.
Do not install PyTorch in the base environment.
Do not run pip install torch from Claude Code.
Before running PyTorch scripts on Windows, set:
KMP_DUPLICATE_LIB_OK=TRUE.
When using PowerShell:
$env:KMP_DUPLICATE_LIB_OK="TRUE"
Then run Python scripts in the active pytorch environment.

## Hard Constraints (do not violate)
- PyTorch code may ONLY be run in the conda environment named `pytorch`
  (PyTorch 2.11.0+cu128, CUDA available).
- Do NOT execute `pip install torch` (or install torch in the `base` env).
- Do NOT delete, move, or modify anything under `data/raw_xtherm/`.
- Do NOT hand-write an UNVERIFIED `.xtherm` parser. The project currently supports only
  the empirically-verified Xiris WeldStudio Temperature `.xtherm` format:
  **56-byte header + 640×512 little-endian uint16 + scale_factor 0.1** (Celsius = raw/10),
  implemented in `scripts/02b_convert_xtherm_binary_to_npy.py` (all parameters in
  `configs/default.yaml` → `xtherm_binary`). Any other source, frame size, or export mode
  must first be re-verified (file size, endianness, dimensions, and temperature range) before
  it may be parsed. `src/io/xtherm_reader.py` stays interface-only
  (raises `NotImplementedError`).

## Current Runnable Scope
- Only the **LSTM baseline** (`src/models/lstm.py`) is trainable.
- `tcn.py`, `transformer.py`, `lstm_tcn.py` are guarded skeletons; `train.py` raises
  `NotImplementedError` for any `model.name` other than `lstm`.
- Metrics are computed on inverse-normalized **Celsius** values. Normalization statistics
  are fit on the TRAINING split only and saved to `results/checkpoints/normalizer.npz`.
- SIMULATED data (from `simulation` in the config) is for code-chain validation ONLY and
  must never be reported as an experimental conclusion.

## Current Actual Experiment Design (formal — single source of truth)
- The formal experiment is **19 conditions** (`C1`, `C2`, `R1`–`R17`), a three-factor
  three-level **Box–Behnken Design** (laser power / scan speed / magnetic field). Full
  parameters: `docs/actual_experiment_plan.md` + `configs/experiments.yaml` (machine-readable).
- Each condition has **3 repeated single tracks** `T1/T2/T3` → **57 independent
  temperature-field samples** total. The three tracks are repeated experiments, NOT one
  track split in three.
- **Single track = the processing unit.** Features are extracted per track (57 rows).
- **Condition = the statistical aggregation unit.** Per-condition results aggregate the
  three tracks (mean / std / coefficient of variation), giving 19 condition-level responses.
- **NEVER concatenate** the raw frames of `T1/T2/T3` end-to-end. Each track is parsed and
  feature-extracted independently.
- **Current phase = whole-track thermal-field analysis only.** Do NOT slice tracks, measure
  cross-sections, build section quality labels, run quality classification, or run scripts
  **13–16**. Those scripts are kept but currently NOT run (later, section-quality phase).
- **Unified ROI strategy = EVALUATED (read-only), not yet applied.** `scripts/03a_evaluate_roi_strategy.py`
  (with `src/processing/hot_region_mask.py` + `src/roi/roi_evaluation.py`) evaluates ROI/window
  candidates over the 57 tracks using `frames[1:]`; it writes tables + a JSON + QC figures under
  `results/` and writes NO ROI `.npy` and never modifies raw matrices. Findings: the legacy ROI
  (top=200,left=0,h=300,w=600) is NOT usable (700-envelope min coverage 99.71%, clips `R2_T3`); the
  recommended fixed global ROI is **(top=175,left=86,bottom=495,right=334) = 320×248 px** (100%/100%
  coverage); recommended strategy is **`global_roi_plus_tracking_window`** (window 192×208 px). Do
  NOT generate the formal ROI matrices until the user confirms a strategy. Details:
  `docs/roi_strategy_evaluation.md`.
- **Formal pipeline entry = `configs/formal_pipeline.yaml`** (active; references
  `configs/experiments.yaml` + `configs/physical_calibration.yaml`). **`configs/default.yaml` is
  LEGACY** (`config_status.role: legacy_pilot`, `formal_processing_enabled: false`): its ROI is
  disabled (`enabled: false`, `legacy_not_approved`), its `dataset` entry and `pixel_size_mm:
  0.03128` are legacy-only. Formal code MUST refuse it — `src/config/formal_config.py`
  (`assert_not_legacy_default` / `reject_legacy_roi` / `assert_no_legacy_dataset_path` /
  `assert_pixel_size_from_calibration`). Formal pixel size comes ONLY from
  `physical_calibration.yaml`. README splits the one formal workflow from a fenced
  "Legacy and future-stage workflows" section. See `docs/formal_pipeline.md`.
- **Canonical raw-data source** is the independent data repo (the early-test copy
  `data/raw_xtherm/dataset` is excluded). The path is **NOT** hard-coded in the public repo:
  resolve it via `src/config/path_resolution.py` — priority CLI `--raw-data-root` > env
  `WENDUCHANG_DATA_ROOT` > `configs/local.yaml` (git-ignored; copy `configs/local.example.yaml`)
  > `configs/experiments.yaml` (`raw_data_root: null`) > clear error. Do NOT commit
  `configs/local.yaml` or a machine-absolute path in any committed config.
- **PyTorch is NOT in `requirements.txt`** (it would overwrite the conda `pytorch` CUDA build).
  Install/verify torch separately; the formal 57-track pipeline does not need it. See
  `docs/environment_setup.md`.
- **Shared temperature-mask primitives live in `src/processing/temperature_mask.py`** (the single
  canonical `build_threshold_mask` etc.). `processing` must NOT import `features`; `features`
  imports `processing` (one direction only, no cycle). `hot_region_mask` composes
  `temperature_mask`; `features.thermal_field_features.compute_high_temperature_mask` is a thin
  wrapper delegating to it.
- Scan speed is always **mm/min** (never mm/s). `sample_id = condition_id + "_" + track_id`.
  The per-track map is built locally to `data/metadata/experiment_master.csv` by
  `scripts/00_build_experiment_master.py` (LOCAL ONLY; `*.csv` is git-ignored). As of the
  metadata freeze, `frame_rate_fps` (52), the effective-frame rule, `scan_axis` (y) /
  `image_scan_direction` (upward), working distance (300 mm), defocus (+14 mm), spot (1 mm),
  powder-feed setpoint (40), substrate (316L 40×16×8) and `plate_id` (`Plate-<condition_id>`)
  are all CONFIRMED and written to every row. Only `emissivity`, `transmission`,
  `exposure_time_us`, `lens_model` (and a per-track numeric effective-end) remain BLANK —
  never guessed.

## Physical Calibration (formal — single source of truth)
- Formal spatial scale is **150.2 px = 5 mm → `pixel_size_x_mm = pixel_size_y_mm =
  0.0332889481` mm/px**, `pixel_area_mm2 = 0.0011081541` mm². User-confirmed
  (`calibration_id: formal_150p2px_5mm`). Source: `configs/physical_calibration.yaml`,
  loaded/validated via `src/config/physical_calibration.py`. Formal ROI/feature code
  MUST read the pixel size from there — NEVER from `configs/default.yaml`'s legacy
  `pixel_size_mm: 0.03128` (95.9 px = 3 mm), which is `legacy_pilot_only` and disabled
  for formal processing.
- The Y scale is **measured** (`calibration_reference_axis: y`); X is **assumed equal**
  (`isotropic_scaling_assumed: true`). X/Y anisotropy is NOT verified — never claim X and Y
  were each calibrated. Do not run formal processing with two active pixel sizes at once.
- **Image geometry CONFIRMED:** `scan_axis: y`, `transverse_axis: x`,
  `image_scan_direction: upward`, `array_scan_direction: decreasing_row_index`,
  `physical_to_array_y_sign: -1`; no rotation/flip. Array origin is image top-left, rows
  increase downward, so the melt pool's row index DECREASES over time; physical +Y = image up.
  The scan direction is user-defined — later QC may check it but must NOT override it.
- **`frame_rate_fps = 52` CONFIRMED** (`user_confirmed_experimental_setting`, NOT from
  `session.xml`). Effective-frame rule (all 57 tracks): exclude startup frame 1; effective =
  frames 2…last (1-based) = `frames[1:]` (0-based start 1); `last_available_frame` is the data
  end, NOT a slice-exclusive endpoint. `require_frame_rate()` now returns 52. (Matrices are
  NOT reprocessed in this phase.)
- **Camera valid range CONFIRMED 300–1800 °C.** Four states (raw NEVER modified/truncated):
  `<300` below_range, `300–1800` valid, `1800<T<6500` above_range, `≥6500` hard_saturation
  (uint16 ceiling 6553.5 °C). Mask-and-report only; P99.9 robust peak is computed WITHIN the
  valid band and the above-range ratio reported alongside. `2739/3085/3000–3700/6553.5 °C` are
  NOT real melt-pool temperatures. Emissivity/transmission are `not_recorded`, so results are
  the **infrared apparent temperature field**, not radiometrically-corrected absolute surface T.
- **Camera/optics & substrate CONFIRMED:** working distance **300 mm**
  (`protective_window_to_molten_pool`; the old 30 mm was an `incorrect_legacy_value`), laser
  spot **1 mm**, defocus **+14 mm** (focal plane above surface), powder-feed **setpoint 40 g/min**
  (`powder_feed_actual_g_min` stays null / `not_measured`). Substrate: **316L 40×16×8 mm**, ONE
  plate per condition (19 plates), `plate_id = Plate-<condition_id>`
  (`logical_condition_based_identifier`), 120 s in-plate inter-track cooling. T1/T2/T3 are
  in-plate repeats — condition effect and plate-to-plate variation cannot be fully separated.

## GitHub Sync Rule
Repo: https://github.com/Meng0916-aa/WenDuChang260604 (already exists; remote branch `main`).
After completing a task, sync with `tools/safe_git_sync.ps1` (or the manual flow below).
- **NEVER** use `git add .`.
- Allowed paths (the ONLY paths that may be staged):
  `README.md`, `CLAUDE.md`, `requirements.txt`, `configs`, `src`, `scripts`, `tests`,
  `docs`, `tools`, `.gitignore`, `.vscode/settings.json`.
- **NEVER** commit `data/`, `results/`, or these extensions:
  `*.xtherm *.npy *.npz *.h5 *.hdf5 *.pt *.pth *.ckpt *.onnx *.png *.jpg *.jpeg *.pdf
  *.avi *.mp4 *.csv *.xlsx *.xls`.
- Before every commit run `git status --short`; if any data/results/weights/figures/tables
  are staged, STOP and do not commit.
- If there is nothing to commit, output exactly: `No source/document changes to commit.`