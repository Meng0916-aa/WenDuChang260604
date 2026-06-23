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
- **Canonical raw-data source** for all formal batch processing is the independent data repo
  **`D:/WenDuChang-data-repo/raw_xtherm`** (`raw_data_root` in `configs/experiments.yaml`).
  Do NOT use `data/raw_xtherm` as the formal source; exclude the early-test copy
  `data/raw_xtherm/dataset`.
- Scan speed is always **mm/min** (never mm/s). `sample_id = condition_id + "_" + track_id`.
  The per-track map is built locally to `data/metadata/experiment_master.csv` by
  `scripts/00_build_experiment_master.py` (LOCAL ONLY; `*.csv` is git-ignored). Unknown
  fields (`frame_rate_fps`, `effective_start_frame`, `effective_end_frame`, `scan_axis`,
  `scan_direction`, `plate_id`) are left BLANK, never guessed.

## Physical Calibration (formal — single source of truth)
- Formal spatial scale is **150.2 px = 5 mm → `pixel_size_x_mm = pixel_size_y_mm =
  0.0332889481` mm/px**, `pixel_area_mm2 = 0.0011081541` mm². User-confirmed
  (`calibration_id: formal_150p2px_5mm`). Source: `configs/physical_calibration.yaml`,
  loaded/validated via `src/config/physical_calibration.py`. Formal ROI/feature code
  MUST read the pixel size from there — NEVER from `configs/default.yaml`'s legacy
  `pixel_size_mm: 0.03128` (95.9 px = 3 mm), which is `legacy_pilot_only` and disabled
  for formal processing.
- X and Y currently share one scale (`isotropic_scaling_assumed: true`); X/Y anisotropy is
  NOT yet verified. Do not run formal processing with two active pixel sizes at once.
- `frame_rate_fps` is **unconfirmed** (historical 52 vs 1000 conflict) → kept null. All time
  axes use the **frame index**, never seconds. `require_frame_rate()` raises for time-domain
  features (cooling rate / dwell / AUC / scan-distance-per-frame) until a real fps is given.
- Temperature: saturated/sentinel pixels at the uint16 ceiling **6553.5 °C** are masked for
  analysis only; raw `.xtherm` and converted matrices are never modified. Prefer the robust
  **P99.9** peak over raw max.

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