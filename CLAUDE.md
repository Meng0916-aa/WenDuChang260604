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
- Do NOT hand-write an unverifiable `.xtherm` binary parser. `src/io/xtherm_reader.py`
  stays interface-only (raises `NotImplementedError`) until real parsing is implemented
  via the Xiris WeldSDK / official export.

## Current Runnable Scope
- Only the **LSTM baseline** (`src/models/lstm.py`) is trainable.
- `tcn.py`, `transformer.py`, `lstm_tcn.py` are guarded skeletons; `train.py` raises
  `NotImplementedError` for any `model.name` other than `lstm`.
- Metrics are computed on inverse-normalized **Celsius** values. Normalization statistics
  are fit on the TRAINING split only and saved to `results/checkpoints/normalizer.npz`.
- SIMULATED data (from `simulation` in the config) is for code-chain validation ONLY and
  must never be reported as an experimental conclusion.