# Formal 57-track Pipeline — entry of record

This document defines the **single active pipeline** for the magnetic-field-assisted
laser-cladding thermal-field study. The machine-readable entry is
`configs/formal_pipeline.yaml`; it references `configs/experiments.yaml`
(process design) and `configs/physical_calibration.yaml` (spatial / temperature /
geometry / frame rate) by **relative path only**. `configs/default.yaml` is
**legacy** and is not part of this pipeline.

> No formal ROI matrices and no formal temperature-field features have been
> generated yet. This document does not claim otherwise.

## Current formal workflow (one line)

```
experiment design + physical metadata
  -> 57 single-track .xtherm conversion        (scripts/02c_batch_convert_tracks.py)
  -> conversion QC                             (scripts/02c --qc / 02d report)
  -> ROI strategy evaluation                   (scripts/03a_evaluate_roi_strategy.py)
  -> USER confirms fixed ROI / tracking window     [decision gate — not automated]
  -> formal ROI or analysis-window generation  (planned)
  -> 57 single-track temperature-field features (planned)
  -> T1/T2/T3 in-plate repeat aggregation       (planned)
  -> 19-condition response-surface analysis     (planned)
```

## Current formal entry points (scripts)

| Stage | Script | Status |
|-------|--------|--------|
| Per-track metadata map | `scripts/00_build_experiment_master.py` | done (local CSV) |
| Metadata audit tables | `scripts/00b_build_metadata_audit.py` | done (local) |
| 57-track `.xtherm` → matrix | `scripts/02c_batch_convert_tracks.py` | done |
| Conversion report | `scripts/02d_conversion_report.py` | done |
| ROI strategy evaluation | `scripts/03a_evaluate_roi_strategy.py` | done (read-only) |
| Formal ROI / window generation | *(planned)* | not started |
| Temperature-field features | *(planned)* | not started |
| Condition aggregation + RSM | *(planned)* | not started |

## Completed stages

- 19 conditions / 57 single-track metadata (`experiment_master.csv`, local).
- Spatial calibration + camera/optics/substrate metadata freeze.
- 57 full temperature matrices converted (`data/processed/matrix/*.npy`, local).
- Conversion QC.
- **ROI strategy evaluation** (candidates + analysis-coordinate plan), read-only.

## Not yet executed

- Formal ROI matrix generation.
- Formal temperature-field feature extraction.
- Condition-level response-surface fitting.
- Section-level quality-label modelling (later stage; scripts 13–16).

## Data-root resolution (portable; no machine-absolute path in the repo)

The raw-data root is resolved by `src/config/path_resolution.py` in priority:

1. CLI `--raw-data-root <path>`
2. environment variable `WENDUCHANG_DATA_ROOT`
3. `configs/local.yaml` → `paths.raw_data_root` (git-ignored; copy
   `configs/local.example.yaml`)
4. `configs/experiments.yaml` → `raw_data_root` (currently `null`)
5. otherwise a clear error (never a silent guess)

The legacy `dataset` pilot path is never returned. Windows and POSIX paths are
both accepted.

## Formal ROI strategy status

`evaluated_not_finalized`. The evaluation (`docs/roi_strategy_evaluation.md`,
`results/qc/roi/roi_strategy_summary.json`) recommends a fixed global ROI
**(top=175, left=86, bottom=495, right=334) = 320×248 px** plus an extent-based
**256×216 px** tracking window (100% coverage of the 700 °C envelope and 800 °C
core, 0 clipped frames). Formal ROI cropping waits for explicit user
confirmation.

## Configs that must NOT be mixed

| Config | Role | Formal use |
|--------|------|-----------|
| `configs/formal_pipeline.yaml` | active pipeline entry | **yes** |
| `configs/experiments.yaml` | 19-condition design, track list | yes |
| `configs/physical_calibration.yaml` | spatial / temperature / geometry / fps | yes (only pixel-size source) |
| `configs/local.yaml` | machine paths (git-ignored) | yes (local only) |
| `configs/default.yaml` | **legacy pilot** (`config_status.role: legacy_pilot`) | **no** |

Formal code refuses `configs/default.yaml`, its disabled legacy ROI, its legacy
pixel size, and the early-test `dataset` path — see
`src/config/formal_config.py`.

## Hard constraints (this stage)

No formal ROI `.npy`, no temperature-field feature extraction, no ML, no
response-surface fitting, no re-conversion or modification of the 57 matrices,
no modification of raw `.xtherm`.
