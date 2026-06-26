# Project State

Baseline commit: `6ab3a19`

## Completed

- Formal experiment design: 19 conditions.
- Formal sample set: 57 single-track samples.
- T1/T2/T3 are in-plate repeated tracks, not independent plate-level repeats.
- Experiment metadata is complete.
- Physical calibration is complete.
- Formal XTherm format migration is complete.
- 57 full-frame matrices exist.
- Conversion QC is complete.
- Formal ROI machine-readable configuration is established in
  `configs/roi_strategy.yaml`.
- Formal thermal-feature dictionary and machine-readable contract are
  established in `docs/formal_feature_dictionary.md` and
  `configs/thermal_feature_contract.yaml`.
- Full test suite baseline: 233 passed in the `pytorch` conda environment.
- 57-track conversion dry-run passed.

## Evaluated But Not Enabled

- ROI strategy evaluation is complete.
- ROI strategy status: evaluated but not activated.
- Fixed global ROI: rows `[175:495]`, cols `[86:334]`.
- Recommended moving tracking window: 256 x 216 px.
- Recommended strategy: `global_roi_plus_tracking_window`.
- Formal ROI generation remains closed.
- Formal feature extraction remains closed.
- Formal feature contract status: designed but not executed.

## Not Started

- Formal ROI matrix generation.
- Formal temperature-field feature table generation.
- Condition-level response-surface analysis.
- Section quality prediction.

## Legacy Or Local Historical Artifacts

- `data/processed/roi/dataset.npy` belongs to legacy data.
- Historical feature or machine-learning tables under `results/tables/` do not
  prove that the formal pipeline has completed feature extraction or modelling.
- Local `data/` and `results/` outputs are ignored and are not repository
  source-of-truth documents.

## Protected Local Files

These local untracked files are protected and must not be touched:

- `scripts/02d_conversion_report.py`
- `项目进展与下一步数据提取说明.docx`

## Current Formal Absences

- No formal ROI matrices have been generated.
- No formal temperature-field feature table has been generated.
