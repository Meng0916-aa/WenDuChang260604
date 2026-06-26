# Next Task

The next task is to implement pure-function frame-level and track-level thermal
feature computation modules, using only synthetic arrays for unit tests.

## Scope

- Implement feature primitives from `configs/thermal_feature_contract.yaml`.
- Use `configs/xtherm_format.yaml`, `configs/roi_strategy.yaml`, and
  `configs/physical_calibration.yaml` as configuration inputs.
- Test frame-level and track-level functions with small synthetic arrays only.
- Preserve the contract that one track produces one feature row, but do not
  create any formal output table in this task.

## Explicitly Forbidden In The Next Task

- Read the 57 formal `.npy` matrices.
- Generate formal ROI matrices.
- Generate formal feature CSV files.
- Modify `data/`.
- Modify `results/`.
- Run ROI evaluation.
- Create a batch extraction script.
- Use `--overwrite`.
- Touch `scripts/02d_conversion_report.py`.
- Touch `项目进展与下一步数据提取说明.docx`.
- Commit or push unless explicitly requested.

The task must stop if repository evidence conflicts with the feature contract.
