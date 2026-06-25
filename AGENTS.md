# Agent Operating Rules

This file is the single formal operating rule file for Codex and other coding
agents working in this repository.

## Project Purpose

This project supports magnetic-field-assisted laser cladding infrared
temperature-field data processing:

- XTherm binary temperature-matrix conversion.
- Melt-pool temperature-field feature extraction.
- T1/T2/T3 repeated-track aggregation.
- Later response-surface analysis.

## Current Formal Pipeline State

The formal experiment has 19 process conditions. Each condition has 3 repeated
single tracks, for 57 single-track temperature-field samples total.

T1/T2/T3 are repeated single tracks on the same plate for a condition. They are
not 57 independent plate-level repeats.

Completed:

- Experiment metadata.
- Physical calibration.
- Formal XTherm format configuration.
- 57 full-frame matrix conversions.
- Conversion QC.
- ROI strategy evaluation.

Not enabled:

- Formal ROI matrix generation.
- Formal temperature-field feature extraction.
- Condition-level response-surface modelling.
- Section quality prediction.

## Authoritative Formal Configuration

Formal authoritative sources:

- `configs/formal_pipeline.yaml`
- `configs/experiments.yaml`
- `configs/physical_calibration.yaml`
- `configs/xtherm_format.yaml`

`configs/default.yaml` is legacy-only. It has no formal authority for the
57-track pipeline.

## Key Formal Parameters

- Frame rate: 52 fps.
- The first acquired frame is the startup frame and is excluded from formal
  processing; the effective Python slice is `frames[1:]`.
- Image size: 512 x 640.
- File header: 56 bytes.
- Data type: little-endian uint16.
- Temperature conversion: raw x 0.1 deg C.
- Camera valid range: 300-1800 deg C.
- 1800-6500 deg C: `above_range`.
- `>=6500` deg C: `hard_saturation`.
- Saturation value: 6553.5 deg C.
- Spatial calibration: 150.2 px = 5 mm.
- `pixel_size_y_mm = 0.0332889481`.
- `pixel_size_x_mm` currently uses the isotropic assumption.

ROI evaluation result:

- Fixed global ROI: rows `[175:495]`, cols `[86:334]`.
- Fixed global ROI height: 320 px.
- Fixed global ROI width: 248 px.
- Moving tracking window width: 256 px.
- Moving tracking window height: 216 px.
- Recommended strategy: `global_roi_plus_tracking_window`.

## Hard Constraints

- 永远不得修改、移动、删除或覆盖原始.xtherm文件。
- 未经用户明确启用数据生成任务，不得修改data/或results/。
- 不得覆盖现有57个全帧温度矩阵。
- 未经用户明确批准，不得使用--overwrite。
- 永远不得运行git add .
- 未经用户明确批准，不得commit或push。
- 不得自动安装依赖。
- 不得把legacy配置作为正式配置使用。
- 不得接触受保护的未跟踪文件。
- 多文件任务必须先给出文件级计划，再实施。
- 发现仓库证据与任务假设冲突时必须停止并报告。

Protected local untracked files:

- `scripts/02d_conversion_report.py`
- `项目进展与下一步数据提取说明.docx`

## Migration And Workflow References

- Codex must read task-relevant workflow documents under `docs/workflows/` when
  such documents exist.
- `docs/claude_to_codex_migration.md` is a migration record, not formal
  configuration.
- `.claude/` is not the current authoritative rule entry.
- Do not restore old Claude rules as active project rules.
- If a Claude skill conflicts with formal repository configuration, `AGENTS.md`,
  or `docs/decisions.md`, the formal configuration, `AGENTS.md`, and
  `docs/decisions.md` take precedence.

## Testing Environment

The complete test suite must be run in the existing `pytorch` conda
environment:

```powershell
conda activate pytorch
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python -m pytest tests -q
```

The formal metadata, conversion, and ROI-evaluation workflows themselves do not
depend on PyTorch. The complete test suite includes legacy LSTM tests, so the
complete suite requires the `pytorch` environment.
