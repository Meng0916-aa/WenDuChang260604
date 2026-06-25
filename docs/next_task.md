# Next Task

The next task is to create and validate a machine-readable formal ROI strategy
configuration while keeping ROI generation and formal feature extraction closed.

## Expected New Files

- `configs/roi_strategy.yaml`
- `src/config/roi_strategy.py`
- `tests/test_roi_strategy_config.py`

## Expected Updated Files

- `README.md`
- `configs/formal_pipeline.yaml`
- `configs/default.yaml`
- `docs/roi_strategy_evaluation.md`
- `docs/formal_pipeline.md`

## Explicitly Forbidden In The Next Task

- Generate formal ROI matrices.
- Extract formal temperature-field features.
- Modify `data/`.
- Modify `results/`.
- Overwrite the 57 full-frame matrices.
- Use `--overwrite`.
- Touch `scripts/02d_conversion_report.py`.
- Touch `项目进展与下一步数据提取说明.docx`.
- Commit or push.

The task must stop if repository evidence conflicts with the planned ROI
configuration.
