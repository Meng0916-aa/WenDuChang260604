# Next Task

The next task is to design the formal temperature-field feature dictionary and
extraction contract, without running feature extraction or generating data.

## Scope

- Feature names.
- Mathematical definitions.
- Units.
- Calculation region.
- Time aggregation method.
- Invalid-pixel handling.
- Feature split between the fixed global ROI and the tracking window.
- T1/T2/T3 aggregation rules.
- Output table structure.

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

The task must stop if repository evidence conflicts with the planned feature
contract.
