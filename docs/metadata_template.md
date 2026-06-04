# Experiment Metadata Template

Record one row per experiment run. This file is **Markdown** (not `.csv`) so it
is tracked by Git — the repo's `.gitignore` excludes `*.csv`, which would hide a
CSV metadata file. Keep the real, filled-in table here (or another `.md`/`.yaml`
file under `data/metadata/`).

> Do not commit raw data or results — only this metadata description.

## Fields

| Field | Meaning / example |
|-------|-------------------|
| `sample_id` | unique run id, e.g. `B100_02` |
| `file_name` | exported matrix file, e.g. `B100_02.npy` |
| `magnetic_field_mT` | field strength in millitesla, e.g. `100` (`0` = none) |
| `magnetic_group` | `without_magnetic_field` or `with_magnetic_field` |
| `laser_power_W` | laser power in watts, e.g. `1500` |
| `scan_speed_mm_s` | scan speed in mm/s, e.g. `5.0` |
| `powder_feed_g_min` | powder feed rate in g/min, e.g. `12.0` |
| `frame_rate_fps` | camera frame rate in fps, e.g. `100` |
| `aoi_height` | acquisition area-of-interest height in pixels |
| `aoi_width` | acquisition area-of-interest width in pixels |
| `temperature_unit` | `Celsius` or `counts` |
| `exported_is_celsius` | `true` if already Celsius, `false` if raw counts |
| `notes` | free text (anomalies, calibration, operator, etc.) |

## Template table (fill in real values)

| sample_id | file_name | magnetic_field_mT | magnetic_group | laser_power_W | scan_speed_mm_s | powder_feed_g_min | frame_rate_fps | aoi_height | aoi_width | temperature_unit | exported_is_celsius | notes |
|-----------|-----------|-------------------|----------------|---------------|-----------------|-------------------|----------------|------------|-----------|------------------|---------------------|-------|
| B0_01   | B0_01.npy   | 0   | without_magnetic_field | [..] | [..] | [..] | [..] | [..] | [..] | Celsius | true  | [..] |
| B0_02   | B0_02.npy   | 0   | without_magnetic_field | [..] | [..] | [..] | [..] | [..] | [..] | Celsius | true  | [..] |
| B100_01 | B100_01.npy | 100 | with_magnetic_field    | [..] | [..] | [..] | [..] | [..] | [..] | Celsius | true  | [..] |
| B100_02 | B100_02.npy | 100 | with_magnetic_field    | [..] | [..] | [..] | [..] | [..] | [..] | Celsius | true  | [..] |

## How this maps to the config

- `magnetic_group` + `sample_id` → `magnetic_field_groups.*.experiment_ids` in
  `configs/default.yaml`.
- `exported_is_celsius` → `data.exported_is_celsius` (must match how the matrices
  were exported; see `docs/real_data_import.md`).
- `laser_power_W`, `scan_speed_mm_s`, ... → optional model inputs via
  `process_params.by_experiment` (keyed by `sample_id`) when
  `process_params.enabled: true`.
