# Experiment Protocol

> This document describes the intended data-collection and processing protocol.
> It is a template — fill in the bracketed `[...]` fields with real values once
> experiments are run. Do NOT populate it with simulated numbers.

## 1. Equipment

- Thermal camera: **Xiris VXIR-3000**, software WeldStudio Pro (`.xtherm` export).
- Process: magnetic-field-assisted laser cladding.
- Magnetic field: `[coil / magnet setup, field strength, orientation]`.

## 2. Experiment groups

Experiments are grouped for comparison:

- `without_magnetic_field` — baseline (no applied field).
- `with_magnetic_field` — field applied.

Record each run's id and group in the config (`magnetic_field_groups.*.experiment_ids`)
and/or a metadata file under `data/metadata/`.

## 3. Process parameters (per experiment)

Record at least: `[laser_power (W), scan_speed (mm/s), powder_feed, magnetic_field flag]`.
If used as model inputs, list them in `process_params.columns` and provide per-experiment
values in `process_params.by_experiment` (keyed by experiment id).

## 4. Data acquisition

- Frame rate: `[fps]`; resolution: `[H x W]`; acquisition window: `[start/stop trigger]`.
- Export each run to a temperature matrix (`.npy` / `.csv` / `.h5`) in Celsius or raw counts.
  Set `data.exported_is_celsius` accordingly so script `02` does not double-scale.

## 5. Processing pipeline

1. `01_check_raw_data.py` — verify directories / file counts.
2. `02_convert_exported_to_npy.py` — standardize to float32 Celsius `N x H x W`.
3. `03_extract_roi.py` — crop the deposition region (`roi.bounds`).
4. `04_extract_thermal_cycle.py` — extract `tmax`, `center_average`, `hot_zone_average`.
5. `05_build_window_dataset.py` — sliding-window samples, split by experiment id.
6. `06_train_model.py` — train the LSTM baseline.
7. `07_evaluate_model.py` — metrics (Celsius), predictions, group comparison.
8. `08_plot_results.py` — figures.

## 6. Splitting rule

Train / validation / test are split by **experiment id**, never by shuffling adjacent
frames. Adjacent frames from one run must never straddle two splits.

## 7. Reproducibility

- Fixed seed (`seed` in config) via `src/utils/seed.py`.
- The exact config used for each training run is copied to
  `results/logs/used_config.yaml`.
