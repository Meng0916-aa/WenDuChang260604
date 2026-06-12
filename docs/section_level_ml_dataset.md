# Section-Level ML Quality Dataset

This is the **section-level** machine-learning main line. Its sample unit is a
**cross-section position** along a cladding track — for example `R01_T1_S1`
(experiment `R01`, track `T1`, section `S1`). Each sample combines the process
parameters of that track with the **local thermal-field features** measured
around the moment the camera passed that section, and is labelled with the
cross-section forming quality.

Pipeline: **`01 → 02b → 02 → 03 → 13 → 14 → 15 → 16`**

| Step | Script | Role |
|------|--------|------|
| 13 | `13_extract_local_section_features.py` | map each section position to a local frame window and extract `local_*` features |
| 14 | `14_build_section_ml_dataset.py` | merge local features with section quality labels, derive ratios + Good/Bad |
| 15 | `15_train_section_quality_model.py` | GroupKFold / LeaveOneGroupOut training of regression + classification models |
| 16 | `16_plot_section_ml_results.py` | pred-vs-true, confusion matrices, feature importance, input-set comparison |

> This complements — it does **not** replace — the experiment-level main line
> (`10 → 11 → 12`), which keeps one feature row per experiment.

## 1. Why NOT one frame = one ML sample

A run records hundreds of frames. Adjacent frames are almost identical (same
melt pool a few milliseconds apart). If each frame were an independent sample:

- the effective sample size would be hugely overstated;
- frames from the same run would land in **both** train and test, so the model
  would essentially memorise the run and report fake-high accuracy (data
  leakage);
- there is only **one** destructive cross-section measurement per cut location,
  so a per-frame label does not even exist.

## 2. Why a cross-section position is the right sample unit

Forming quality (height, width, penetration, wetting angle, defects) is measured
by **cutting and polishing the substrate at specific positions**. Each polished
cross-section is one physical, independently-labelled observation. The matching
ML sample is therefore "the thermal state **local to that position**", not the
whole run and not a single frame.

## 3. Mapping a section position to frames

For an ROI matrix with `N` frames and a track of effective length
`travel_distance_mm`, a section at `section_position_mm` maps to:

```
frame_center = round(section_position_mm / travel_distance_mm * (N - 1))
frame_start  = max(0,     frame_center - window_half_frames)
frame_end    = min(N - 1, frame_center + window_half_frames)
```

The local window `frames[frame_start : frame_end+1]` feeds
`src/features/local_section_features.py`, which reuses the experiment-level
`thermal_field_features` math (same thresholds, `pixel_size_mm`,
`frame_rate_fps`) and renames the outputs with a `local_` prefix:

```
local_peak_temperature      local_high_temp_area_mean/max
local_mean_temperature      local_haz_width_mean/max
local_temperature_fluctuation  local_gradient_mean/max
local_cooling_rate_mean     local_cooling_rate_max_abs
local_center_offset_mean/max/std
local_dwell_time            local_temperature_auc
```

A section position outside `[0, travel_distance_mm]` is a hard error. A window
with fewer than 3 frames is allowed but warns (cooling-rate / fluctuation
features get noisy).

## 4. `section_plan.csv` (LOCAL — never committed)

One row per cross-section. Put it at `data/metadata/section_plan.csv`
(git-ignored). Columns:

```
sample_id,experiment_id,track_id,section_id,roi_file,section_position_mm,travel_distance_mm,laser_power_W,scan_speed_mm_min,powder_feed_g_min,magnetic_field_mT,frame_rate_fps,window_half_frames,notes
```

| Field | Meaning |
|-------|---------|
| `sample_id` | unique section id, e.g. `R01_T1_S1` (**must be unique**) |
| `experiment_id` | experiment / run id, e.g. `R01_P300_V400_B60` (the GROUP key) |
| `track_id` | cladding track id, e.g. `T1` |
| `section_id` | section id within the track, e.g. `S1` |
| `roi_file` | ROI matrix file under `data/processed/roi/`, e.g. `dataset.npy` |
| `section_position_mm` | distance of the section from the track start (mm) |
| `travel_distance_mm` | effective track length (mm); default `30` if blank |
| `laser_power_W` | laser power (process parameter) |
| `scan_speed_mm_min` | scan speed |
| `powder_feed_g_min` | powder feed rate |
| `magnetic_field_mT` | magnetic field strength |
| `frame_rate_fps` | effective frame rate of this track |
| `window_half_frames` | half-window in frames; default from config if blank |
| `notes` | free text |

Example:

```csv
sample_id,experiment_id,track_id,section_id,roi_file,section_position_mm,travel_distance_mm,laser_power_W,scan_speed_mm_min,powder_feed_g_min,magnetic_field_mT,frame_rate_fps,window_half_frames,notes
R01_T1_S1,R01_P300_V400_B60,T1,S1,R01_P300_V400_B60_T1.npy,6,30,300,400,40,60,52,10,
R01_T1_S2,R01_P300_V400_B60,T1,S2,R01_P300_V400_B60_T1.npy,12,30,300,400,40,60,52,10,
R01_T1_S3,R01_P300_V400_B60,T1,S3,R01_P300_V400_B60_T1.npy,18,30,300,400,40,60,52,10,
```

(Numbers are illustrative placeholders — fill with your real plan. `roi_file`
may be `dataset.npy` if all tracks share one exported ROI matrix.)

## 5. `section_quality_labels.csv`

See `docs/section_quality_label_template.md`. Keyed by the same `sample_id`.
Script 14 derives the ratios and the Good/Bad label automatically when they are
not supplied.

## 6. Avoiding data leakage — GroupKFold is mandatory

All sections of the same `experiment_id` form **one group**. Splits use
`GroupKFold` or `LeaveOneGroupOut` on `experiment_id` (config
`section_ml.cv`), so **no experiment's sections ever appear in both train and
test**. There is no random `train_test_split` and no shuffling of sections
across the group boundary anywhere in the pipeline. Scores are computed on
out-of-fold predictions (`cross_val_predict` with the group splitter). Training
is refused if fewer than 2 experiment groups exist — you cannot hold out an
experiment for testing otherwise.

Inside every fold the model is a `Pipeline(StandardScaler → estimator)`, so the
scaler is fit on the training fold only (no normalization leakage either).

## 7. Three input sets compared

Script 15 trains each task with three feature sets (config
`section_ml.input_sets`):

- **process_only** — laser power, scan speed, powder feed, magnetic field;
- **thermal_only** — the 15 `local_*` thermal-field features;
- **fused** — process parameters **+** local thermal-field features.

Comparing them answers the core research question: *do local thermal-field
features add predictive power over process parameters alone?*

## 8. Outputs (under `results/`, LOCAL only)

- `results/tables/local_section_features.csv` — one row per section (script 13)
- `results/tables/section_ml_dataset.csv` — merged features + labels (script 14)
- `results/tables/section_ml_regression_metrics.csv` — MAE/RMSE/R² per
  `(input_set, target, model)`
- `results/tables/section_ml_regression_predictions.csv` — per-sample predictions
- `results/tables/section_ml_classification_metrics.csv` —
  accuracy/precision/recall/f1 per `(input_set, model)`
- `results/tables/section_ml_classification_predictions.csv`
- `results/tables/section_ml_feature_importance.csv` — Random Forest importances
- `results/figures/section_ml/*.png/.pdf` — pred-vs-true, confusion matrices,
  feature importance, `input_set_performance_comparison`

Every metrics row records `n_samples`, `n_groups`, and `cv_method` so the
group-CV setup is auditable.

## 9. How to phrase it in the paper

> Each cross-section position is treated as one machine-learning sample,
> described by the deposition process parameters together with local
> thermal-field features extracted from the temperature-field frames acquired as
> the camera passed that position. Forming quality (dilution rate, aspect ratio,
> wetting angle, and a Good/Bad classification) is predicted from these inputs.
> To prevent information leakage between highly-correlated sections of the same
> experiment, model evaluation uses experiment-grouped cross-validation
> (GroupKFold / leave-one-experiment-out); sections of any one experiment never
> appear in both training and test folds. Process-only, thermal-only, and fused
> feature sets are compared to quantify the contribution of the local
> thermal-field features.
