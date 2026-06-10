# ML-Based Cladding Quality Assessment

This is the project's **current main line**: extract temperature-field features
from infrared (xtherm-exported) data and train traditional machine-learning
models against substrate cross-section quality labels.

## 1. Why not deep learning (for now)

The real experiment set is **small** (a handful of runs). Deep sequence models
(LSTM/TCN/Transformer) have many parameters and need large, diverse datasets;
with only a few experiments they overfit and their reported accuracy does not
generalize. The LSTM baseline is kept in the repo but is **optional** — suitable
later when enough data is available.

## 2. Why traditional machine learning

With few samples, the robust approach is:

1. Compress each experiment's temperature field into a **small set of physically
   meaningful features** (one feature vector per experiment).
2. Train compact models (Random Forest, SVM, KNN, Logistic Regression) with
   strong cross-validation (LeaveOneOut).
3. Use Random Forest **feature importance** for interpretation.

This trades raw capacity for stability and interpretability, which suits small
samples and an engineering-analysis goal.

## 3. Temperature-field features (per experiment)

Extracted by `src/features/thermal_field_features.py` / script 10:

| Feature | Symbol | Physical meaning |
|---------|--------|------------------|
| peak_temperature | T_max | hottest melt-pool state |
| mean_temperature | T_avg | overall melt-pool thermal state |
| max/mean_high_temp_area | A_T | melting extent |
| max/mean_gradient | G_T | solidification driving force / thermal stress |
| max/mean_cooling_rate | R_c | microstructure refinement / solidification speed |
| haz_width_max/mean | W_HAZ | heat-diffusion range |
| temperature_fluctuation | sigma_T | melt-pool thermal stability |
| center_offset_mean/max/std | D_T | temperature-field symmetry |
| dwell_time_above_threshold, temperature_auc | — | thermal exposure |

## 4. Quality labels (substrate cross-section)

Labels come from cutting and measuring the deposited bead cross-section. See
`docs/quality_label_template.md`. Key derived quantities:

```
dilution_rate     = D / (D + H)
aspect_ratio      = W / H
wetting_angle_avg = (theta_left + theta_right) / 2
```

Recommended Good/Bad rule (configurable in `quality_labels.quality_rule`):
`Good` if `0.30 <= dilution_rate <= 0.50` and `3 <= aspect_ratio <= 6` and
`30 <= wetting_angle_avg <= 55`, else `Bad`.

## 5. One experiment = one sample (no per-frame leakage)

Each experiment is aggregated into **one** feature row. Adjacent frames within a
run are highly correlated; if frames were treated as independent samples, the
same run would appear in both training and test folds, leaking information and
massively inflating apparent accuracy. Aggregating per experiment prevents this.

## 6. Workflow

Minimal small-sample ML line (ROI → features → labels → models):

```
01 -> 02 -> 03 -> 10 -> 11 -> 12
```

With the temporal-curve analysis included (optional, for Chapter-3 figures):

```
01 -> 02 -> 03 -> 04 -> 09 -> 10 -> 11 -> 12
```

Windows PowerShell:

```powershell
conda activate pytorch
$env:KMP_DUPLICATE_LIB_OK="TRUE"

python scripts/01_check_raw_data.py --config configs/default.yaml
python scripts/02_convert_exported_to_npy.py --config configs/default.yaml
python scripts/03_extract_roi.py --config configs/default.yaml
python scripts/10_extract_thermal_field_features.py --config configs/default.yaml
python scripts/11_build_ml_quality_dataset.py --config configs/default.yaml
python scripts/12_train_ml_quality_model.py --config configs/default.yaml
```

(Step 11 needs the local label file `data/metadata/quality_labels.csv`.)

## 7. Interpreting feature importance

Script 12 fits a Random Forest on all features and writes
`results/tables/ml_feature_importance.csv` plus a bar chart. Higher importance =
the feature the model relied on more to separate Good/Bad. Use it to discuss
**which temperature-field characteristics most influence cladding quality**
(e.g. whether cooling rate or HAZ width dominates).

## 8. Small-sample honesty

With very few experiments, all metrics are **exploratory only**. Scripts warn
when `n_samples < 5`. Do not present such results as proof of generalization;
they indicate trends and guide which features/measurements to collect more of.
SIMULATED inputs (script-05/-10 fallbacks) are tagged `SIMULATED` and validate
the code chain only — never cite them as experimental conclusions.

## 9. Outputs (local, not uploaded)

- `results/tables/thermal_field_features.csv` (script 10)
- `results/tables/ml_quality_dataset.csv` (script 11)
- `results/tables/ml_quality_metrics.csv`, `ml_quality_predictions.csv`,
  `ml_feature_importance.csv` (script 12)
- `results/figures/ml_quality_confusion_matrix.*`, `ml_feature_importance.*`,
  `ml_regression_prediction.*` (script 12)

`data/` and `results/` are git-ignored.
