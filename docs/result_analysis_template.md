# Results Analysis Template (Chapter 3)

> **Template only.** Every numeric cell below is a placeholder `[...]`. Fill these in
> **only** with results computed from **real experimental data**. Results produced from
> SIMULATED data (the fallback in script `05`) prove the code chain runs and must **never**
> be written into this chapter as conclusions.

## 3.1 Experimental setup summary

- Number of experiments: `[N_total]` (`[N_with_B]` with field, `[N_without_B]` without).
- Process parameters: `[table of laser power / scan speed / ...]`.
- Data split (by experiment id): train `[..]`, val `[..]`, test `[..]`.

## 3.2 Model and training configuration

- Model: LSTM baseline (`input_len = [..]`, `pred_len = [..]`, `hidden_dim = [..]`,
  `num_layers = [..]`, bidirectional `[true/false]`).
- Normalization: standard (z-score), statistics fit on the training split only.
- Reference: `results/logs/used_config.yaml` for the exact configuration.

## 3.3 Overall prediction accuracy (test set, Celsius)

Source: `results/tables/lstm_metrics.csv`.

| Metric | Value |
|--------|-------|
| RMSE (°C) | `[..]` |
| MAE (°C) | `[..]` |
| Waveform similarity | `[..]` |

## 3.4 With- vs. without-magnetic-field comparison

Source: `results/tables/lstm_metrics_by_magnetic_group.csv`.
(If empty, magnetic group metadata is not yet available.)

| Group | RMSE (°C) | MAE (°C) | Waveform sim. | Samples |
|-------|-----------|----------|---------------|---------|
| without_B | `[..]` | `[..]` | `[..]` | `[..]` |
| with_B    | `[..]` | `[..]` | `[..]` | `[..]` |

Discussion: `[interpret the effect of the magnetic field on prediction accuracy and on the
physical thermal cycle — only from real data]`.

## 3.5 Per-experiment breakdown

Source: `results/tables/lstm_metrics_by_experiment.csv`. Highlight best/worst runs and any
outliers: `[..]`.

## 3.6 Qualitative curves

Figures from `results/figures/` (`prediction_curve_sample_*`, `metrics_overview`,
`metrics_by_magnetic_group`). Describe how well predicted curves track ground truth and the
error magnitude across the prediction horizon: `[..]`.

## 3.7 Limitations

`[data volume, horizon length, single-model baseline, normalization assumptions, etc.]`.
