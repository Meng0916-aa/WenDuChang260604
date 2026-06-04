# After-Experiment Checklist

Operations to perform **after** a real experiment has been recorded with the
Xiris VXIR-3000 camera. Work through this list in order each time you bring a
new batch of runs into the pipeline.

> Reminder: `data/` and `results/` are local-only and are never uploaded to
> GitHub. The scripts never delete anything — any cleanup below is manual.

## 1. Export temperature matrices from WeldStudio

For each run, export the **quantitative temperature matrix** (not a pseudo-color
image). Pseudo-color PNG/JPG/TIFF heatmaps are not valid model input.

## 2. Recommended save format

- Format: `.npy` (preferred)
- Shape: **N × H × W** (N frames, H height, W width)
- dtype: `float32`
- Unit: Celsius

(`.csv` and `.h5` are also accepted — see `docs/real_data_import.md`.)

## 3. Back up the raw `.xtherm`

- Copy the original `.xtherm` files into **`data/raw_xtherm/`** as a backup.
- Never delete, move, or modify anything under `data/raw_xtherm/`. The project
  does not parse `.xtherm` directly.

## 4. Place exported matrices

- Put the exported `.npy` files into **`data/exported/npy/`**.
- Name one file per run, e.g. `B0_01.npy`, `B100_02.npy` (prefix = magnetic
  group: `B0` = no field, `B50` / `B100` / `B150` = field strength). The file
  stem becomes the `experiment_id`.

## 5. Verify `exported_is_celsius`

In `configs/default.yaml` under `data:`, confirm the unit setting matches how
the matrices were exported:

- Exported as **raw xtherm counts** → `exported_is_celsius: false`
  (script `02` applies `raw_value / 10.0`).
- Exported as **Celsius** → `exported_is_celsius: true`
  (script `02` does NOT scale again).

Getting this wrong makes temperatures ~10× too high or too low — sanity-check
the `min/max ... C` printed by script `02`.

## 6. Verify ROI covers the melt pool + heat-affected zone

In `configs/default.yaml` under `roi:`, check `top`, `left`, `height`, `width`
(pixels; `top` = row, `left` = column). The cropped window must cover both the
**melt pool** and the surrounding **heat-affected zone** for every run. Adjust
if the deposition region moved between experiments.

## 7. Archive any leftover SIMULATED data first

If `data/processed/thermal_cycle/` contains `SIM_*.csv` files (leftover from a
code-chain test), **manually archive or remove them** before processing real
data, so simulated curves are not mixed with real experiments. Running
`python scripts/01_check_raw_data.py --config configs/default.yaml` warns when
`SIM_*.csv` is present. (The scripts never delete files for you.)

## 8. Run the full pipeline (Windows PowerShell)

```powershell
conda activate pytorch
$env:KMP_DUPLICATE_LIB_OK="TRUE"

python scripts/01_check_raw_data.py --config configs/default.yaml
python scripts/02_convert_exported_to_npy.py --config configs/default.yaml
python scripts/03_extract_roi.py --config configs/default.yaml
python scripts/04_extract_thermal_cycle.py --config configs/default.yaml
python scripts/05_build_window_dataset.py --config configs/default.yaml
python scripts/06_train_model.py --config configs/default.yaml
python scripts/07_evaluate_model.py --config configs/default.yaml
python scripts/08_plot_results.py --config configs/default.yaml
```

## 9. Check the outputs

- `results/tables/` — `lstm_metrics.csv`, `lstm_predictions.csv`,
  `lstm_metrics_by_experiment.csv`, and (if magnetic groups are set in the
  config) `lstm_metrics_by_magnetic_group.csv`.
- `results/figures/` — prediction curves and metric charts (`.png` + `.pdf`).

Confirm the result files are **not** tagged `SIMULATED` (which would mean real
data was not picked up and the simulation fallback ran instead).

## 10. Do NOT upload data or results to GitHub

`data/` and `results/` are git-ignored and must never be committed — including
`*.xtherm`, `*.npy`, `*.npz`, `*.h5`, `*.pt`, `*.png`, `*.pdf`, `*.csv`. Only
source code, configs, docs, scripts, and tests are tracked. Real experiment
data and trained weights stay on your local machine.
