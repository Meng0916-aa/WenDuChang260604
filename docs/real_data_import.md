# Real Data Import Guide

How to bring real Xiris VXIR-3000 camera data into the pipeline. The internal
`.xtherm` binary is **not** parsed by this project — you export temperature
matrices from WeldStudio and feed those in.

## 1. Raw `.xtherm` files — backup only

- Put original `.xtherm` files in **`data/raw_xtherm/`**.
- These are kept as a **backup only**. The project never parses them directly
  (`src/io/xtherm_reader.py` is interface-only).
- **Never** delete, move, or modify anything under `data/raw_xtherm/`.

## 2. Export temperature matrices from WeldStudio

From WeldStudio, export each run as a **temperature matrix** and place it in one of:

- `data/exported/npy/`  ← **recommended**
- `data/exported/csv/`
- `data/exported/h5/`

## 3. Recommended format

| Property | Value |
|----------|-------|
| Format   | `.npy` (preferred) |
| Shape    | `N × H × W` (N frames, H height, W width) |
| dtype    | `float32` |
| Unit     | Celsius |

For `.h5`, store the array under the dataset key `temperature`. For `.csv`,
store one frame per row (flattened `H*W`) or a single `H × W` matrix.

## 4. Counts vs. Celsius (avoid double `/10`)

The raw→Celsius conversion is `temperature = raw_value / 10.0`
(`data.temperature_scale = 0.1`). Control it in `configs/default.yaml`:

- Exported values are **raw xtherm temperature counts** →
  set `data.exported_is_celsius: false`. Script `02` applies `raw_value / 10.0`.
- Exported values are **already Celsius** →
  set `data.exported_is_celsius: true`. Script `02` does **not** scale again.

> Setting this wrong is the most common mistake: leaving `false` on
> already-Celsius data divides everything by 10; setting `true` on raw counts
> leaves temperatures ~10× too high. Check the printed `min/max ... C` in the
> `02` output against physically plausible values.

## 5. Shape is enforced (no silent reshape)

Script `02` enforces the `N × H × W` contract (`data.expected_ndim = 3`):

- A single `H × W` frame is auto-expanded to `(1, H, W)`.
- Any other ndim, or a degenerate spatial dimension, raises an explicit error
  naming the file and shape (e.g. an `H × W × N` array is rejected — re-export
  with frames on the first axis).

## 6. File naming convention

Name one file per experiment run, encoding the magnetic group and a run index:

```
B0_01.npy      B0_02.npy        # no magnetic field, runs 01, 02
B100_01.npy    B100_02.npy      # 100 mT field, runs 01, 02
```

The file stem (e.g. `B100_02`) becomes the `experiment_id` used throughout the
pipeline (splitting, per-experiment metrics, magnetic grouping).

## 7. Magnetic-group naming

| Prefix | Meaning |
|--------|---------|
| `B0`   | no magnetic field (`without_magnetic_field`) |
| `B50`  | 50 mT field |
| `B100` | 100 mT field |
| `B150` | 150 mT field |

List the experiment ids in `configs/default.yaml` so evaluation can produce the
with/without comparison:

```yaml
magnetic_field_groups:
  without_magnetic_field:
    experiment_ids: ["B0_01", "B0_02"]
    label: "without_B"
  with_magnetic_field:
    experiment_ids: ["B100_01", "B100_02", "B150_01"]
    label: "with_B"
```

(Record full per-experiment metadata in `docs/metadata_template.md`.)

## 8. Full run (Windows PowerShell)

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

Before importing real data, check for leftover simulated data:

```powershell
python scripts/01_check_raw_data.py --config configs/default.yaml
```

If it warns about `SIM_*.csv` in `data/processed/thermal_cycle/`, remove or
archive those files **manually** (the scripts never delete anything) so real
data is not mixed with simulated curves.

## 9. What is NOT uploaded to GitHub

`data/` and `results/` are git-ignored and **must never be committed**. This
includes `*.xtherm`, `*.npy`, `*.npz`, `*.h5`, `*.hdf5`, `*.pt`, `*.png`,
`*.pdf`, `*.csv`. Only source, configs, docs, scripts, and tests are tracked.
Real experiment data and trained weights stay on your local machine.
