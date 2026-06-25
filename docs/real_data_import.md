# Real Data Import Guide

How to bring real Xiris VXIR-3000 camera data into the pipeline. The project
supports two import paths:

1. **Formal 57-track path:** `scripts/02c_batch_convert_tracks.py` directly
   parses verified binary `.xtherm` frames using `configs/xtherm_format.yaml`
   and writes one full-frame matrix per single track.
2. **Legacy-compatible paths:** `scripts/02b_convert_xtherm_binary_to_npy.py`
   converts a historical single dataset directory, while exported `.npy` /
   `.csv` / `.h5` files remain supported by the older import workflow.

`src/io/xtherm_reader.py` is interface-only, but verified binary parsing is
implemented in `src/conversion/xtherm_binary.py`.

## Formal 57-track workflow

The current formal workflow is the 57-track path:

- Experiment IDs: `C1`, `C2`, `R1`-`R17`.
- Track IDs: `T1`, `T2`, `T3`.
- Magnetic-field levels: 0, 60, 120 mT.
- Master file: `data/metadata/experiment_master.csv`.
- Converter: `scripts/02c_batch_convert_tracks.py`.
- Format authority: `configs/xtherm_format.yaml`.
- No need to re-export matrices through WeldStudio.
- Existing 57 matrices must not be regenerated merely for validation.
- The converter writes one matrix per track and never concatenates `T1/T2/T3`.
- `configs/default.yaml` is not used as formal authority.

The formal 57-track conversion is already complete. Do not use `--overwrite`
unless explicitly approved by the user.

## Legacy pilot import workflow

The remaining sections describe historical or legacy-compatible import paths.
They are not part of the current formal 57-track workflow unless the user
explicitly activates a legacy task.

## 1. Raw `.xtherm` files in the legacy pilot layout

- Put original `.xtherm` files in **`data/raw_xtherm/`**.
- Files may be organized into **per-dataset subfolders**, e.g.:

  ```
  data/raw_xtherm/dataset/001027.xtherm
  data/raw_xtherm/dataset/001028.xtherm
  ...
  data/raw_xtherm/dataset/001260.xtherm
  ```

- Script `01` counts `.xtherm` files **recursively** (`data/raw_xtherm/**/*.xtherm`)
  and reports, per subfolder, the file count and the first/last frame filenames,
  e.g. `dataset: 234 files` / `dataset: first=001027.xtherm, last=001260.xtherm`.
  A subfolder containing no `.xtherm` files is flagged with a NOTE.
- In the legacy exported-matrix workflow, these files are retained as source
  data while processing may proceed through `02b` or through exported matrices
  under `data/exported/npy/` (e.g. `data/exported/npy/dataset.npy`) — see below.
- **Never** delete, move, or modify anything under `data/raw_xtherm/`.

## 1b. Binary `.xtherm` → npy with script 02b (verified format)

WeldStudio's `.xtherm` export is a **binary temperature matrix**, not a text
CSV. The layout used by this project has been verified on real data:

| Property      | Value |
|---------------|-------|
| Header        | **56 bytes** (skipped) |
| Payload       | **640 × 512** pixels, **little-endian uint16** (`<u2`) |
| File size     | `56 + 640 × 512 × 2 = 655416` bytes per frame |
| Reshape       | `512 × 640` (H × W) |
| Celsius       | `raw_value × 0.1` (`scale_factor = 0.1`) |

> Big-endian reads give ~6500 °C on the first frame — clearly wrong. Always
> little-endian.

Convert all frames in one go:

```powershell
python scripts/02b_convert_xtherm_binary_to_npy.py --format-config configs/xtherm_format.yaml --config configs/default.yaml
```

This reads `xtherm_binary.input_dir` recursively (e.g.
`data/raw_xtherm/dataset/`), sorts frames by filename, stacks them into
`data/exported/npy/dataset.npy` (**N × H × W float32 Celsius**) and writes
`dataset_meta.json` alongside. The formal XTherm source is
`configs/xtherm_format.yaml`, which defines the binary layout, image dimensions,
temperature conversion, camera valid range, and conversion-QC thresholds.
`configs/default.yaml` is legacy configuration only; it may supply historical
pilot input/output paths for this legacy-compatible utility, but it is not used
as the authoritative format source and is not part of the formal 57-track
pipeline. The raw `.xtherm` files are only read — never deleted, moved, or
modified.

**Important:** because `02b` already applies `× 0.1`, its output is Celsius —
keep `data.exported_is_celsius: true` so script `02` does not divide by 10
again. `dataset.npy` / `dataset_meta.json` are local-only and are never
committed to GitHub.

## 2. Export temperature matrices from WeldStudio

Alternatively, export each run from WeldStudio directly as a **temperature
matrix** and place it in one of:

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

## 5. Shape is enforced (N × H × W, no silent reshape)

Script `02` enforces the `N × H × W` axis order (`data.expected_ndim = 3`,
`data.expected_frame_axis = 0`) — **frames must be the first axis**:

- A single `H × W` frame is auto-expanded to `(1, H, W)`.
- Any other ndim, a degenerate spatial dimension, or too few frames on axis 0
  (`data.min_frames`, default 2) raises an explicit error of the form
  `expected N x H x W, got shape=...`.
- Optionally pin the spatial size with `data.expected_height` /
  `data.expected_width` (default `null` = unchecked); a mismatch is rejected.

> **If your software exports `H × W × N` (frames last), the script does NOT
> transpose for you** — it raises an error on purpose. Convert it to `N × H × W`
> first, e.g.:
>
> ```python
> import numpy as np
> a = np.load("export_HWN.npy")        # shape (H, W, N)
> np.save("export_NHW.npy", np.moveaxis(a, -1, 0))   # -> (N, H, W)
> ```
>
> Re-export / convert and place the corrected `N × H × W` file in
> `data/exported/npy/`.

## 5b. Archive leftover SIMULATED data before importing

Before importing real data, make sure `data/processed/thermal_cycle/` contains
**no** `SIM_*.csv` files (the script-05 fallback). Script `05` now **refuses to
run** if real and `SIM_*.csv` files are mixed in that directory, and script `01`
warns when `SIM_*.csv` is present. Manually archive or delete the `SIM_*.csv`
files first (the scripts never delete anything for you).

## 6. File naming convention

This naming convention belongs to the legacy exported-matrix workflow, not the
formal 57-track workflow.

Name one file per experiment run, encoding the magnetic group and a run index:

```
B0_01.npy      B0_02.npy        # no magnetic field, runs 01, 02
B100_01.npy    B100_02.npy      # 100 mT field, runs 01, 02
```

The file stem (e.g. `B100_02`) becomes the `experiment_id` used throughout the
pipeline (splitting, per-experiment metrics, magnetic grouping).

## 7. Magnetic-group naming

These magnetic-group names belong to the legacy exported-matrix workflow. The
formal 57-track workflow uses `C1`, `C2`, and `R1`-`R17` with magnetic-field
levels 0, 60, and 120 mT from `configs/experiments.yaml`.

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

This is the legacy LSTM-related full run. It is not the current formal
57-track workflow.

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
