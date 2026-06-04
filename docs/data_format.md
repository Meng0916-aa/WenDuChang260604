# Data Format

All temperature data in this project follows two hard conventions:

- **Unit:** float32 degrees **Celsius**.
- **Shape:** `N × H × W` (N frames, H height, W width).
- **Raw → Celsius:** `temperature = raw_value / 10.0` (config `data.temperature_scale = 0.1`).

## 1. Raw `.xtherm` (Xiris VXIR-3000 / WeldStudio Pro)

- Binary format produced by the camera software. Contains raw digital counts.
- The internal layout is **not parsed** in this project yet. `src/io/xtherm_reader.py`
  is interface-only and raises `NotImplementedError`. Real parsing is deferred to the
  Xiris WeldSDK or an official export interface.
- Raw files live under `data/raw_xtherm/` and must **never** be deleted, moved, or modified.
- Pseudo-color images (PNG/JPG/TIFF heatmaps) are **not** quantitative and must not be used
  as model input.

## 2. Exported matrices `.npy` / `.csv` / `.h5`

Exported from WeldStudio Pro as temperature matrices.

| Format | Expected content |
|--------|------------------|
| `.npy` | array `(N, H, W)` or `(H, W)` (single frame, auto-expanded) |
| `.csv` | one frame per row flattened `(N, H*W)`, or a single `(H, W)` matrix |
| `.h5`  | dataset `temperature` of shape `(N, H, W)` |

Celsius handling is **config-controlled**, not guessed:

- `data.exported_is_celsius = false` → values are raw counts; script `02` multiplies by
  `data.temperature_scale` (0.1) to get Celsius.
- `data.exported_is_celsius = true` → values are already Celsius; script `02` does **not**
  divide again.

Loader: `src/io/export_loader.py`. Output of script `02`: `data/processed/matrix/*.npy`.

## 3. Processed matrices `data/processed/matrix`, `data/processed/roi`

- `data/processed/matrix/*.npy`: standardized `(N, H, W)` float32 Celsius.
- `data/processed/roi/*.npy`: ROI-cropped frames (`src/preprocess/roi.py`), bounds from
  `roi.bounds = [x1, y1, x2, y2]`.

## 4. Thermal-cycle CSV `data/processed/thermal_cycle`

One CSV per experiment, produced by script `04` (`src/preprocess/thermal_cycle.py`):

```
frame,tmax,center_average,hot_zone_average
0,812.3000,701.5000,845.2000
1,...
```

- `tmax`: max temperature per frame.
- `center_average`: mean temperature in a circular ROI of radius
  `thermal_cycle.center_average_radius` around the frame center.
- `hot_zone_average`: mean over pixels ≥ `thermal_cycle.hot_zone_threshold_celsius`.

## 5. Window samples `data/processed/samples/window_samples.npz`

Produced by script `05`. Keys:

- `X_train / X_val / X_test`: `(num_samples, input_len, feature_dim)`
- `y_train / y_val / y_test`: `(num_samples, pred_len, n_curves)`
- `exp_train / exp_val / exp_test`: per-sample experiment id
- `mag_group_test`: per-test-sample magnetic group label
- `feature_dim`, `feature_columns`, `input_len`, `pred_len`, `simulated`

`feature_dim = len(feature_columns) + (number of process params, if enabled)`.
The prediction target is feature channel 0 (the first entry of `dataset.feature_columns`).
