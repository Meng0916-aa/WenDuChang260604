# Physical Calibration & Process Parameters

Status of the physical metadata for the 57-track experimental dataset. The
machine-readable sources are `configs/physical_calibration.yaml` (spatial scale,
image geometry, temperature range, frame rate, working distance, effective-frame
rule) and `configs/experiments.yaml` (process parameters, fixed hardware params,
powder feed, substrate, track repetition). The validated loader is
`src/config/physical_calibration.py`.

> **Most first-order parameters are now user-confirmed** (this freeze). The only
> items still **not recorded** are emissivity, transmission, exposure time and
> lens model — kept null, never guessed.

## 1. Spatial calibration — CONFIRMED

User-confirmed reference: **150.2 pixels = 5 mm**, measured along the **image
vertical (Y) axis**.

| Quantity | Value |
|----------|-------|
| `calibration_reference_length_mm` | 5.0 |
| `calibration_reference_pixels` | 150.2 |
| `calibration_reference_axis` | **y** |
| `pixel_size_y_mm` | **0.0332889481** mm/pixel (= 5 / 150.2, measured) |
| `pixel_size_x_mm` | **0.0332889481** mm/pixel (assumed equal — isotropic) |
| `pixel_area_mm2` | **0.0011081541** mm²/pixel (= pixel_size²) |
| `calibration_id` | `formal_150p2px_5mm` |
| `calibration_source` | `user_confirmed` |
| `calibration_status` | `confirmed` |

- The **Y scale is measured**; the **X scale is assumed equal** by the isotropic
  assumption (`isotropic_scaling_assumed: true`).
- **X/Y anisotropy has NOT been independently verified**
  (`x_y_anisotropy_verified: false`). Do **not** claim X and Y were each
  calibrated separately. If a separate X calibration is obtained later, update
  the two scales.

### Legacy calibration (pilot only — NOT formal)

A pilot value `95.9 px = 3 mm ≈ 0.03128 mm/pixel` exists only as
`legacy_calibration`. It is **disabled for formal processing**
(`enabled_for_formal_processing: false`) and is
`not_active_for_formal_processing`. It survives only in `configs/default.yaml`
(`thermal_field_features.pixel_size_mm`, kept so existing feature/section tests
still resolve a value) and in the disabled legacy block of
`configs/physical_calibration.yaml`.

## 2. Image geometry & scan direction — CONFIRMED

User-confirmed: the image is **not rotated, not flipped**; the melt pool moves
toward the **image top** over time.

| Field | Value |
|-------|-------|
| `image_height_px` × `image_width_px` | 512 × 640 |
| `scan_axis` | **y** |
| `transverse_axis` | **x** |
| `image_scan_direction` | **upward** |
| `array_scan_direction` | **decreasing_row_index** |
| `physical_positive_y_direction` | `image_upward` |
| `physical_to_array_y_sign` | **−1** |
| `rotation_deg` / `flip_horizontal` / `flip_vertical` | 0 / false / false |

**Coordinate meaning.** The NumPy array origin is the image **top-left**; the row
index increases **downward**. Because the melt pool moves toward the image top
over time, the array **row index decreases during scanning**. Physical **+Y**
corresponds to image **up**. A later signed coordinate (after ROI fixes a
reference row) may use:

```
y_physical_mm = (reference_row_px - row_index_px) × pixel_size_y_mm
```

No physical zero/reference origin is defined in this phase. The scan direction is
**user-defined** — a later scan-direction QC may *check* track motion but must
**not** override this definition.

## 3. Temperature calibration & measurement range — CONFIRMED

Binary format (verified): `56-byte header + 640×512 little-endian uint16`,
`T = raw × 0.1 °C`. The camera is quantitatively valid only over **300 – 1800 °C**
(`measurement_range_status: confirmed`).

Four temperature states (raw matrices are **never** modified or truncated):

| Band | State | Interpretation |
|------|-------|----------------|
| `T < 300 °C` | `below_range` | below camera valid lower limit — not a quantitative value |
| `300 ≤ T ≤ 1800 °C` | `valid` | camera valid quantitative measurement band |
| `1800 < T < 6500 °C` | `above_range` | over the camera upper limit — **not** a real temperature |
| `T ≥ 6500 °C` | `hard_saturation` | uint16 ceiling (`6553.5 °C`) / invalid encoding |

Analysis policy (`temperature_analysis_policy`): `preserve_raw_matrix: true`,
`below/above/hard` → `mask_and_report`, `interpolate_invalid_pixels: false`,
`robust_peak_quantile: 0.999`.

- **P99.9 is computed within the valid band only.** If the in-band peak
  approaches 1800 °C, the **above-range ratio must be reported** alongside it.
- `2739 °C`, `3085 °C`, the `3000–3700 °C` values, and `6553.5 °C` are **not**
  real melt-pool temperatures.
- Feature code must report `below_range_*`, `above_range_pixel_count` /
  `above_range_pixel_ratio`, and `hard_saturation_pixel_count` quality metrics.
- This phase only **fixes the policy** in config/docs; the masking implementation
  and formal feature program are **not** changed/run here.

> Because emissivity/transmission are not recorded, the analysis object is more
> rigorously the **infrared-camera output apparent temperature field**, not a
> fully radiometrically-corrected absolute surface temperature.

## 4. Acquisition / time & camera optics — CONFIRMED

| Parameter | Value | Status / source |
|-----------|-------|-----------------|
| `frame_rate_fps` | **52.0** | confirmed — `user_confirmed_experimental_setting` (NOT read from `session.xml`) |
| Effective frame rule | frame 1 excluded (startup); effective = frames 2…last (1-based) = `frames[1:]` (0-based start 1) | confirmed |
| `working_distance_mm` | **300.0** (= 30 cm) | confirmed; reference = `protective_window_to_molten_pool` |

- An earlier **30 mm** working distance was an `incorrect_legacy_value`; the
  formal value is **300 mm**. Only one active working distance exists.
- `"last_available_frame"` is the **end of the data**, not a Python
  slice-exclusive endpoint. Effective data is `frames[1:]` — but the 57 matrices
  are **not** reprocessed in this task.

## 5. Process & substrate parameters — CONFIRMED

| Parameter | Value |
|-----------|-------|
| `laser_spot_diameter_mm` | 1.0 |
| `defocus_mm` | **+14.0** (`positive` = focal plane **above** substrate surface) |
| `travel_distance_mm` | 30 |
| `powder_feed_set_g_min` | **40.0** (equipment setpoint) |
| `powder_feed_actual_g_min` | **null** (`not_measured` — never weighed) |
| Substrate material | **316L** |
| Substrate size | **40 × 16 × 8 mm** |
| Plates | **19** (one independent plate per condition; T1/T2/T3 share it) |
| `plate_id` | `Plate-<condition_id>`, `plate_id_type = logical_condition_based_identifier` |
| Track order / cooling | T1→T2→T3, **120 s** between consecutive tracks (in-plate) |

The response-surface factors are unchanged: laser power 300/450/600 W, scan speed
400/600/800 mm/min, magnetic field 0/60/120 mT.

> **Design limitation.** Each response-surface condition uses **one** 316L plate;
> T1/T2/T3 are **in-plate repeats**, not three independent plates. Therefore the
> **condition effect and plate-to-plate individual variation cannot be fully
> separated**. The 120 s interval is the in-plate inter-track cooling, **not** a
> between-condition (between-plate) cooling time.

## 6. Still MISSING / not recorded (kept null — not guessed)

| Parameter | Status |
|-----------|--------|
| `emissivity` | `not_recorded` |
| `transmission` | `not_recorded` |
| `exposure_time_us` | `not_recorded` |
| `lens_model` | `not_recorded` |
| `effective_end_frame` (numeric) | uses rule `last_available_frame` (no per-track number yet) |

## 7. What can / cannot be computed now

**Computable now** (apparent-temperature features; frame-rate now available):
- valid-band temperature statistics (°C); robust peak **P99.9** (°C) within band;
- above-range / hard-saturation pixel counts and ratios;
- high-temperature **area (mm²)**; 700/800 °C isotherm region **width (mm)**;
  hot-zone bounding-box size (mm);
- scan-direction & transverse temperature gradient (°C/mm), **signed** center
  offset (mm), left/right area asymmetry (scan geometry is now confirmed);
- cooling rate (°C/s), high-temperature dwell time (s), temperature AUC (°C·s),
  scan distance per frame (mm/frame), effective scan duration (s) (frame rate is
  now confirmed at 52 fps).

**Still required before formal feature extraction** (not done in this task):
unified ROI confirmation, invalid-pixel masking implementation, valid-temperature
range masking implementation, final feature-definition review.

## 8. Not run in this phase

No unified ROI evaluation, no formal ROI crop, no effective-frame *detection*, no
formal temperature-field feature extraction (script 10), no scripts 13–16, no ML,
no response-surface fitting, no re-conversion or modification of the 57 matrices.
This phase only **freezes** the calibration + process + camera + substrate
metadata in the configs, the master generator, docs and tests.
