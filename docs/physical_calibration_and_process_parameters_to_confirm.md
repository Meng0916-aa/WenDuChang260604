# Physical Calibration & Process Parameters

Status of the physical metadata for the 57-track experimental dataset. The
machine-readable source is `configs/physical_calibration.yaml` (spatial +
temperature) and `configs/experiments.yaml` (process parameters); the validated
loader is `src/config/physical_calibration.py`.

## 1. Spatial calibration — CONFIRMED

User-confirmed reference: **150.2 pixels = 5 mm**.

| Quantity | Value |
|----------|-------|
| `calibration_reference_length_mm` | 5.0 |
| `calibration_reference_pixels` | 150.2 |
| `pixel_size_x_mm` | **0.0332889481** mm/pixel (= 5 / 150.2) |
| `pixel_size_y_mm` | **0.0332889481** mm/pixel |
| `pixel_area_mm2` | **0.0011081541** mm²/pixel (= pixel_size²) |
| `calibration_id` | `formal_150p2px_5mm` |
| `calibration_source` | `user_confirmed` |
| `calibration_status` | `confirmed` |

- **X and Y use the same scale** (`isotropic_scaling_assumed: true`).
- **X/Y anisotropy has NOT been independently verified**
  (`x_y_anisotropy_verified: false`, `calibration_reference_axis: unspecified`).
  If separate X/Y calibration is obtained later, update the two scales.

### Legacy calibration (pilot only — NOT formal)

A pilot value `95.9 px = 3 mm ≈ 0.03128 mm/pixel` exists only as
`legacy_pilot_calibration`. It is **disabled for formal processing**
(`legacy_calibration.enabled_for_formal_processing: false`) and must not be used
for the 57 tracks. It survives only in `configs/default.yaml`
(`thermal_field_features.pixel_size_mm`, kept so existing feature/section tests
still resolve a value) and in the disabled legacy block of
`configs/physical_calibration.yaml`.

## 2. Process parameters — CONFIRMED

The 19-condition 3-factor 3-level Box–Behnken plan is **user-confirmed**:

- `process_parameter_source: user_confirmed_response_surface_plan`
- `process_parameter_status: confirmed`

All **19 conditions** and all **57 tracks** carry the confirmed parameters; T1/T2/T3
within a condition are identical. Parameters are NOT re-inferred from frame counts.
Full table: `docs/actual_experiment_plan.md` / `configs/experiments.yaml`.

## 3. Temperature calibration — binary format verified

`56-byte header + 640×512 little-endian uint16`, `T = raw × 0.1 °C`. Saturation
value is `6553.5 °C` (uint16 ceiling). Robust peak quantile for features:
`0.999`. Saturated/sentinel pixels are masked for analysis only — the raw data
and converted matrices are never modified.

## 4. Still MISSING / unconfirmed (kept null — not guessed)

| Parameter | Status |
|-----------|--------|
| `frame_rate_fps` | **conflicting/missing** — historical values 52 and 1000 are inconsistent; neither adopted |
| `scan_axis`, `scan_direction`, `transverse_axis` | **missing** |
| `exposure_time_us`, `working_distance_mm`, `defocus_mm`, `laser_spot_diameter_mm` | missing |
| `plate_id`, `initial_temperature_C`, `cooling_interval_s`, `powder_feed_actual_g_min` | missing |
| `emissivity`, `transmission`, `camera_temperature_range_C` | missing |
| `effective_start_frame`, `effective_end_frame` | missing (to be detected later) |

While `frame_rate_fps` is unconfirmed, **all time axes use the FRAME INDEX, never
seconds**. `require_frame_rate()` raises rather than defaulting.

## 5. What can / cannot be computed now

**Computable now (spatial / temperature, frame-index based):**
- temperature statistics (°C); robust peak temperature **P99.9** (°C);
- high-temperature pixel count; high-temperature **area (mm²)**;
- 700 °C isotherm region **width (mm)**; hot-zone bounding-box size (mm);
- **unsigned** spatial temperature-gradient magnitude (°C/mm);
- **unsigned** center-offset distance (mm).

**NOT computable yet:**
- scan-direction gradient, transverse gradient (need `scan_axis`/`scan_direction`);
- signed center offset, left/right area asymmetry (need scan direction);
- cooling rate (°C/s), dwell time (s), temperature AUC (°C·s),
  scan distance per frame (mm/frame), effective scan duration (s)
  (need a confirmed `frame_rate_fps`).

## 6. Not run in this phase

No unified ROI evaluation, no formal ROI crop, no effective-frame detection, no
formal temperature-field feature extraction (script 10), no scripts 13–16, no ML,
no response-surface fitting. This phase only fixes the calibration + process
metadata.
