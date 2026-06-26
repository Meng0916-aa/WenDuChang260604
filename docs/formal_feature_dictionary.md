# Formal Thermal-Field Feature Dictionary

This document defines the first formal feature contract for the 57-track
thermal-field analysis. It is a design and contract document only: no formal ROI
matrices, feature tables, or response-surface inputs have been generated.

Machine-readable source: `configs/thermal_feature_contract.yaml`.

## Design Principles

- One single track is one processing unit. T1/T2/T3 are never concatenated.
- The first acquired frame is the startup frame; formal effective frames are
  `frames[1:]`.
- Track-level features produce 57 rows in the future formal table.
- Condition-level responses aggregate T1/T2/T3 to 19 rows.
- Temperatures are infrared apparent temperatures, not emissivity-corrected
  absolute surface temperatures.
- Quantitative temperature statistics use only camera-valid pixels:
  300-1800 C.
- 1800-6500 C (`above_range`) and >=6500 C (`hard_saturation`) are not valid
  quantitative temperatures. They are reported by QC and may contribute to
  geometry only when connected to a genuine valid hot region.
- The fixed global ROI preserves absolute position and trajectory. The moving
  tracking window supports local temperature, morphology, gradient, offset, and
  asymmetry descriptors.
- The first formal feature set is intentionally small: 15 Core features.

## Frame Populations

| Name | Definition | Use |
|---|---|---|
| `effective_frame` | Any frame in `frames[1:]` | QC over the whole usable sequence |
| `active_700_frame` | Effective frame with a non-empty cleaned main 700 C region | Temperature, 700 geometry, gradient, offset, asymmetry |
| `active_800_frame` | Effective frame with a non-empty cleaned main 800 C region | 800 core presence duration |

No additional fixed head or tail frame trimming is part of the first contract.

Rules:

- QC uses all effective frames.
- Temperature, 700 morphology, gradient, and local asymmetry aggregate over
  `active_700_frame`.
- 800 area aggregates over `active_700_frame`; if the frame has no 800 C main
  region, the 800 area is 0.
- `hot_core_presence_duration_800_C_s = active_800_frame_count / 52`.

## Temperature Validity

| State | Range | Quantitative use | Geometry use | QC |
|---|---:|---|---|---|
| below range | `<300 C` | no | no | count/report |
| camera-valid | `300-1800 C` | yes | yes | count/report |
| above range | `1800-6500 C` | no | only if connected to valid hot main region | count/report |
| hard saturation | `>=6500 C` | no | only as part of connected main-region mask/hole handling | count/report |
| NaN/Inf | nonfinite | no | no | count/report |

Forbidden:

- Clipping above-range values to 1800 C and using them as valid temperatures.
- Interpolating hard-saturation pixels.
- Replacing invalid feature values with 0.
- Silently deleting abnormal tracks.

## Geometry Contract

Main 700/800 regions use the cleaned main connected component already established
by the ROI evaluation policy:

- threshold at 700 C or 800 C;
- 8-connectivity;
- small components below 9 px removed;
- a component is genuine only if it contains at least one camera-valid hot pixel;
- isolated above-range or hard-saturation components cannot become the main
  region.

Area and sizes:

- area = `pixel_count * pixel_area_mm2`;
- transverse width = `(max_col - min_col + 1) * pixel_size_x_mm`;
- scan length = `(max_row - min_row + 1) * pixel_size_y_mm`.

The `+1` is part of the inclusive span definition over occupied pixel indices.

## Coordinate Convention

Trajectory features use the fixed global ROI and full-image coordinates.

- array origin: top-left;
- row increases downward;
- column increases to the image right;
- `scan_axis = y`;
- `image_scan_direction = upward`;
- `physical_to_array_y_sign = -1`;
- `x_mm = col * pixel_size_x_mm`;
- `y_mm = -row * pixel_size_y_mm`.

Thus positive scan-direction displacement corresponds to image movement upward
(row index decreasing). Positive transverse drift corresponds to increasing
image column index. Until a physical left/right calibration is established, the
column direction must not be described as physical left or right on the workpiece.

## ROI Responsibilities

| Feature category | Fixed global ROI | Moving tracking window | Final region | Reason |
|---|---|---|---|---|
| QC pixel-state fractions | optional | optional | full frame | Detect whole-matrix quality issues |
| Absolute position | yes | no | fixed global ROI | Moving windows erase absolute position |
| Trajectory, displacement, drift | yes | no | fixed global ROI | Requires full-image coordinates |
| Local temperature statistics | no | yes | tracking window | Avoid background dilution |
| Main hot-region geometry | optional | yes | tracking window | Local morphology, less background |
| Internal gradient | no | yes | tracking window | Local valid-neighborhood operation |
| Thermal centroid offset | no | yes | tracking window | Local symmetry around the main region |
| Excess-temperature asymmetry | no | yes | tracking window | Local transverse energy imbalance |

## Core Features

The first formal feature set contains exactly these 15 Core features.

| Feature | Definition | Unit | Region | Frame calculation | Track aggregation | Missing policy |
|---|---|---:|---|---|---|---|
| `mean_active_frame_valid_temperature_C` | Mean of frame-level valid-pixel means | C | tracking window | mean of 300-1800 C pixels in each active 700 frame | arithmetic mean over frames | frame NaN if no valid pixel |
| `max_frame_p999_valid_temperature_C` | Maximum frame P99.9 in the valid band | C | tracking window | spatial P99.9 of 300-1800 C pixels | max over frames | not a true highest surface T |
| `mean_main_area_above_700_C_mm2` | Mean cleaned main 700 C area | mm2 | tracking window | count main 700 pixels * pixel area | arithmetic mean | no active 700 frames invalid |
| `mean_main_area_above_800_C_mm2` | Mean cleaned main 800 C area | mm2 | tracking window | count main 800 pixels * pixel area; empty 800 = 0 | arithmetic mean | empty 800 counted by QC |
| `mean_main_transverse_width_above_700_C_mm` | Mean main 700 transverse span | mm | tracking window | `(max_col-min_col+1)*dx` | arithmetic mean | no active 700 frames invalid |
| `mean_main_scan_length_above_700_C_mm` | Mean main 700 scan-direction span | mm | tracking window | `(max_row-min_row+1)*dy` | arithmetic mean | no active 700 frames invalid |
| `centroid_path_length_mm` | Sum of adjacent centroid distances | mm | fixed global ROI | main 700 geometric centroid in full-image coordinates | sum adjacent valid steps | do not connect across missing centroid gaps |
| `signed_scan_direction_displacement_mm` | Final minus initial scan-direction position | mm | fixed global ROI | `y_mm=-row*dy` | last valid minus first valid | NaN if fewer than two centroids |
| `signed_transverse_drift_mm` | Final minus initial transverse position | mm | fixed global ROI | `x_mm=col*dx` | last valid minus first valid | NaN if fewer than two centroids |
| `centroid_transverse_jitter_mm` | RMS residual after fitting `x=a*y+b` | mm | fixed global ROI | main 700 geometric centroids | RMS transverse residual | NaN if fewer than three centroids |
| `median_frame_p95_internal_gradient_magnitude_700_C_per_mm` | Median frame P95 internal gradient magnitude | C/mm | tracking window | center difference inside main 700; center and neighbors valid; no smoothing | median of frame P95 values | frame NaN if too few gradient pixels |
| `mean_signed_thermal_centroid_offset_from_geometric_center_mm` | Excess-temperature centroid transverse offset from geometric centroid | mm | tracking window | weights `max(T-700,0)` for `700<=T<=1800 C` | arithmetic mean | NaN if zero weight |
| `mean_left_right_excess_temperature_asymmetry_700_fraction` | Mean transverse excess-temperature asymmetry | fraction | tracking window | `(S_right-S_left)/(S_right+S_left)` where S is sum excess above 700 C | arithmetic mean | NaN if denominator zero |
| `hot_core_presence_duration_800_C_s` | Time with an 800 C main core present | s | tracking window | active 800 frame indicator | `active_800_frame_count / 52` | zero allowed; not material dwell |
| `main_area_above_700_C_temporal_cv` | Temporal CV of main 700 area | cv | tracking window | frame area over active 700 frames | sample std / mean | NaN if fewer than two frames or mean near zero |

## QC-Only Fields

QC fields are not default model inputs and are not default response-surface
responses.

- `total_effective_frame_count`
- `active_700_frame_count`
- `active_800_frame_count`
- `full_frame_valid_pixel_fraction`
- `full_frame_above_range_pixel_fraction`
- `full_frame_hard_saturation_pixel_fraction`
- `full_frame_nonfinite_pixel_fraction`
- `empty_700_frame_fraction`
- `empty_800_frame_fraction`
- `feature_valid`
- `warning_flags`

QC must never silently delete a track. It may mark features or whole-track rows
invalid and record warning flags.

## Secondary Features

These are deferred from the first Core set to control feature count and avoid
unstable definitions:

- `p95_valid_temperature_C`
- `p99_valid_temperature_C`
- `width_above_800_C_mm`
- `scan_length_above_800_C_mm`
- `compactness_700`
- `aspect_ratio_700`
- `front_rear_asymmetry`
- `centroid_velocity_mm_per_s`
- `time_to_peak_s`
- `post_peak_temperature_decay_rate_C_per_s`
- `heating_rate_C_per_s`
- `temperature_auc_C_s`
- `peak_temperature_fluctuation_C`

The P99.9 process curve is not a fixed material-point thermal cycle. Time-based
features derived from it must be described as process-curve descriptors, not as
material-point thermal histories.

## Rejected Features

| Feature | Reason |
|---|---|
| `single_pixel_max_temperature_C` | Polluted by above-range and hard-saturation pixels |
| `temperature_above_range_clipped_to_1800_C` | Above-range values are not quantitative temperatures |
| `interpolated_hard_saturation_temperature` | Invalid pixels must be mask-and-report only |
| `pseudo_color_image_features` | Pseudo-color images are not quantitative temperature data |
| `full_frame_background_temperature_mean` | Dominated by background, not a melt-pool response |
| `all_temperature_percentiles_simultaneously` | Too redundant for 19 condition-level responses |
| `concatenated_T1_T2_T3_sequence` | Repeated tracks must not be concatenated |
| `legacy_500_C_haz_width` | Not part of the formal 700/800 C threshold contract |

## Track-Level Output Contract

Planned path, not created in this phase:
`results/tables/formal_track_thermal_features.csv`.

Expected rows: 57.

Required field groups:

- identifiers: `sample_id`, `condition_id`, `track_id`, `plate_id`,
  `track_order`;
- process parameters: `laser_power_W`, `scan_speed_mm_min`,
  `magnetic_field_mT`, `powder_feed_set_g_min`, `travel_distance_mm`;
- config provenance: `formal_pipeline_config`, `roi_strategy_config`,
  `physical_calibration_config`, `thermal_feature_contract_config`;
- QC-only fields;
- Core feature fields;
- validity fields and warning flags.

## Condition-Level Output Contract

Planned path, not created in this phase:
`results/tables/formal_condition_thermal_responses.csv`.

Expected rows: 19.

Column naming:

- `feature__mean`
- `feature__std`
- `feature__cv`
- `feature__n_valid`

Rules:

- group only by `condition_id`;
- aggregate T1/T2/T3 only within the same condition;
- do not concatenate tracks;
- mean is arithmetic mean over valid track values;
- std is sample standard deviation with `ddof=1`;
- CV is `std / abs(mean)` only for `cv_applicable: true`;
- signed features have `cv_applicable: false`;
- if `n_valid < 2` or `abs(mean)` is near zero, CV is NaN with a status flag;
- abnormal repeats are flagged, not silently deleted.

## Research Limitations

- Apparent infrared temperature is not radiometrically corrected absolute
  surface temperature.
- Emissivity and transmission are not recorded.
- 1800 C and above are not quantitative.
- Saturation can bias peak metrics; robust valid-band P99.9 is used instead of
  single-pixel max.
- Moving tracking windows cannot preserve absolute position; trajectory uses
  fixed global ROI.
- T1/T2/T3 are in-plate repeated tracks, not independent plate-level repeats.
- There are 19 condition-level responses, so feature count must remain limited.
- Magnetic field, laser power, and scan speed may interact; later RSM should
  treat interactions explicitly.
