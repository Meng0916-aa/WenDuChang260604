# Actual Experiment Plan (Formal)

> **This is the formal experiment design and the single source of truth.**
> The machine-readable copy is `configs/experiments.yaml`; all programs must
> read process parameters and the track list from there. The older
> `docs/experiment_protocol.md` and `docs/metadata_template.md` are generic
> templates / examples and are **not** the formal plan.

## 1. Research purpose

Study how laser-cladding process parameters **and an applied magnetic field**
shape the **temperature field** of single-track CoCrNi deposits. The current
phase quantifies, per single track, the whole-track thermal-field response, then
aggregates the three repeats of each process condition to describe the
condition-level thermal response and (later) a response-surface / magnetic-field
effect analysis.

A later phase will join these thermal-field features with cross-section quality
labels for forming-quality prediction — see §7.

## 2. Experimental factors and levels

Three-factor, three-level **Box–Behnken Design (BBD)**. Coded levels -1 / 0 / +1:

| Factor | Unit | Low (-1) | Center (0) | High (+1) |
|--------|------|---------:|-----------:|----------:|
| Laser power      | W      | 300 | 450 | 600 |
| Scan speed       | mm/min | 400 | 600 | 800 |
| Magnetic field   | mT     |   0 |  60 | 120 |

> Scan speed is in **mm/min** — never mm/s.
> A track is **with magnetic field** iff `magnetic_field_mT > 0`, otherwise
> **without magnetic field**.

## 3. Fixed process parameters (user-confirmed)

| Parameter | Value | Note |
|-----------|------:|------|
| Powder feed — **setpoint** (`powder_feed_set_g_min`) | 40 g/min | equipment setpoint |
| Powder feed — **actual** (`powder_feed_actual_g_min`) | *null* | `not_measured` (never weighed) |
| Travel distance (`travel_distance_mm`) | 30 mm | |
| Laser spot diameter (`laser_spot_diameter_mm`) | 1.0 mm | |
| Defocus (`defocus_mm`) | **+14.0 mm** | `positive` = focal plane **above** substrate surface |
| Repeated tracks per condition | 3 (T1, T2, T3) | in-plate repeats |
| Track order | T1 → T2 → T3 | `track_order` 1/2/3 |
| Cooling between consecutive tracks | **120 s** | in-plate inter-track interval (not between conditions) |
| Frame rate (`frame_rate_fps`) | **52 fps** | confirmed experimental setting (not from `session.xml`) |
| Working distance | **300 mm** | protective-window-to-molten-pool (earlier 30 mm was wrong) |
| Camera valid range | **300 – 1800 °C** | outside → masked & reported, raw never modified |

> 40 g/min is the **setpoint** — do not write it into the *actual* powder-feed
> field. The full machine-readable record lives in `configs/experiments.yaml`
> (`fixed_process_parameters`, `powder_feed`, `track_repetition`) and
> `configs/physical_calibration.yaml` (geometry, frame rate, range, optics).

## 4. The 19 conditions

19 process conditions × 3 repeated single tracks = **57 independent thermal-field
samples**.

| condition_id | design_role | laser_power_W | scan_speed_mm_min | magnetic_field_mT | powder_feed_g_min | travel_distance_mm |
|---|---|---:|---:|---:|---:|---:|
| C1  | control_no_field          | 450 | 600 |   0 | 40 | 30 |
| C2  | control_high_field        | 450 | 600 | 120 | 40 | 30 |
| R1  | box_behnken_edge          | 300 | 400 |  60 | 40 | 30 |
| R2  | box_behnken_edge          | 600 | 400 |  60 | 40 | 30 |
| R3  | box_behnken_edge          | 300 | 800 |  60 | 40 | 30 |
| R4  | box_behnken_edge          | 600 | 800 |  60 | 40 | 30 |
| R5  | box_behnken_edge          | 300 | 600 |   0 | 40 | 30 |
| R6  | box_behnken_edge          | 600 | 600 |   0 | 40 | 30 |
| R7  | box_behnken_edge          | 300 | 600 | 120 | 40 | 30 |
| R8  | box_behnken_edge          | 600 | 600 | 120 | 40 | 30 |
| R9  | box_behnken_edge          | 450 | 400 |   0 | 40 | 30 |
| R10 | box_behnken_edge          | 450 | 800 |   0 | 40 | 30 |
| R11 | box_behnken_edge          | 450 | 400 | 120 | 40 | 30 |
| R12 | box_behnken_edge          | 450 | 800 | 120 | 40 | 30 |
| R13 | box_behnken_center_repeat | 450 | 600 |  60 | 40 | 30 |
| R14 | box_behnken_center_repeat | 450 | 600 |  60 | 40 | 30 |
| R15 | box_behnken_center_repeat | 450 | 600 |  60 | 40 | 30 |
| R16 | box_behnken_center_repeat | 450 | 600 |  60 | 40 | 30 |
| R17 | box_behnken_center_repeat | 450 | 600 |  60 | 40 | 30 |

`R13`–`R17` are five replicates of the BBD center point (used to estimate pure
error / repeatability). `C1` and `C2` are reference runs at the center power/speed
with the field off vs. at the high level.

## 5. Repeated tracks (T1 / T2 / T3)

Each condition contains three single tracks **T1, T2, T3**. They are **three
repeated experiments under the same process condition**, i.e. three independent
realizations — **not** one long track split into three.

**Hard rule — no concatenation.** The raw frames of T1, T2, T3 must **never** be
joined end-to-end. Each single track is parsed independently, has its
thermal-field features extracted independently, and only afterwards are the three
repeats aggregated per condition (mean / std / coefficient of variation).

`track_order`: T1 = 1, T2 = 2, T3 = 3. If real acquisition-order evidence
contradicts this, **stop and report** — do not silently change it.

### Substrate & logical plate mapping (CONFIRMED)

Each of the 19 conditions uses **one independent 316L plate**
(`40 × 16 × 8 mm`); its three single tracks **T1/T2/T3 are made on that same
plate** in order T1 → T2 → T3, with **120 s cooling between consecutive tracks**.
A logical plate id `Plate-<condition_id>` (e.g. `R5_T1/T2/T3 → Plate-R5`) is used
for data management — `plate_id_type = logical_condition_based_identifier`; it
does **not** claim the physical plates were stamped with these numbers.

> **Design limitation (must be stated in analysis).** Because each condition has
> exactly one plate and T1/T2/T3 are **in-plate repeats** (NOT three independent
> plates), the **condition (process) effect and plate-to-plate individual
> variation cannot be fully separated**. Do not describe T1/T2/T3 as three fully
> independent substrate replicates.

## 6. Current phase: whole-track thermal-field analysis only

The current research line is:

```
process parameters
  -> 57 independent single-track temperature fields
  -> per-track thermal-field features
  -> per-condition aggregation over T1/T2/T3 (mean, std, CV)
  -> 19 condition-level thermal responses
  -> response surface & magnetic-field effect analysis
```

The current phase does **not**: slice tracks, measure cross-sections, build
section quality labels, run quality classification, or run scripts 13–16.

### Data units

- **Track level (processing unit):** one single track = one temperature-field
  sample. All raw temperature is float32 **degrees Celsius** (`raw/10`), shape
  **N × H × W** (N frames, H=512, W=640). Per-track features are extracted from
  this whole-track sequence.
- **Condition level (statistical aggregation unit):** one of the 19 conditions =
  aggregate of its T1/T2/T3 per-track features, summarized by **mean, standard
  deviation, and coefficient of variation (CV = std / mean)**.

### Subsequent data-processing order

1. Read raw `.xtherm` per track from the canonical data source
   (`configs/experiments.yaml` → `raw_data_root`).
2. Convert raw `.xtherm` → float32 Celsius matrices (N × H × W), per track. *(not run yet)*
3. ROI crop per track. *(not run yet)*
4. Extract whole-track thermal-field features, **one row per track** (57 rows). *(not run yet)*
5. Aggregate per condition over T1/T2/T3 → mean / std / CV (19 rows). *(not run yet)*
6. Response-surface and with-/without-magnetic-field comparison. *(not run yet)*

> Steps 2–6 are **not executed** in this task. This task only records the formal
> plan, writes the machine-readable config, and builds the local metadata map.

## 7. Later phase (not now)

Once cross-section quality labels exist:

```
thermal-field features + cross-section quality labels
  -> forming-quality prediction
```

Scripts 13–16 (section-level ML) belong to this later phase. They are **kept** in
the repository but are **currently not run** (see `README.md` and `CLAUDE.md`).

## 8. Data source & metadata

- Canonical raw data (formal source): `D:/WenDuChang-data-repo/raw_xtherm`
  (private Git-LFS data repository).
- Do **not** use `D:/WenDuChang/data/raw_xtherm` as the formal batch source, and
  exclude the early-test copy `D:/WenDuChang/data/raw_xtherm/dataset` entirely.
- The per-track metadata map is generated locally to
  `data/metadata/experiment_master.csv` (57 rows) by
  `scripts/00_build_experiment_master.py`. That CSV is **local only** and is not
  committed (the repo `.gitignore` excludes `*.csv`).

### Physical calibration (CONFIRMED)

Formal spatial scale is **150.2 px = 5 mm → `pixel_size_x_mm = pixel_size_y_mm =
0.0332889481` mm/px**, `pixel_area_mm2 = 0.0011081541` mm² (user-confirmed,
`calibration_id: formal_150p2px_5mm`). Source: `configs/physical_calibration.yaml`
(loader `src/config/physical_calibration.py`); every track in
`experiment_master.csv` carries these. X and Y share one scale
(`isotropic_scaling_assumed: true`; anisotropy not yet verified). The legacy pilot
`0.03128` mm/px (95.9 px = 3 mm) is **not** used for formal processing. The
19-condition process parameters are user-confirmed
(`process_parameter_status: confirmed`). See
`docs/physical_calibration_and_process_parameters_to_confirm.md`.

### Metadata status after the freeze

Now **confirmed** (single source of truth in the two config files; mirrored into
every row of `experiment_master.csv`):

- `frame_rate_fps = 52` (confirmed experimental setting), effective-frame rule
  (exclude startup frame 1; effective = `frames[1:]`);
- `scan_axis = y`, `image_scan_direction = upward`,
  `array_scan_direction = decreasing_row_index`, `physical_to_array_y_sign = −1`;
- camera valid range **300–1800 °C** + four-state masking policy;
- working distance **300 mm** (protective-window-to-molten-pool);
- laser spot **1 mm**, defocus **+14 mm**, powder-feed **setpoint 40 g/min**;
- substrate **316L 40×16×8 mm**, one plate per condition, `Plate-<condition_id>`,
  120 s in-plate inter-track cooling.

Still **not recorded** (kept null — never guessed): `emissivity`,
`transmission`, `exposure_time_us`, `lens_model`. The per-track effective
**end** uses the rule `last_available_frame` (no per-track numeric end yet). See
`docs/physical_calibration_and_process_parameters_to_confirm.md`.
