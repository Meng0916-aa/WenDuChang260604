# Section Quality Label Template

Cross-section quality labels for the **section-level** ML main line (scripts
`13 → 14 → 15 → 16`). Each polished cross-section is one independently-measured
observation, keyed by `sample_id` (the same id used in
`data/metadata/section_plan.csv`, e.g. `R01_T1_S1`).

> The actual label file is **LOCAL only** and is never uploaded to GitHub
> (`*.csv` / `*.xlsx` are git-ignored). Put it at:
> `data/metadata/section_quality_labels.csv` (preferred) or
> `data/metadata/section_quality_labels.xlsx` (needs `openpyxl`).

## Fields (one row per cross-section)

```
sample_id,experiment_id,track_id,section_id,section_position_mm,H_mm,W_mm,D_mm,theta_left_deg,theta_right_deg,defect_presence,quality_label,notes
```

| Field | Meaning / unit |
|-------|----------------|
| `sample_id` | section id, must match `section_plan.csv` (e.g. `R01_T1_S1`) |
| `experiment_id` | experiment / run id (group key) |
| `track_id` | cladding track id (e.g. `T1`) |
| `section_id` | section id within the track (e.g. `S1`) |
| `section_position_mm` | section position along the track (mm) |
| `H_mm` | cladding (bead) height above the substrate (mm) |
| `W_mm` | cladding (bead) width (mm) |
| `D_mm` | melted / penetration depth into the substrate (mm) |
| `theta_left_deg` | left wetting (contact) angle (degrees) |
| `theta_right_deg` | right wetting (contact) angle (degrees) |
| `defect_presence` | `0` = no obvious defect, `1` = obvious pore / crack / lack-of-fusion |
| `quality_label` | `Good` / `Bad` — fill manually OR leave blank for auto-rule |
| `notes` | free text |

## Derived columns (computed by script 14 when not supplied)

```
dilution_rate     = D_mm / (D_mm + H_mm) * 100        # PERCENT
aspect_ratio      = W_mm / H_mm
wetting_angle_avg = (theta_left_deg + theta_right_deg) / 2
wetting_angle_diff= abs(theta_left_deg - theta_right_deg)
```

You may pre-compute any of these columns in the label file; script 14 only fills
the ones that are missing/blank and keeps your values otherwise.

> **Unit note:** here `dilution_rate` is a **percent** (0–100), so the rule
> bounds below are `30..50`. This differs from the experiment-level main line
> (`docs/quality_label_template.md`), where dilution is a fraction `0.30..0.50`.

## Required measurements (build stops if any are missing)

These six columns must be present and **numeric** for every section that has
features — empty / NaN / non-numeric values are **never** auto-labelled "Bad":

```
H_mm   W_mm   D_mm   theta_left_deg   theta_right_deg   defect_presence
```

If any are missing, script 14 stops without writing
`section_ml_dataset.csv` and reports each offending row, e.g.:

```
Missing required section label values:
sample_id=P300_V400_F40_B60_T1_S1 missing H_mm,W_mm,D_mm,theta_left_deg,theta_right_deg,defect_presence
[14] Fill data/metadata/section_quality_labels.csv with measured cross-section values before running 14.
```

So a freshly-copied template (with blank measurements) will **not** produce a
polluted all-"Bad" dataset — fill in the measured values first.

## Auto Good/Bad rule

Once the required measurements are present, `quality_label = Good` **iff all** of:

```
30 <= dilution_rate     <= 50      (percent)
3  <= aspect_ratio      <= 6
30 <= wetting_angle_avg <= 55
defect_presence == 0
```

otherwise `quality_label = Bad`. The numeric bounds are configurable in
`configs/default.yaml` → `section_quality_labels.quality_rule.*`. If
`quality_label` is filled in manually it is preserved; the continuous derived
labels (`dilution_rate`, `aspect_ratio`, `wetting_angle_avg`,
`wetting_angle_diff`) are still computed either way.

## Template (CSV header + example rows — fill with REAL measurements)

```csv
sample_id,experiment_id,track_id,section_id,section_position_mm,H_mm,W_mm,D_mm,theta_left_deg,theta_right_deg,defect_presence,quality_label,notes
R01_T1_S1,R01_P300_V400_B60,T1,S1,6,1.20,5.00,0.90,42,45,0,,
R01_T1_S2,R01_P300_V400_B60,T1,S2,12,1.10,4.80,1.30,58,61,0,,high dilution
R02_T1_S1,R02_P360_V400_B60,T1,S1,6,1.30,5.20,1.00,38,41,1,Bad,visible pore
```

(Numbers are illustrative placeholders, not measurements. Leaving
`quality_label` blank lets script 14 fill it from the rule.)

## How labels feed the pipeline

- Script `14` merges these labels with
  `results/tables/local_section_features.csv` on `sample_id` into
  `results/tables/section_ml_dataset.csv`. Every feature `sample_id` **must**
  have a label row (script 14 errors and lists any that don't); extra label rows
  with no features are reported as a warning.
- Script `15` predicts the regression targets
  (`dilution_rate`, `aspect_ratio`, `wetting_angle_avg`, `wetting_angle_diff`)
  and the classification target (`quality_label`), grouped by `experiment_id`.
