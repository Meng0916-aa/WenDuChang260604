# Quality Label Template

Cladding quality labels come from **substrate cross-section measurements** (cut,
polish, etch, then measure the deposited bead geometry under a microscope). Each
experiment contributes **one** label row, keyed by `sample_id` (which must match
the `experiment_id` / ROI file stem used by scripts 03/10).

> The actual label file is LOCAL only and is **not** uploaded to GitHub
> (`*.csv` / `*.xlsx` are git-ignored). Put it at:
> `data/metadata/quality_labels.csv` (preferred) or
> `data/metadata/quality_labels.xlsx` (needs `openpyxl`).

## Fields (one row per experiment)

| Field | Meaning / unit |
|-------|----------------|
| `sample_id` | experiment id, must match the ROI/feature `experiment_id` (e.g. `B100_02`) |
| `cladding_height_H` | bead height H above the substrate (mm) |
| `cladding_width_W` | bead width W (mm) |
| `molten_depth_D` | melted/penetration depth D into the substrate (mm) |
| `wetting_angle_left` | left wetting (contact) angle theta_left (degrees) |
| `wetting_angle_right` | right wetting (contact) angle theta_right (degrees) |
| `dilution_rate` | D / (D + H) — derived (see below) |
| `aspect_ratio` | W / H — derived |
| `wetting_angle_avg` | (theta_left + theta_right) / 2 — derived |
| `quality_label` | `Good` / `Bad` classification label (see rule below) |
| `notes` | free text (defects, operator, anomalies, ...) |

## Derived formulas

```
dilution_rate     = D / (D + H)
aspect_ratio      = W / H
wetting_angle_avg = (theta_left + theta_right) / 2
```

You may pre-compute `dilution_rate`, `aspect_ratio`, `wetting_angle_avg` in the
label file, or leave only the raw measurements (`H`, `W`, `D`,
`wetting_angle_left/right`) and compute the derived columns yourself — the ML
scripts read whatever columns are present.

## Recommended quality criterion

`quality_label = Good` when **all** of:

```
0.30 <= dilution_rate     <= 0.50
3    <= aspect_ratio      <= 6
30   <= wetting_angle_avg <= 55
```

otherwise `quality_label = Bad`.

These bounds are configurable in `configs/default.yaml`
(`quality_labels.quality_rule.*`).

## Template (CSV header + example rows — fill with REAL measurements)

```csv
sample_id,cladding_height_H,cladding_width_W,molten_depth_D,wetting_angle_left,wetting_angle_right,dilution_rate,aspect_ratio,wetting_angle_avg,quality_label,notes
B0_01,1.20,5.00,0.90,42,45,0.428,4.17,43.5,Good,
B0_02,1.10,4.80,1.30,58,61,0.542,4.36,59.5,Bad,high dilution
B100_01,1.30,5.20,1.00,38,41,0.435,4.00,39.5,Good,
```

(Numbers above are illustrative placeholders, not measurements.)

## How labels feed the ML pipeline

- Script `11` merges these labels with `results/tables/thermal_field_features.csv`
  on `sample_id == experiment_id` into `ml_quality_dataset.csv`.
- Script `12` trains models on `quality_label` (classification) or on
  `dilution_rate` / `aspect_ratio` / `wetting_angle_avg` (regression).
