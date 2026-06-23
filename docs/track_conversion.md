# Per-Track .xtherm → Matrix Conversion (script 02c)

Converts each single track's binary `.xtherm` frames into one float32 Celsius
matrix `(N, 512, 640)`. The processing unit is **one single track**; `T1/T2/T3`
are never concatenated.

## Config alignment (single source of truth)

| Source | Provides |
|--------|----------|
| `configs/experiments.yaml` | formal 19-condition design, `raw_data_root`, tracks, fixed params |
| `data/metadata/experiment_master.csv` | per-track processing list (`sample_id`, `raw_folder`, `xtherm_count`, …) |
| `configs/default.yaml → xtherm_binary` | verified binary FORMAT only (56-byte header, 640×512, little-endian uint16, scale 0.1) |

The legacy `data/raw_xtherm/dataset` + `dataset.npy` in `default.yaml` stay as the
old single-dataset test path **only** — `02c` never uses them and refuses any
`dataset` path. The verified parse algorithm lives once in
`src/conversion/xtherm_binary.py` and is shared by `02b` and `02c`.

## Usage

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"

# one track
python scripts/02c_batch_convert_tracks.py --master-csv data/metadata/experiment_master.csv --sample-id R1_T1

# several tracks
python scripts/02c_batch_convert_tracks.py --master-csv data/metadata/experiment_master.csv --sample-ids R1_T1 R13_T1 R3_T1

# plan everything (lists 57 tracks + disk estimate, writes nothing)
python scripts/02c_batch_convert_tracks.py --master-csv data/metadata/experiment_master.csv --all --dry-run

# auto-pick 3 representative tracks (one per scan speed, median frame count) + QC
python scripts/02c_batch_convert_tracks.py --master-csv data/metadata/experiment_master.csv --select-representative --qc
```

Flags: `--sample-id`, `--sample-ids`, `--all`, `--select-representative`,
`--output-root`, `--dry-run`, `--overwrite`, `--continue-on-error`, `--qc`.

## Output (local only; git-ignored)

```
data/processed/matrix/<sample_id>.npy          # (N, 512, 640) float32 Celsius
data/processed/matrix_meta/<sample_id>.json    # per-track conversion metadata
results/qc/conversion/<sample_id>/             # QC plots + conversion_qc.json   (with --qc)
results/tables/pilot_conversion_qc.csv         # pilot QC summary               (with --qc)
```

## Safety properties

- **Read-only** on raw `.xtherm` (only opened for reading); never writes
  `dataset.npy`.
- Per-track pre-checks: folder + `session.xml` exist, count matches master,
  contiguous frame numbering, every file exactly 655416 bytes, no empty/duplicate
  files, `session.xml` excluded from frames, `dataset` refused. Any failure →
  `conversion_status = fail` with a recorded reason (never silently skipped).
- Natural (numeric) filename ordering: `2.xtherm` before `10.xtherm`.
- Atomic write: data goes to `<id>.npy.tmp`, is reloaded/verified
  (shape + dtype), then atomically renamed — an interrupted run never leaves a
  corrupt official `.npy`.
- Resume: existing output with matching metadata is **skipped**; a mismatch
  stops that sample unless `--overwrite` is given.
- One track in memory at a time.

## Scope / status

Current phase converts whole tracks only. ROI (`03`), thermal-field feature
extraction (`10`), section-level scripts `13–16`, and full-57 batch conversion
are **not run yet** in this phase.
