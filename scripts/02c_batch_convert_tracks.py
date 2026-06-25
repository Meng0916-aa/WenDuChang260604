"""
02c_batch_convert_tracks.py

Per-single-track batch conversion of binary .xtherm frames into one float32
Celsius matrix per track. Driven by the formal design + the local track list:

    configs/experiments.yaml  (raw_data_root, design)  ->  experiment_master.csv
    -> read a track folder -> natural-sort .xtherm -> stack (N, 512, 640) float32
    -> data/processed/matrix/<sample_id>.npy + data/processed/matrix_meta/<sample_id>.json

The processing unit is ONE single track (e.g. R1_T1). T1/T2/T3 are NEVER
concatenated. The verified parse algorithm is reused from
``src/conversion/xtherm_binary.py`` (no second copy). The formal binary-format
parameters (56-byte header, 640x512, little-endian uint16, scale 0.1) come from
``configs/xtherm_format.yaml`` through ``src/config/xtherm_format.py``. The
legacy ``configs/default.yaml`` format block and its ``dataset`` paths are NOT
used by this formal 57-track converter.

READ-ONLY on raw data: source .xtherm files are only read, never written, moved,
renamed or deleted. The early-test directory ``dataset`` is refused.

Examples (PowerShell):
    python scripts/02c_batch_convert_tracks.py --master-csv data/metadata/experiment_master.csv --sample-id R1_T1
    python scripts/02c_batch_convert_tracks.py --master-csv data/metadata/experiment_master.csv --sample-ids R1_T1 R13_T1 R3_T1
    python scripts/02c_batch_convert_tracks.py --master-csv data/metadata/experiment_master.csv --all --dry-run
    python scripts/02c_batch_convert_tracks.py --master-csv data/metadata/experiment_master.csv --select-representative --dry-run
"""

import os
import sys
import csv
import json
import time
import glob
import hashlib
import argparse
import statistics

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from config.xtherm_format import load_xtherm_format
from conversion.xtherm_binary import (
    XthermSizeError,
    build_numpy_dtype,
    list_xtherm_files,
    read_xtherm_frame,
    expected_frame_size,
    natural_sort_key,
)

FORBIDDEN_OUTPUT_NAMES = {"dataset"}  # never (re)create dataset.npy here


# --------------------------------------------------------------------------
# Config / master loading
# --------------------------------------------------------------------------

def load_format(format_config_path):
    """Load the formal verified XTherm binary-format configuration.

    The authoritative source is ``configs/xtherm_format.yaml``.  The returned
    dictionary preserves the keys expected by the conversion and QC code while
    keeping the camera measurement range, gross QC limit, and hard-saturation
    threshold as separate concepts.
    """
    cfg = load_xtherm_format(format_config_path)

    fmt = {
        "width": int(cfg.width_px),
        "height": int(cfg.height_px),
        "header_bytes": int(cfg.header_bytes),
        "dtype": str(cfg.raw_dtype),
        "endian": str(cfg.byte_order),
        "scale_factor": float(cfg.scale_C_per_count),
        "offset_C": float(cfg.offset_C),
        "camera_valid_temperature_min_C": float(
            cfg.camera_valid_temperature_min_C
        ),
        "camera_valid_temperature_max_C": float(
            cfg.camera_valid_temperature_max_C
        ),
        "binary_qc_gross_upper_limit_C": float(
            cfg.binary_qc_gross_upper_limit_C
        ),
        "hard_saturation_threshold_C": float(
            cfg.hard_saturation_threshold_C
        ),
        "hard_saturation_value_C": float(
            cfg.hard_saturation_value_C
        ),
        "zero_ratio_note_threshold": float(
            getattr(cfg, "zero_ratio_note_threshold", 0.05)
        ),
    }

    # Backward-compatible QC alias.  This is a gross anomaly threshold, not
    # the camera's valid quantitative temperature maximum.
    fmt["valid_max"] = fmt["binary_qc_gross_upper_limit_C"]

    fmt["np_dtype"] = build_numpy_dtype(fmt["dtype"], fmt["endian"])
    fmt["expected_size"] = expected_frame_size(
        fmt["width"],
        fmt["height"],
        fmt["header_bytes"],
        fmt["np_dtype"],
    )

    if fmt["expected_size"] != int(cfg.expected_file_size_bytes):
        raise ValueError(
            "XTherm format mismatch: computed frame size "
            f"{fmt['expected_size']} != configured "
            f"{cfg.expected_file_size_bytes}"
        )

    return fmt


def read_master(csv_path):
    """Read experiment_master.csv into a list of row dicts (preserving order)."""
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(_ROOT, csv_path)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"master csv not found: {csv_path}")
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"master csv is empty: {csv_path}")
    return rows


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def assert_not_dataset(path):
    """Refuse anything under the early-test 'dataset' directory."""
    parts = {p.lower() for p in str(path).replace("\\", "/").split("/")}
    if "dataset" in parts:
        raise ValueError(f"refusing early-test 'dataset' path: {path}")


def frame_index(name):
    """Integer frame index parsed from a filename's last digit run, or None."""
    digits = "".join(c if c.isdigit() else " " for c in os.path.basename(name)).split()
    return int(digits[-1]) if digits else None


def source_manifest_sha256(files):
    """SHA-256 over natural-sorted 'basename<TAB>size' lines (name+size manifest).

    Cheap (stat only) and deterministic; detects added/removed/resized source
    files for safe skip/resume decisions. Not a content hash.
    """
    h = hashlib.sha256()
    for p in sorted(files, key=lambda x: natural_sort_key(os.path.basename(x))):
        h.update(f"{os.path.basename(p)}\t{os.path.getsize(p)}\n".encode("utf-8"))
    return h.hexdigest()


def output_paths(output_root, sample_id):
    if not os.path.isabs(output_root):
        output_root = os.path.join(_ROOT, output_root)
    matrix_dir = os.path.join(output_root, "matrix")
    meta_dir = os.path.join(output_root, "matrix_meta")
    return (os.path.join(matrix_dir, f"{sample_id}.npy"),
            os.path.join(meta_dir, f"{sample_id}.json"))


def index_master(rows):
    return {r["sample_id"]: r for r in rows}


# --------------------------------------------------------------------------
# Validation (section six: 10 checks)
# --------------------------------------------------------------------------

def validate_track(row, fmt):
    """Run all pre-conversion checks. Returns dict(ok, reasons, files)."""
    reasons = []
    folder = row["raw_folder"]
    sample_id = row["sample_id"]

    # (10) never process the early-test dataset directory
    try:
        assert_not_dataset(folder)
    except ValueError as e:
        reasons.append(str(e))

    # (1) raw directory exists
    if not os.path.isdir(folder):
        reasons.append(f"raw_folder missing: {folder}")
        return {"ok": False, "reasons": reasons, "files": []}

    # (2) session.xml exists
    if not os.path.isfile(os.path.join(folder, "session.xml")):
        reasons.append("session.xml missing")

    files = list_xtherm_files(folder, recursive=False)

    # (9) session.xml is never treated as a temperature frame
    if any(os.path.basename(f).lower() == "session.xml" for f in files):
        reasons.append("session.xml present in frame list (must be excluded)")

    if not files:
        reasons.append("no .xtherm files found")
        return {"ok": False, "reasons": reasons, "files": []}

    # (3) count matches master
    try:
        expected_n = int(row["xtherm_count"])
    except (KeyError, ValueError):
        expected_n = None
    if expected_n is not None and len(files) != expected_n:
        reasons.append(f"count mismatch: found {len(files)} != master {expected_n}")

    # (8) no duplicate frame indices
    idxs = [frame_index(f) for f in files]
    if any(i is None for i in idxs):
        reasons.append("a filename has no parseable frame index")
    else:
        if len(set(idxs)) != len(idxs):
            reasons.append("duplicate frame indices")
        # (4) numbering contiguous
        elif max(idxs) - min(idxs) + 1 != len(idxs):
            reasons.append(
                f"non-contiguous numbering: {min(idxs)}..{max(idxs)} for {len(idxs)} files")

    # (5,6,7) every file is exactly one valid frame, none empty
    for f in files:
        size = os.path.getsize(f)
        if size == 0:
            reasons.append(f"empty file: {os.path.basename(f)}")
        elif size != fmt["expected_size"]:
            reasons.append(
                f"bad size {size} (expected {fmt['expected_size']}): {os.path.basename(f)}")

    return {"ok": len(reasons) == 0, "reasons": reasons, "files": files}


# --------------------------------------------------------------------------
# Conversion (sections five/seven/eight)
# --------------------------------------------------------------------------

def _empty_meta(row, status, reasons, fmt):
    return {
        "sample_id": row["sample_id"],
        "condition_id": row.get("condition_id", ""),
        "track_id": row.get("track_id", ""),
        "source_folder": row.get("raw_folder", ""),
        "conversion_status": status,
        "reasons": reasons,
    }


def convert_track(row, fmt, output_root, overwrite=False):
    """Convert ONE track. Returns a meta dict including conversion_status.

    Statuses: 'converted', 'skipped', 'fail'. Atomic tmp write; no partial
    official output is ever left behind. Raw files are only read.

    ``fmt`` normally comes from :func:`load_format`. A few unit tests and
    legacy callers construct a minimal format dictionary directly, so formal
    metadata fields are filled with safe defaults here without changing the
    verified binary parsing parameters.
    """
    fmt = dict(fmt)
    fmt.setdefault("offset_C", 0.0)
    fmt.setdefault("camera_valid_temperature_min_C", 300.0)
    fmt.setdefault("camera_valid_temperature_max_C", 1800.0)
    fmt.setdefault(
        "binary_qc_gross_upper_limit_C",
        float(fmt.get("valid_max", 3000.0)),
    )
    fmt.setdefault("hard_saturation_threshold_C", 6500.0)
    fmt.setdefault("hard_saturation_value_C", 6553.5)

    sample_id = row["sample_id"]
    if sample_id.lower() in FORBIDDEN_OUTPUT_NAMES:
        return _empty_meta(row, "fail", [f"forbidden sample_id: {sample_id}"], fmt)

    out_npy, out_meta = output_paths(output_root, sample_id)
    # Hard guard: we must produce <sample_id>.npy, never dataset.npy
    assert os.path.basename(out_npy) == f"{sample_id}.npy"
    if os.path.basename(out_npy).lower() == "dataset.npy":
        return _empty_meta(row, "fail", ["refusing to write dataset.npy"], fmt)

    # Validate before touching anything
    v = validate_track(row, fmt)
    files = v["files"]
    if not v["ok"]:
        return _empty_meta(row, "fail", v["reasons"], fmt)

    manifest_sha = source_manifest_sha256(files)

    # Skip / overwrite / mismatch policy (section seven)
    if os.path.exists(out_npy) and not overwrite:
        prior = None
        if os.path.isfile(out_meta):
            try:
                with open(out_meta, "r", encoding="utf-8") as f:
                    prior = json.load(f)
            except (ValueError, OSError):
                prior = None
        consistent = (
            prior is not None
            and prior.get("sample_id") == sample_id
            and prior.get("source_manifest_sha256") == manifest_sha
            and prior.get("source_folder") == row["raw_folder"]
            and list(prior.get("output_shape", [])[:1]) == [len(files)]
        )
        if consistent:
            return {**prior, "conversion_status": "skipped"}
        return _empty_meta(
            row, "fail",
            ["existing output present but metadata inconsistent; "
             "use --overwrite to replace"], fmt)

    if os.path.exists(out_npy) and overwrite:
        # Re-confirm identity before overwriting.
        if os.path.isfile(out_meta):
            try:
                with open(out_meta, "r", encoding="utf-8") as f:
                    prior = json.load(f)
                if prior.get("sample_id") not in (None, sample_id):
                    return _empty_meta(
                        row, "fail",
                        [f"overwrite identity mismatch: existing meta sample_id="
                         f"{prior.get('sample_id')}"], fmt)
            except (ValueError, OSError):
                pass

    os.makedirs(os.path.dirname(out_npy), exist_ok=True)
    os.makedirs(os.path.dirname(out_meta), exist_ok=True)
    tmp_npy = out_npy + ".tmp"

    t0 = time.perf_counter()
    H, W = fmt["height"], fmt["width"]
    data = np.empty((len(files), H, W), dtype=np.float32)
    try:
        for i, path in enumerate(files):
            data[i] = read_xtherm_frame(
                path, W, H, fmt["header_bytes"], fmt["np_dtype"], fmt["scale_factor"])

        nan_count = int(np.count_nonzero(np.isnan(data)))
        inf_count = int(np.count_nonzero(np.isinf(data)))
        finite = data[np.isfinite(data)]
        t_min = float(finite.min()) if finite.size else float("nan")
        t_max = float(finite.max()) if finite.size else float("nan")
        t_mean = float(finite.mean()) if finite.size else float("nan")
        zero_ratio = float(np.count_nonzero(data == 0.0) / data.size)

        # Atomic write: open handle so np.save does NOT append .npy to the tmp name.
        with open(tmp_npy, "wb") as fh:
            np.save(fh, data)

        # Verify the tmp file BEFORE promoting it.
        reloaded = np.load(tmp_npy, mmap_mode="r")
        if tuple(reloaded.shape) != (len(files), H, W):
            raise ValueError(f"reloaded shape {reloaded.shape} != {(len(files), H, W)}")
        if reloaded.dtype != np.float32:
            raise ValueError(f"reloaded dtype {reloaded.dtype} != float32")
        del reloaded

        os.replace(tmp_npy, out_npy)   # atomic promotion
    except Exception as e:  # noqa: BLE001 — record reason, never leave partial output
        if os.path.exists(tmp_npy):
            try:
                os.remove(tmp_npy)
            except OSError:
                pass
        return _empty_meta(row, "fail", [f"{type(e).__name__}: {e}"], fmt)
    finally:
        del data

    elapsed = time.perf_counter() - t0
    src_total = sum(os.path.getsize(f) for f in files)
    meta = {
        "sample_id": sample_id,
        "condition_id": row.get("condition_id", ""),
        "track_id": row.get("track_id", ""),
        "source_folder": row["raw_folder"],
        "source_file_count": len(files),
        "first_source_file": os.path.basename(files[0]),
        "last_source_file": os.path.basename(files[-1]),
        "source_total_size_bytes": int(src_total),
        "output_file": out_npy,
        "output_shape": [len(files), H, W],
        "output_dtype": "float32",
        "temperature_unit": "Celsius",
        "temperature_scale_C_per_count": fmt["scale_factor"],
        "temperature_scale": fmt["scale_factor"],
        "temperature_offset_C": fmt["offset_C"],
        "camera_valid_temperature_min_C": (
            fmt["camera_valid_temperature_min_C"]
        ),
        "camera_valid_temperature_max_C": (
            fmt["camera_valid_temperature_max_C"]
        ),
        "binary_qc_gross_upper_limit_C": (
            fmt["binary_qc_gross_upper_limit_C"]
        ),
        "hard_saturation_threshold_C": (
            fmt["hard_saturation_threshold_C"]
        ),
        "hard_saturation_value_C": fmt["hard_saturation_value_C"],
        "header_bytes": fmt["header_bytes"],
        "image_height": H,
        "image_width": W,
        "minimum_temperature_C": t_min,
        "maximum_temperature_C": t_max,
        "mean_temperature_C": t_mean,
        "zero_pixel_ratio": zero_ratio,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "conversion_time_seconds": round(elapsed, 3),
        "source_manifest_sha256": manifest_sha,
        "conversion_status": "converted",
    }
    tmp_meta = out_meta + ".tmp"
    with open(tmp_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    os.replace(tmp_meta, out_meta)
    return meta


# --------------------------------------------------------------------------
# Representative-sample selection (section nine)
# --------------------------------------------------------------------------

def select_representative(rows):
    """Pick one track per scan speed: frame count closest to that speed's median,
    preferring T1 on ties. Returns list of dicts (speed, sample_id, reason)."""
    cond_index = {}
    for r in rows:
        cond_index.setdefault(r["condition_id"], len(cond_index))

    by_speed = {}
    for r in rows:
        by_speed.setdefault(int(r["scan_speed_mm_min"]), []).append(r)

    chosen = []
    for speed in sorted(by_speed):
        group = by_speed[speed]
        counts = [int(r["xtherm_count"]) for r in group]
        median = float(statistics.median(counts))

        def key(r):
            dist = abs(int(r["xtherm_count"]) - median)
            track_pref = 0 if r["track_id"] == "T1" else 1
            return (dist, track_pref, int(r.get("track_order", 99)),
                    cond_index[r["condition_id"]])

        best = sorted(group, key=key)[0]
        reason = (f"speed {speed} mm/min: median frames={median:g}; chosen "
                  f"{best['sample_id']} ({best['xtherm_count']} frames, "
                  f"|delta|={abs(int(best['xtherm_count']) - median):g}, "
                  f"prefer T1)")
        chosen.append({"speed": speed, "sample_id": best["sample_id"],
                       "row": best, "reason": reason})
    return chosen


# --------------------------------------------------------------------------
# Target enumeration + dry run
# --------------------------------------------------------------------------

def enumerate_targets(rows, args):
    """Resolve which master rows to process from the CLI selectors."""
    by_id = index_master(rows)
    if args.all:
        return list(rows)
    if args.select_representative:
        return [c["row"] for c in select_representative(rows)]
    if args.sample_id:
        if args.sample_id not in by_id:
            raise KeyError(f"sample_id not in master: {args.sample_id}")
        return [by_id[args.sample_id]]
    if args.sample_ids:
        missing = [s for s in args.sample_ids if s not in by_id]
        if missing:
            raise KeyError(f"sample_ids not in master: {missing}")
        return [by_id[s] for s in args.sample_ids]
    raise SystemExit("error: specify one of --sample-id / --sample-ids / --all / "
                     "--select-representative")


def dry_run(targets, fmt, output_root):
    """List planned work and disk estimate. Writes NOTHING."""
    H, W = fmt["height"], fmt["width"]
    bytes_per_frame = H * W * 4
    total_bytes = 0
    print(f"[02c] DRY-RUN: {len(targets)} track(s) planned (no .npy written)")
    print("-" * 78)
    for r in targets:
        folder = r["raw_folder"]
        n_files = len(glob.glob(os.path.join(folder, "*.xtherm"))) if os.path.isdir(folder) else 0
        out_npy, _ = output_paths(output_root, r["sample_id"])
        est = n_files * bytes_per_frame
        total_bytes += est
        try:
            disp = os.path.relpath(out_npy, _ROOT)
        except ValueError:           # different drive (e.g. temp on C:, repo on D:)
            disp = out_npy
        print(f"  {r['sample_id']:8s} files={n_files:>4}  -> {disp}"
              f"  (~{est/1e6:.1f} MB)")
    print("-" * 78)
    print(f"[02c] estimated total output: {total_bytes/1e9:.2f} GB "
          f"({total_bytes/2**30:.2f} GiB) across {len(targets)} track(s)")
    return total_bytes


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--master-csv", default="data/metadata/experiment_master.csv")
    p.add_argument(
        "--format-config",
        default="configs/xtherm_format.yaml",
        help=(
            "Formal YAML configuration containing the verified XTherm "
            "binary layout, image dimensions, temperature scaling, "
            "measurement range, and QC thresholds"
        ),
    )
    p.add_argument("--sample-id", default=None)
    p.add_argument("--sample-ids", nargs="+", default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--select-representative", action="store_true",
                   help="auto-pick one track per scan speed (median frame count)")
    p.add_argument("--output-root", default="data/processed")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--qc", action="store_true",
                   help="run conversion QC (plots + qc json) for each converted track")
    p.add_argument("--qc-root", default="results/qc/conversion")
    p.add_argument("--qc-summary", default="results/tables/pilot_conversion_qc.csv")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    fmt = load_format(args.format_config)
    rows = read_master(args.master_csv)

    if args.select_representative:
        print("[02c] Representative-sample selection (per scan speed):")
        for c in select_representative(rows):
            print(f"    {c['reason']}")
        print("-" * 78)

    targets = enumerate_targets(rows, args)

    if args.dry_run:
        dry_run(targets, fmt, args.output_root)
        print("[02c] DRY-RUN complete -- no data written.")
        return 0

    qc_rows = []
    results = []
    for r in targets:
        meta = convert_track(r, fmt, args.output_root, overwrite=args.overwrite)
        status = meta["conversion_status"]
        results.append(meta)
        if status == "converted":
            print(f"[02c] {r['sample_id']:8s} converted  shape={meta['output_shape']} "
                  f"min/max/mean={meta['minimum_temperature_C']:.1f}/"
                  f"{meta['maximum_temperature_C']:.1f}/{meta['mean_temperature_C']:.2f} C "
                  f"nan={meta['nan_count']} inf={meta['inf_count']}")
        elif status == "skipped":
            print(f"[02c] {r['sample_id']:8s} skipped (up-to-date)")
        else:
            print(f"[02c] {r['sample_id']:8s} FAIL: {meta.get('reasons')}")
            if not args.continue_on_error:
                print("[02c] stopping (use --continue-on-error to keep going).")
                return 1

        if args.qc and status in ("converted", "skipped"):
            from visualization.conversion_qc import run_conversion_qc
            qc = run_conversion_qc(meta, r, args.qc_root, fmt)
            qc_rows.append(qc)

    if args.qc and qc_rows:
        _write_qc_summary(qc_rows, results, args.qc_summary)
        print(f"[02c] QC summary -> {args.qc_summary}")

    print("-" * 78)
    print("[02c] Raw .xtherm files were only READ -- never modified, moved, or deleted.")
    return 0


def _write_qc_summary(qc_rows, metas, out_csv):
    meta_by_id = {m["sample_id"]: m for m in metas}
    if not os.path.isabs(out_csv):
        out_csv = os.path.join(_ROOT, out_csv)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    cols = ["sample_id", "condition_id", "track_id", "scan_speed_mm_min",
            "xtherm_count", "output_frame_count", "output_shape", "output_dtype",
            "min_temperature_C", "max_temperature_C", "mean_temperature_C",
            "zero_pixel_ratio", "nan_count", "inf_count", "possible_saturation",
            "possible_corruption", "conversion_status", "qc_status", "notes"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for qc in qc_rows:
            w.writerow({c: qc.get(c, "") for c in cols})


if __name__ == "__main__":
    raise SystemExit(main())
