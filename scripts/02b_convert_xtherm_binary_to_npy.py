"""
02b_convert_xtherm_binary_to_npy.py

Legacy-compatible single-directory conversion of binary WeldStudio ``.xtherm``
frames into one stacked ``N x H x W`` float32 Celsius matrix plus JSON metadata.

The verified parser is shared with the formal 57-track converter through
``src/conversion/xtherm_binary.py``. Binary layout, image dimensions,
temperature scaling, camera-valid range, and QC thresholds are loaded from the
authoritative ``configs/xtherm_format.yaml``.

``configs/default.yaml`` is optional and legacy-only here: it supplies the
historical pilot input/output paths when explicit CLI paths are not provided.
Its duplicated binary-format values are ignored.

Examples:

    python scripts/02b_convert_xtherm_binary_to_npy.py
    python scripts/02b_convert_xtherm_binary_to_npy.py ^
        --input-dir data/raw_xtherm/example ^
        --output-npy data/exported/npy/example.npy ^
        --output-meta data/exported/npy/example_meta.json ^
        --sample-id example

This script is read-only with respect to source ``.xtherm`` files.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from config.xtherm_format import load_xtherm_format
from conversion.xtherm_binary import (  # noqa: F401 - compatibility re-export
    XthermSizeError,
    build_numpy_dtype,
    convert_xtherm_dir,
    list_xtherm_files,
    read_xtherm_frame,
)
from utils.config import load_config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--format-config",
        default="configs/xtherm_format.yaml",
        help="Authoritative formal XTherm binary-format YAML.",
    )
    parser.add_argument(
        "--config",
        dest="legacy_config",
        default="configs/default.yaml",
        help=(
            "Optional legacy YAML used only for historical input/output paths. "
            "Its binary-format values are ignored."
        ),
    )
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-npy", default=None)
    parser.add_argument("--output-meta", default=None)
    parser.add_argument("--sample-id", default=None)
    return parser.parse_args(argv)


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else _ROOT / path


def _load_legacy_paths(path: str | None) -> dict:
    """Read legacy pilot paths only; never use its duplicated format fields."""
    if path is None:
        return {}
    config_path = _resolve_repo_path(path)
    if not config_path.is_file():
        return {}
    config = load_config(str(config_path))
    block = config.get("xtherm_binary", {})
    return {
        "input_dir": block.get("input_dir"),
        "output_npy": block.get("output_npy"),
        "output_meta": block.get("output_meta"),
        "sample_id": block.get("sample_id"),
    }


def _require_value(name: str, explicit, fallback):
    value = explicit if explicit is not None else fallback
    if value in (None, ""):
        raise ValueError(
            f"{name} is required. Provide it explicitly or define the "
            "legacy path in configs/default.yaml."
        )
    return value


def _atomic_save_npy(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    try:
        with temporary.open("wb") as stream:
            np.save(stream, data)
        reloaded = np.load(temporary, mmap_mode="r")
        if reloaded.shape != data.shape or reloaded.dtype != np.float32:
            raise ValueError("Temporary NPY verification failed.")
        del reloaded
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv=None):
    args = parse_args(argv)
    formal = load_xtherm_format(_resolve_repo_path(args.format_config))
    legacy_paths = _load_legacy_paths(args.legacy_config)

    input_dir = _resolve_repo_path(
        _require_value(
            "--input-dir",
            args.input_dir,
            legacy_paths.get("input_dir"),
        )
    )
    output_npy = _resolve_repo_path(
        _require_value(
            "--output-npy",
            args.output_npy,
            legacy_paths.get("output_npy"),
        )
    )
    output_meta = _resolve_repo_path(
        _require_value(
            "--output-meta",
            args.output_meta,
            legacy_paths.get("output_meta"),
        )
    )
    sample_id = str(
        _require_value(
            "--sample-id",
            args.sample_id,
            legacy_paths.get("sample_id"),
        )
    )

    print(
        "[02b] Binary .xtherm -> NPY conversion "
        "(formal format config; source files read-only)"
    )
    print("-" * 72)

    data, files = convert_xtherm_dir(
        input_dir,
        width=formal.width_px,
        height=formal.height_px,
        header_bytes=formal.header_bytes,
        dtype_name=formal.raw_dtype,
        endian=formal.byte_order,
        scale_factor=formal.scale_C_per_count,
        offset_C=formal.offset_C,
        recursive=True,
    )

    finite = data[np.isfinite(data)]
    t_min = float(finite.min()) if finite.size else float("nan")
    t_max = float(finite.max()) if finite.size else float("nan")
    t_mean = float(finite.mean()) if finite.size else float("nan")
    zero_ratio = float(np.count_nonzero(data == 0.0) / data.size)
    above_range_count = int(
        np.count_nonzero(
            (data > formal.camera_valid_temperature_max_C)
            & (data < formal.hard_saturation_threshold_C)
        )
    )
    hard_saturation_count = int(
        np.count_nonzero(data >= formal.hard_saturation_threshold_C)
    )

    _atomic_save_npy(output_npy, data)

    metadata = {
        "sample_id": sample_id,
        "source_dir": str(input_dir),
        "frame_count": len(files),
        "first_file": Path(files[0]).name,
        "last_file": Path(files[-1]).name,
        "width_px": formal.width_px,
        "height_px": formal.height_px,
        "header_bytes": formal.header_bytes,
        "raw_dtype": formal.raw_dtype,
        "byte_order": formal.byte_order,
        "scale_C_per_count": formal.scale_C_per_count,
        "offset_C": formal.offset_C,
        "expected_file_size_bytes": formal.expected_file_size_bytes,
        "camera_valid_temperature_min_C": (
            formal.camera_valid_temperature_min_C
        ),
        "camera_valid_temperature_max_C": (
            formal.camera_valid_temperature_max_C
        ),
        "binary_qc_gross_upper_limit_C": (
            formal.binary_qc_gross_upper_limit_C
        ),
        "hard_saturation_threshold_C": (
            formal.hard_saturation_threshold_C
        ),
        "hard_saturation_value_C": formal.hard_saturation_value_C,
        "output_file": str(output_npy),
        "output_shape": list(data.shape),
        "output_dtype": "float32",
        "unit": "Celsius",
        "minimum_temperature_C": t_min,
        "maximum_temperature_C": t_max,
        "mean_temperature_C": t_mean,
        "zero_pixel_ratio": zero_ratio,
        "above_range_pixel_count": above_range_count,
        "hard_saturation_pixel_count": hard_saturation_count,
        "created_by": "scripts/02b_convert_xtherm_binary_to_npy.py",
        "formal_format_config": str(
            _resolve_repo_path(args.format_config)
        ),
        "legacy_path_config": (
            str(_resolve_repo_path(args.legacy_config))
            if args.legacy_config
            else None
        ),
    }
    _atomic_write_json(output_meta, metadata)

    print(f"[02b] loaded frames     : {len(files)}")
    print(f"[02b] first filename    : {metadata['first_file']}")
    print(f"[02b] last filename     : {metadata['last_file']}")
    print(
        f"[02b] output shape      : {data.shape} "
        "(N x H x W, float32 Celsius)"
    )
    print(
        "[02b] min/max/mean temp : "
        f"{t_min:.1f} / {t_max:.1f} / {t_mean:.2f} C"
    )
    print(f"[02b] zero pixel ratio  : {zero_ratio:.6f}")
    print(f"[02b] above-range pixels: {above_range_count}")
    print(f"[02b] hard-sat. pixels  : {hard_saturation_count}")
    print(f"[02b] output_npy        : {output_npy}")
    print(f"[02b] output_meta       : {output_meta}")

    if t_max > formal.camera_valid_temperature_max_C:
        print(
            "[02b] NOTE: one or more pixels exceed the camera-valid "
            f"quantitative maximum of "
            f"{formal.camera_valid_temperature_max_C:.0f} C. "
            "They are reported as above-range, not interpreted as true "
            "temperatures."
        )
    if t_max > formal.binary_qc_gross_upper_limit_C:
        print(
            "[02b] WARNING: maximum temperature exceeds the gross binary-QC "
            f"limit of {formal.binary_qc_gross_upper_limit_C:.0f} C."
        )
    if hard_saturation_count:
        print(
            "[02b] WARNING: hard-saturation pixels were detected at or above "
            f"{formal.hard_saturation_threshold_C:.0f} C."
        )
    if zero_ratio > formal.zero_ratio_note_threshold:
        print(
            f"[02b] NOTE: zero pixel ratio {zero_ratio:.6f} exceeds "
            f"{formal.zero_ratio_note_threshold:.3f}; edge zeros may be "
            "invalid pixels or background."
        )

    print("-" * 72)
    print(
        "[02b] Raw .xtherm files were only read and were never modified, "
        "moved, renamed, or deleted."
    )
    print(
        "[02b] Output is already Celsius. Do not apply the raw-count scale "
        "again in downstream legacy conversion."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
