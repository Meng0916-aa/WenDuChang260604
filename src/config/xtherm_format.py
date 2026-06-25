"""Validated loader for the formal XTherm binary-format configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class XThermFormat:
    """Verified binary layout, temperature scaling, and QC thresholds."""

    width_px: int
    height_px: int
    header_bytes: int
    raw_dtype: str
    byte_order: str
    scale_C_per_count: float
    offset_C: float
    expected_file_size_bytes: int
    filename_extension: str
    require_continuous_numeric_sequence: bool
    exclude_filenames: tuple[str, ...]
    camera_valid_temperature_min_C: float
    camera_valid_temperature_max_C: float
    binary_qc_gross_upper_limit_C: float
    hard_saturation_threshold_C: float
    hard_saturation_value_C: float
    zero_ratio_note_threshold: float


def load_xtherm_format(path: str | Path) -> XThermFormat:
    """Load and validate the authoritative formal XTherm-format YAML file."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(
            f"XTherm format configuration not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)

    if not isinstance(data, dict):
        raise ValueError("XTherm format configuration must be a YAML mapping.")

    try:
        format_block = data["format"]
        image = data["image"]
        binary = data["binary"]
        validation = data["file_validation"]
        qc = data["temperature_qc"]
    except KeyError as exc:
        raise ValueError(
            f"Missing required XTherm configuration block: {exc.args[0]}"
        ) from exc

    if format_block.get("status") != "verified_for_formal_processing":
        raise ValueError(
            "XTherm format configuration is not approved for formal processing."
        )

    config = XThermFormat(
        width_px=int(image["width_px"]),
        height_px=int(image["height_px"]),
        header_bytes=int(binary["header_bytes"]),
        raw_dtype=str(binary["raw_dtype"]),
        byte_order=str(binary["byte_order"]),
        scale_C_per_count=float(binary["scale_C_per_count"]),
        offset_C=float(binary.get("offset_C", 0.0)),
        expected_file_size_bytes=int(
            validation["expected_file_size_bytes"]
        ),
        filename_extension=str(
            validation.get("filename_extension", ".xtherm")
        ),
        require_continuous_numeric_sequence=bool(
            validation.get("require_continuous_numeric_sequence", True)
        ),
        exclude_filenames=tuple(
            str(value) for value in validation.get("exclude_filenames", ())
        ),
        camera_valid_temperature_min_C=float(
            qc["camera_valid_temperature_min_C"]
        ),
        camera_valid_temperature_max_C=float(
            qc["camera_valid_temperature_max_C"]
        ),
        binary_qc_gross_upper_limit_C=float(
            qc["binary_qc_gross_upper_limit_C"]
        ),
        hard_saturation_threshold_C=float(
            qc["hard_saturation_threshold_C"]
        ),
        hard_saturation_value_C=float(
            qc["hard_saturation_value_C"]
        ),
        zero_ratio_note_threshold=float(
            qc.get("zero_ratio_note_threshold", 0.05)
        ),
    )
    _validate_xtherm_format(config)
    return config


def _validate_xtherm_format(config: XThermFormat) -> None:
    """Reject internally inconsistent or unsupported formal configurations."""
    if config.width_px <= 0 or config.height_px <= 0:
        raise ValueError("Image dimensions must be positive.")

    if config.header_bytes < 0:
        raise ValueError("header_bytes must be non-negative.")

    if config.raw_dtype != "uint16":
        raise ValueError(
            f"Unsupported formal raw dtype: {config.raw_dtype!r}"
        )

    if config.byte_order not in {"little", "big"}:
        raise ValueError(
            f"Unsupported byte order: {config.byte_order!r}"
        )

    if config.scale_C_per_count <= 0:
        raise ValueError("scale_C_per_count must be positive.")

    if (
        config.camera_valid_temperature_min_C
        >= config.camera_valid_temperature_max_C
    ):
        raise ValueError("Invalid camera quantitative measurement range.")

    if (
        config.binary_qc_gross_upper_limit_C
        <= config.camera_valid_temperature_max_C
    ):
        raise ValueError(
            "binary_qc_gross_upper_limit_C must exceed the camera-valid "
            "temperature maximum."
        )

    if (
        config.hard_saturation_threshold_C
        <= config.binary_qc_gross_upper_limit_C
    ):
        raise ValueError(
            "hard_saturation_threshold_C must exceed the gross QC limit."
        )

    if (
        config.hard_saturation_value_C
        < config.hard_saturation_threshold_C
    ):
        raise ValueError(
            "hard_saturation_value_C must be at or above the hard-saturation "
            "threshold."
        )

    if not (0.0 <= config.zero_ratio_note_threshold <= 1.0):
        raise ValueError(
            "zero_ratio_note_threshold must lie within [0, 1]."
        )

    # Formal dtype is uint16, so the payload uses two bytes per pixel.
    expected_payload = config.width_px * config.height_px * 2
    expected_total = config.header_bytes + expected_payload
    if config.expected_file_size_bytes != expected_total:
        raise ValueError(
            "expected_file_size_bytes does not match "
            "header_bytes + width_px * height_px * uint16_itemsize."
        )
