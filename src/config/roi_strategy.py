"""Validated loader for the formal ROI-strategy configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import math

import yaml

from config.physical_calibration import load_physical_calibration
from config.xtherm_format import load_xtherm_format


_ROOT = Path(__file__).resolve().parents[2]
_EXPECTED_STRATEGY = "global_roi_plus_tracking_window"
_EXPECTED_ACTIVATION = "evaluated_not_activated"
_EXPECTED_EVALUATION = "completed"
_EXPECTED_SLICE = "frames[1:]"
_FLOAT_TOL = 1e-9


class RoiStrategyError(ValueError):
    """Raised when the formal ROI-strategy configuration is inconsistent."""


@dataclass(frozen=True)
class RoiStrategyStatus:
    role: str
    strategy_name: str
    evaluation_status: str
    activation_status: str
    formal_roi_generation_enabled: bool
    formal_feature_extraction_enabled: bool


@dataclass(frozen=True)
class RoiStrategySources:
    evaluation_script: str
    evaluation_document: str
    evaluation_summary_local_artifact: str
    required_for_config_loading: bool
    xtherm_format_config: str
    physical_calibration_config: str


@dataclass(frozen=True)
class EffectiveFrameRule:
    startup_frames_excluded: int
    start_index_0_based: int
    python_slice: str


@dataclass(frozen=True)
class ThresholdsC:
    envelope: float
    core: float


@dataclass(frozen=True)
class CoordinateSystem:
    source_image_height_px: int
    source_image_width_px: int
    convention: str
    reference_frame: str


@dataclass(frozen=True)
class PixelSize:
    source: str
    pixel_size_x_mm: float
    pixel_size_y_mm: float
    isotropic_scaling_assumed: bool


@dataclass(frozen=True)
class CoverageResult:
    envelope_700_fraction: float
    core_800_fraction: float


@dataclass(frozen=True)
class FixedGlobalRoi:
    row_start: int
    row_stop: int
    col_start: int
    col_stop: int
    height_px: int
    width_px: int
    coverage: CoverageResult
    evaluation_accepted: bool
    formal_use_enabled: bool
    purpose: tuple[str, ...]


@dataclass(frozen=True)
class TrackingWindow:
    method: str
    width_px: int
    height_px: int
    width_mm: float
    height_mm: float
    coverage: CoverageResult
    clipped_frame_count: int
    edge_adjusted_frame_count: int
    evaluation_accepted: bool
    formal_use_enabled: bool
    purpose: tuple[str, ...]


@dataclass(frozen=True)
class LegacyFixedRoiCandidate:
    top: int
    left: int
    height: int
    width: int
    minimum_envelope_700_coverage_fraction: float
    evaluation_accepted: bool
    formal_use_enabled: bool
    rejection_reason: str


@dataclass(frozen=True)
class LegacyTrackingWindowCandidate:
    width_px: int
    height_px: int
    evaluation_accepted: bool
    formal_use_enabled: bool
    rejection_reason: str


@dataclass(frozen=True)
class LegacyCandidates:
    fixed_roi: LegacyFixedRoiCandidate
    tracking_window: LegacyTrackingWindowCandidate


@dataclass(frozen=True)
class RoiStrategyConfig:
    schema_version: int
    status: RoiStrategyStatus
    sources: RoiStrategySources
    effective_frames: EffectiveFrameRule
    thresholds_C: ThresholdsC
    coordinate_system: CoordinateSystem
    pixel_size: PixelSize
    fixed_global_roi: FixedGlobalRoi
    tracking_window: TrackingWindow
    legacy_candidates: LegacyCandidates


def load_roi_strategy(
    path: str | Path = "configs/roi_strategy.yaml",
) -> RoiStrategyConfig:
    """Load and validate the formal ROI-strategy YAML file.

    The loader is deliberately a pure configuration reader: it does not import
    scripts, read data/results, run ROI evaluation, or write files.
    """
    config_path = _resolve_repo_path(path)
    if not config_path.is_file():
        raise RoiStrategyError(f"ROI strategy configuration not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)

    if not isinstance(data, dict):
        raise RoiStrategyError("ROI strategy configuration must be a YAML mapping.")

    config = _parse_config(data)
    _validate(config)
    _cross_validate_authorities(config)
    return config


def _resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else _ROOT / candidate


def _section(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise RoiStrategyError(f"missing or invalid '{key}' section")
    return value


def _required(mapping: Mapping[str, Any], key: str, section: str) -> Any:
    if key not in mapping:
        raise RoiStrategyError(f"missing '{section}.{key}'")
    return mapping[key]


def _required_int(mapping: Mapping[str, Any], key: str, section: str) -> int:
    value = _required(mapping, key, section)
    if type(value) is not int:
        raise RoiStrategyError(
            f"{section}.{key} must be an integer, got {type(value).__name__}"
        )
    return value


def _required_float(mapping: Mapping[str, Any], key: str, section: str) -> float:
    value = _required(mapping, key, section)
    if isinstance(value, bool) or type(value) not in (int, float):
        raise RoiStrategyError(
            f"{section}.{key} must be a finite number, got {type(value).__name__}"
        )

    result = float(value)
    if not math.isfinite(result):
        raise RoiStrategyError(f"{section}.{key} must be finite, got {result!r}")
    return result


def _required_bool(mapping: Mapping[str, Any], key: str, section: str) -> bool:
    value = _required(mapping, key, section)
    if type(value) is not bool:
        raise RoiStrategyError(f"'{section}.{key}' must be a YAML boolean")
    return value


def _parse_config(data: dict) -> RoiStrategyConfig:
    status = _section(data, "status")
    sources = _section(data, "sources")
    frames = _section(data, "effective_frames")
    thresholds = _section(data, "thresholds_C")
    coords = _section(data, "coordinate_system")
    pixel = _section(data, "pixel_size")
    fixed = _section(data, "fixed_global_roi")
    tracking = _section(data, "tracking_window")
    legacy = _section(data, "legacy_candidates")

    return RoiStrategyConfig(
        schema_version=_required_int(data, "schema_version", "root"),
        status=RoiStrategyStatus(
            role=str(_required(status, "role", "status")),
            strategy_name=str(_required(status, "strategy_name", "status")),
            evaluation_status=str(_required(status, "evaluation_status", "status")),
            activation_status=str(_required(status, "activation_status", "status")),
            formal_roi_generation_enabled=_required_bool(
                status, "formal_roi_generation_enabled", "status"
            ),
            formal_feature_extraction_enabled=_required_bool(
                status, "formal_feature_extraction_enabled", "status"
            ),
        ),
        sources=RoiStrategySources(
            evaluation_script=str(
                _required(sources, "evaluation_script", "sources")
            ),
            evaluation_document=str(
                _required(sources, "evaluation_document", "sources")
            ),
            evaluation_summary_local_artifact=str(
                _required(sources, "evaluation_summary_local_artifact", "sources")
            ),
            required_for_config_loading=_required_bool(
                sources, "required_for_config_loading", "sources"
            ),
            xtherm_format_config=str(
                _required(sources, "xtherm_format_config", "sources")
            ),
            physical_calibration_config=str(
                _required(sources, "physical_calibration_config", "sources")
            ),
        ),
        effective_frames=EffectiveFrameRule(
            startup_frames_excluded=_required_int(
                frames, "startup_frames_excluded", "effective_frames"
            ),
            start_index_0_based=_required_int(
                frames, "start_index_0_based", "effective_frames"
            ),
            python_slice=str(_required(frames, "python_slice", "effective_frames")),
        ),
        thresholds_C=ThresholdsC(
            envelope=_required_float(thresholds, "envelope", "thresholds_C"),
            core=_required_float(thresholds, "core", "thresholds_C"),
        ),
        coordinate_system=CoordinateSystem(
            source_image_height_px=_required_int(
                coords, "source_image_height_px", "coordinate_system"
            ),
            source_image_width_px=_required_int(
                coords, "source_image_width_px", "coordinate_system"
            ),
            convention=str(_required(coords, "convention", "coordinate_system")),
            reference_frame=str(
                _required(coords, "reference_frame", "coordinate_system")
            ),
        ),
        pixel_size=PixelSize(
            source=str(_required(pixel, "source", "pixel_size")),
            pixel_size_x_mm=_required_float(pixel, "pixel_size_x_mm", "pixel_size"),
            pixel_size_y_mm=_required_float(pixel, "pixel_size_y_mm", "pixel_size"),
            isotropic_scaling_assumed=_required_bool(
                pixel, "isotropic_scaling_assumed", "pixel_size"
            ),
        ),
        fixed_global_roi=_parse_fixed_roi(fixed),
        tracking_window=_parse_tracking_window(tracking),
        legacy_candidates=_parse_legacy_candidates(legacy),
    )


def _parse_coverage(mapping: dict, section: str) -> CoverageResult:
    coverage = _section(mapping, "coverage")
    return CoverageResult(
        envelope_700_fraction=_required_float(
            coverage, "envelope_700_fraction", f"{section}.coverage"
        ),
        core_800_fraction=_required_float(
            coverage, "core_800_fraction", f"{section}.coverage"
        ),
    )


def _parse_fixed_roi(mapping: dict) -> FixedGlobalRoi:
    return FixedGlobalRoi(
        row_start=_required_int(mapping, "row_start", "fixed_global_roi"),
        row_stop=_required_int(mapping, "row_stop", "fixed_global_roi"),
        col_start=_required_int(mapping, "col_start", "fixed_global_roi"),
        col_stop=_required_int(mapping, "col_stop", "fixed_global_roi"),
        height_px=_required_int(mapping, "height_px", "fixed_global_roi"),
        width_px=_required_int(mapping, "width_px", "fixed_global_roi"),
        coverage=_parse_coverage(mapping, "fixed_global_roi"),
        evaluation_accepted=_required_bool(
            mapping, "evaluation_accepted", "fixed_global_roi"
        ),
        formal_use_enabled=_required_bool(
            mapping, "formal_use_enabled", "fixed_global_roi"
        ),
        purpose=tuple(str(item) for item in mapping.get("purpose", ())),
    )


def _parse_tracking_window(mapping: dict) -> TrackingWindow:
    return TrackingWindow(
        method=str(_required(mapping, "method", "tracking_window")),
        width_px=_required_int(mapping, "width_px", "tracking_window"),
        height_px=_required_int(mapping, "height_px", "tracking_window"),
        width_mm=_required_float(mapping, "width_mm", "tracking_window"),
        height_mm=_required_float(mapping, "height_mm", "tracking_window"),
        coverage=_parse_coverage(mapping, "tracking_window"),
        clipped_frame_count=int(
            _required(mapping, "clipped_frame_count", "tracking_window")
        ),
        edge_adjusted_frame_count=int(
            _required(mapping, "edge_adjusted_frame_count", "tracking_window")
        ),
        evaluation_accepted=_required_bool(
            mapping, "evaluation_accepted", "tracking_window"
        ),
        formal_use_enabled=_required_bool(
            mapping, "formal_use_enabled", "tracking_window"
        ),
        purpose=tuple(str(item) for item in mapping.get("purpose", ())),
    )


def _parse_legacy_candidates(mapping: dict) -> LegacyCandidates:
    fixed = _section(mapping, "fixed_roi")
    tracking = _section(mapping, "tracking_window")
    return LegacyCandidates(
        fixed_roi=LegacyFixedRoiCandidate(
            top=_required_int(fixed, "top", "legacy_candidates.fixed_roi"),
            left=_required_int(fixed, "left", "legacy_candidates.fixed_roi"),
            height=_required_int(fixed, "height", "legacy_candidates.fixed_roi"),
            width=_required_int(fixed, "width", "legacy_candidates.fixed_roi"),
            minimum_envelope_700_coverage_fraction=_required_float(
                fixed,
                "minimum_envelope_700_coverage_fraction",
                "legacy_candidates.fixed_roi",
            ),
            evaluation_accepted=_required_bool(
                fixed,
                "evaluation_accepted",
                "legacy_candidates.fixed_roi",
            ),
            formal_use_enabled=_required_bool(
                fixed, "formal_use_enabled", "legacy_candidates.fixed_roi"
            ),
            rejection_reason=str(
                _required(fixed, "rejection_reason", "legacy_candidates.fixed_roi")
            ),
        ),
        tracking_window=LegacyTrackingWindowCandidate(
            width_px=_required_int(
                tracking,
                "width_px",
                "legacy_candidates.tracking_window",
            ),
            height_px=_required_int(
                tracking,
                "height_px",
                "legacy_candidates.tracking_window",
            ),
            evaluation_accepted=_required_bool(
                tracking,
                "evaluation_accepted",
                "legacy_candidates.tracking_window",
            ),
            formal_use_enabled=_required_bool(
                tracking,
                "formal_use_enabled",
                "legacy_candidates.tracking_window",
            ),
            rejection_reason=str(
                _required(
                    tracking,
                    "rejection_reason",
                    "legacy_candidates.tracking_window",
                )
            ),
        ),
    )


def _validate(config: RoiStrategyConfig) -> None:
    if config.schema_version != 1:
        raise RoiStrategyError(f"schema_version must be 1, got {config.schema_version}")

    status = config.status
    if status.role != "formal_roi_strategy":
        raise RoiStrategyError(f"unexpected ROI strategy role: {status.role!r}")
    if status.strategy_name != _EXPECTED_STRATEGY:
        raise RoiStrategyError(
            f"strategy_name must be {_EXPECTED_STRATEGY!r}, got {status.strategy_name!r}"
        )
    if status.evaluation_status != _EXPECTED_EVALUATION:
        raise RoiStrategyError(
            "evaluation_status must be "
            f"{_EXPECTED_EVALUATION!r}, got {status.evaluation_status!r}"
        )
    if status.activation_status != _EXPECTED_ACTIVATION:
        raise RoiStrategyError(
            "activation_status must be "
            f"{_EXPECTED_ACTIVATION!r}, got {status.activation_status!r}"
        )
    if status.formal_roi_generation_enabled is not False:
        raise RoiStrategyError("formal_roi_generation_enabled must remain false")
    if status.formal_feature_extraction_enabled is not False:
        raise RoiStrategyError("formal_feature_extraction_enabled must remain false")
    if config.sources.required_for_config_loading is not False:
        raise RoiStrategyError("local ROI evaluation artifacts must not be required")

    _validate_effective_frames(config.effective_frames)
    _validate_coordinate_system(config.coordinate_system)
    _validate_pixel_size(config.pixel_size)
    _validate_fixed_roi(config.fixed_global_roi, config.coordinate_system)
    _validate_tracking_window(
        config.tracking_window,
        config.coordinate_system,
        config.pixel_size,
    )
    _validate_legacy_candidates(config.legacy_candidates)


def _validate_effective_frames(rule: EffectiveFrameRule) -> None:
    if rule.startup_frames_excluded != 1:
        raise RoiStrategyError("startup_frames_excluded must be 1")
    if rule.start_index_0_based != 1:
        raise RoiStrategyError("start_index_0_based must be 1")
    if rule.python_slice != _EXPECTED_SLICE:
        raise RoiStrategyError(f"python_slice must be {_EXPECTED_SLICE!r}")


def _validate_coordinate_system(coords: CoordinateSystem) -> None:
    if coords.source_image_height_px <= 0 or coords.source_image_width_px <= 0:
        raise RoiStrategyError("source image dimensions must be positive")
    if coords.convention != "half_open_python":
        raise RoiStrategyError("coordinate convention must be half_open_python")
    if coords.reference_frame != "full_frame_array":
        raise RoiStrategyError("reference_frame must be full_frame_array")


def _validate_pixel_size(pixel_size: PixelSize) -> None:
    if pixel_size.pixel_size_x_mm <= 0 or pixel_size.pixel_size_y_mm <= 0:
        raise RoiStrategyError("pixel sizes must be positive")
    if pixel_size.isotropic_scaling_assumed is not True:
        raise RoiStrategyError("formal ROI config expects isotropic scaling")


def _validate_fixed_roi(roi: FixedGlobalRoi, coords: CoordinateSystem) -> None:
    if not (roi.row_stop > roi.row_start):
        raise RoiStrategyError("fixed_global_roi row_stop must exceed row_start")
    if not (roi.col_stop > roi.col_start):
        raise RoiStrategyError("fixed_global_roi col_stop must exceed col_start")
    if roi.height_px != roi.row_stop - roi.row_start:
        raise RoiStrategyError("fixed_global_roi height_px does not match rows")
    if roi.width_px != roi.col_stop - roi.col_start:
        raise RoiStrategyError("fixed_global_roi width_px does not match cols")
    if roi.row_start < 0 or roi.row_stop > coords.source_image_height_px:
        raise RoiStrategyError("fixed_global_roi rows fall outside source image")
    if roi.col_start < 0 or roi.col_stop > coords.source_image_width_px:
        raise RoiStrategyError("fixed_global_roi cols fall outside source image")
    _validate_coverage(roi.coverage, "fixed_global_roi")
    if roi.evaluation_accepted is not True:
        raise RoiStrategyError("current fixed global ROI must be evaluation_accepted")
    if roi.formal_use_enabled is not False:
        raise RoiStrategyError("current fixed global ROI formal_use_enabled must be false")


def _validate_tracking_window(
    window: TrackingWindow,
    coords: CoordinateSystem,
    pixel_size: PixelSize,
) -> None:
    if window.width_px <= 0 or window.height_px <= 0:
        raise RoiStrategyError("tracking window dimensions must be positive")
    if window.width_px > coords.source_image_width_px:
        raise RoiStrategyError("tracking window width exceeds source image width")
    if window.height_px > coords.source_image_height_px:
        raise RoiStrategyError("tracking window height exceeds source image height")
    _validate_coverage(window.coverage, "tracking_window")
    if window.clipped_frame_count < 0:
        raise RoiStrategyError("clipped_frame_count must be non-negative")
    if window.edge_adjusted_frame_count < 0:
        raise RoiStrategyError("edge_adjusted_frame_count must be non-negative")
    if window.evaluation_accepted is not True:
        raise RoiStrategyError("current tracking window must be evaluation_accepted")
    if window.formal_use_enabled is not False:
        raise RoiStrategyError("current tracking window formal_use_enabled must be false")

    expected_width_mm = window.width_px * pixel_size.pixel_size_x_mm
    expected_height_mm = window.height_px * pixel_size.pixel_size_y_mm
    if not math.isclose(
        window.width_mm, expected_width_mm, rel_tol=_FLOAT_TOL, abs_tol=_FLOAT_TOL
    ):
        raise RoiStrategyError("tracking window width_mm is inconsistent with pixel size")
    if not math.isclose(
        window.height_mm, expected_height_mm, rel_tol=_FLOAT_TOL, abs_tol=_FLOAT_TOL
    ):
        raise RoiStrategyError("tracking window height_mm is inconsistent with pixel size")


def _validate_coverage(coverage: CoverageResult, section: str) -> None:
    for name, value in (
        ("envelope_700_fraction", coverage.envelope_700_fraction),
        ("core_800_fraction", coverage.core_800_fraction),
    ):
        if not (0.0 <= value <= 1.0):
            raise RoiStrategyError(f"{section}.{name} must lie within [0, 1]")


def _validate_legacy_candidates(legacy: LegacyCandidates) -> None:
    if not (0.0 <= legacy.fixed_roi.minimum_envelope_700_coverage_fraction <= 1.0):
        raise RoiStrategyError("legacy fixed ROI coverage must lie within [0, 1]")
    if legacy.fixed_roi.evaluation_accepted is not False:
        raise RoiStrategyError("legacy fixed ROI must not be evaluation_accepted")
    if legacy.tracking_window.evaluation_accepted is not False:
        raise RoiStrategyError("legacy tracking window must not be evaluation_accepted")
    if legacy.fixed_roi.formal_use_enabled is not False:
        raise RoiStrategyError("legacy fixed ROI formal_use_enabled must be false")
    if legacy.tracking_window.formal_use_enabled is not False:
        raise RoiStrategyError("legacy tracking window formal_use_enabled must be false")
    if legacy.fixed_roi.height <= 0 or legacy.fixed_roi.width <= 0:
        raise RoiStrategyError("legacy fixed ROI dimensions must be positive")
    if legacy.tracking_window.width_px <= 0 or legacy.tracking_window.height_px <= 0:
        raise RoiStrategyError("legacy tracking window dimensions must be positive")


def _cross_validate_authorities(config: RoiStrategyConfig) -> None:
    xtherm = load_xtherm_format(_resolve_repo_path(config.sources.xtherm_format_config))
    if xtherm.width_px != config.coordinate_system.source_image_width_px:
        raise RoiStrategyError("source_image_width_px differs from xtherm format")
    if xtherm.height_px != config.coordinate_system.source_image_height_px:
        raise RoiStrategyError("source_image_height_px differs from xtherm format")

    calibration = load_physical_calibration(
        str(_resolve_repo_path(config.sources.physical_calibration_config))
    )
    if not math.isclose(
        calibration.pixel_size_x_mm,
        config.pixel_size.pixel_size_x_mm,
        rel_tol=_FLOAT_TOL,
        abs_tol=_FLOAT_TOL,
    ):
        raise RoiStrategyError("pixel_size_x_mm differs from physical calibration")
    if not math.isclose(
        calibration.pixel_size_y_mm,
        config.pixel_size.pixel_size_y_mm,
        rel_tol=_FLOAT_TOL,
        abs_tol=_FLOAT_TOL,
    ):
        raise RoiStrategyError("pixel_size_y_mm differs from physical calibration")
    if calibration.isotropic != config.pixel_size.isotropic_scaling_assumed:
        raise RoiStrategyError("isotropic flag differs from physical calibration")
