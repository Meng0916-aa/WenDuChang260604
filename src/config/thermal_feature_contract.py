"""Validated loader for the formal thermal-feature contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from types import MappingProxyType
import math
import re

import yaml

from config.xtherm_format import load_xtherm_format


_ROOT = Path(__file__).resolve().parents[2]
_VALID_ROLES = {"core", "qc_only", "secondary", "rejected"}
_EXPECTED_CORE_COUNT = 15
_EXPECTED_CORE_NAMES = frozenset(
    {
        "mean_active_frame_valid_temperature_C",
        "max_frame_p999_valid_temperature_C",
        "mean_main_area_above_700_C_mm2",
        "mean_main_area_above_800_C_mm2",
        "mean_main_transverse_width_above_700_C_mm",
        "mean_main_scan_length_above_700_C_mm",
        "centroid_path_length_mm",
        "signed_scan_direction_displacement_mm",
        "signed_transverse_drift_mm",
        "centroid_transverse_jitter_mm",
        "median_frame_p95_internal_gradient_magnitude_700_C_per_mm",
        "mean_signed_thermal_centroid_offset_from_geometric_center_mm",
        "mean_left_right_excess_temperature_asymmetry_700_fraction",
        "hot_core_presence_duration_800_C_s",
        "main_area_above_700_C_temporal_cv",
    }
)
_EXPECTED_TRACK_ROWS = 57
_EXPECTED_CONDITION_ROWS = 19
_XTHERM_FORMAT_SOURCE = "configs/xtherm_format.yaml"
_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_UNIT_SUFFIXES = {
    "count": "_count",
    "fraction": "_fraction",
    "C": "_C",
    "mm2": "_mm2",
    "mm": "_mm",
    "s": "_s",
    "C_per_mm": "_C_per_mm",
    "C_per_s": "_C_per_s",
    "mm_per_s": "_mm_per_s",
    "cv": "_cv",
}


class ThermalFeatureContractError(ValueError):
    """Raised when the formal feature contract is missing or inconsistent."""


@dataclass(frozen=True)
class ContractStatus:
    role: str
    design_status: str
    execution_status: str
    formal_feature_extraction_enabled: bool
    formal_roi_generation_enabled: bool


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    role: str
    physical_meaning: str
    unit: str
    region: str
    frame_population: str
    frame_calculation: str
    track_aggregation: str
    condition_aggregations: tuple[str, ...]
    missing_policy: str
    cv_applicable: bool
    default_model_input: bool
    default_rsm_response: bool
    signed: bool = False
    enabled: bool = True
    rejection_reason: str | None = None


@dataclass(frozen=True)
class OutputTableContract:
    planned_path: str
    expected_rows: int
    row_unit: str
    creation_enabled: bool


@dataclass(frozen=True)
class ThermalFeatureContract:
    schema_version: int
    status: ContractStatus
    sources: Mapping[str, str]
    frame_definitions: Mapping[str, Any]
    temperature_validity: Mapping[str, Any]
    geometry_mask_policy: Mapping[str, Any]
    coordinate_convention: Mapping[str, Any]
    feature_naming_rules: Mapping[str, Any]
    core_features: tuple[FeatureDefinition, ...]
    qc_only_features: tuple[FeatureDefinition, ...]
    secondary_features: tuple[FeatureDefinition, ...]
    rejected_features: tuple[FeatureDefinition, ...]
    track_level_output_contract: OutputTableContract
    condition_level_output_contract: OutputTableContract
    aggregation_policy: Mapping[str, Any]
    forbidden_operations: tuple[str, ...]

    @property
    def all_features(self) -> tuple[FeatureDefinition, ...]:
        return (
            self.core_features
            + self.qc_only_features
            + self.secondary_features
            + self.rejected_features
        )


def load_thermal_feature_contract(
    path: str | Path = "configs/thermal_feature_contract.yaml",
) -> ThermalFeatureContract:
    """Load and validate the formal thermal-feature contract.

    This is a pure configuration reader: it does not import feature extraction
    code, does not import numpy, and never reads data/results.
    """
    config_path = _resolve_repo_path(path)
    if not config_path.is_file():
        raise ThermalFeatureContractError(
            f"thermal feature contract not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)

    if not isinstance(data, dict):
        raise ThermalFeatureContractError(
            "thermal feature contract must be a YAML mapping"
        )

    contract = _parse_contract(data)
    _validate_contract(contract)
    return contract


def _resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else _ROOT / candidate


def _section(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key)
    if not isinstance(value, Mapping):
        raise ThermalFeatureContractError(f"missing or invalid '{key}' section")
    return value


def _required(mapping: Mapping[str, Any], key: str, section: str) -> Any:
    if key not in mapping:
        raise ThermalFeatureContractError(f"missing '{section}.{key}'")
    return mapping[key]


def _required_int(mapping: Mapping[str, Any], key: str, section: str) -> int:
    value = _required(mapping, key, section)
    if type(value) is not int:
        raise ThermalFeatureContractError(
            f"{section}.{key} must be an integer, got {type(value).__name__}"
        )
    return value


def _required_float(mapping: Mapping[str, Any], key: str, section: str) -> float:
    value = _required(mapping, key, section)
    if isinstance(value, bool) or type(value) not in (int, float):
        raise ThermalFeatureContractError(
            f"{section}.{key} must be a finite number, got {type(value).__name__}"
        )
    result = float(value)
    if not math.isfinite(result):
        raise ThermalFeatureContractError(
            f"{section}.{key} must be finite, got {result!r}"
        )
    return result


def _required_bool(mapping: Mapping[str, Any], key: str, section: str) -> bool:
    value = _required(mapping, key, section)
    if type(value) is not bool:
        raise ThermalFeatureContractError(
            f"{section}.{key} must be a YAML boolean"
        )
    return value


def _parse_contract(data: Mapping[str, Any]) -> ThermalFeatureContract:
    status = _section(data, "status")
    sources = _section(data, "sources")
    return ThermalFeatureContract(
        schema_version=_required_int(data, "schema_version", "root"),
        status=ContractStatus(
            role=str(_required(status, "role", "status")),
            design_status=str(_required(status, "design_status", "status")),
            execution_status=str(_required(status, "execution_status", "status")),
            formal_feature_extraction_enabled=_required_bool(
                status, "formal_feature_extraction_enabled", "status"
            ),
            formal_roi_generation_enabled=_required_bool(
                status, "formal_roi_generation_enabled", "status"
            ),
        ),
        sources=MappingProxyType({str(k): str(v) for k, v in sources.items()}),
        frame_definitions=_freeze_mapping(_section(data, "frame_definitions")),
        temperature_validity=_freeze_mapping(_section(data, "temperature_validity")),
        geometry_mask_policy=_freeze_mapping(_section(data, "geometry_mask_policy")),
        coordinate_convention=_freeze_mapping(_section(data, "coordinate_convention")),
        feature_naming_rules=_freeze_mapping(_section(data, "feature_naming_rules")),
        core_features=_parse_features(data, "core_features", "core"),
        qc_only_features=_parse_features(data, "qc_only_features", "qc_only"),
        secondary_features=_parse_features(data, "secondary_features", "secondary"),
        rejected_features=_parse_features(data, "rejected_features", "rejected"),
        track_level_output_contract=_parse_output_contract(
            _section(data, "track_level_output_contract"),
            "track_level_output_contract",
        ),
        condition_level_output_contract=_parse_output_contract(
            _section(data, "condition_level_output_contract"),
            "condition_level_output_contract",
        ),
        aggregation_policy=_freeze_mapping(_section(data, "aggregation_policy")),
        forbidden_operations=_required_str_tuple(data, "forbidden_operations", "root"),
    )


def _parse_features(
    data: Mapping[str, Any],
    section: str,
    expected_role: str,
) -> tuple[FeatureDefinition, ...]:
    raw = _required(data, section, "root")
    if not isinstance(raw, list):
        raise ThermalFeatureContractError(f"{section} must be a list")

    features = []
    for index, item in enumerate(raw):
        item_path = f"{section}[{index}]"
        if not isinstance(item, dict):
            raise ThermalFeatureContractError(f"{item_path} must be a mapping")
        role = str(_required(item, "role", item_path))
        if role != expected_role:
            raise ThermalFeatureContractError(
                f"{item_path}.role must be {expected_role!r}, got {role!r}"
            )
        enabled = item.get("enabled", role != "rejected")
        if type(enabled) is not bool:
            raise ThermalFeatureContractError(f"{item_path}.enabled must be boolean")
        features.append(
            FeatureDefinition(
                name=str(_required(item, "name", item_path)),
                role=role,
                physical_meaning=str(
                    _required(item, "physical_meaning", item_path)
                ),
                unit=str(_required(item, "unit", item_path)),
                region=str(_required(item, "region", item_path)),
                frame_population=str(
                    _required(item, "frame_population", item_path)
                ),
                frame_calculation=str(
                    _required(item, "frame_calculation", item_path)
                ),
                track_aggregation=str(
                    _required(item, "track_aggregation", item_path)
                ),
                condition_aggregations=_required_str_tuple(
                    item, "condition_aggregations", item_path
                ),
                missing_policy=str(_required(item, "missing_policy", item_path)),
                cv_applicable=_required_bool(item, "cv_applicable", item_path),
                default_model_input=_required_bool(
                    item, "default_model_input", item_path
                ),
                default_rsm_response=_required_bool(
                    item, "default_rsm_response", item_path
                ),
                signed=_optional_bool(item, "signed", False, item_path),
                enabled=enabled,
                rejection_reason=(
                    str(item["rejection_reason"])
                    if "rejection_reason" in item
                    else None
                ),
            )
        )
    return tuple(features)


def _optional_bool(
    mapping: Mapping[str, Any],
    key: str,
    default: bool,
    section: str,
) -> bool:
    if key not in mapping:
        return default
    value = mapping[key]
    if type(value) is not bool:
        raise ThermalFeatureContractError(f"{section}.{key} must be boolean")
    return value


def _required_str_tuple(
    mapping: Mapping[str, Any],
    key: str,
    section: str,
) -> tuple[str, ...]:
    value = _required(mapping, key, section)
    if not isinstance(value, (list, tuple)):
        raise ThermalFeatureContractError(f"{section}.{key} must be a list")
    return tuple(str(item) for item in value)


def _assert_equal(field: str, contract_value: Any, authority_value: Any) -> None:
    if _normalize_scalar(contract_value) != _normalize_scalar(authority_value):
        raise ThermalFeatureContractError(
            f"{field} conflicts with {_XTHERM_FORMAT_SOURCE}: "
            f"thermal_feature_contract.yaml={contract_value!r}, "
            f"xtherm_format.yaml={authority_value!r}"
        )


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 12)
    if isinstance(value, int):
        return value
    try:
        return round(float(value), 12)
    except (TypeError, ValueError):
        return str(value)


def _parse_output_contract(
    data: Mapping[str, Any],
    section: str,
) -> OutputTableContract:
    return OutputTableContract(
        planned_path=str(_required(data, "planned_path", section)),
        expected_rows=_required_int(data, "expected_rows", section),
        row_unit=str(_required(data, "row_unit", section)),
        creation_enabled=_required_bool(data, "creation_enabled", section),
    )


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    frozen = {}
    for key, item in value.items():
        if isinstance(item, dict):
            frozen[str(key)] = _freeze_mapping(item)
        elif isinstance(item, list):
            frozen[str(key)] = tuple(item)
        else:
            frozen[str(key)] = item
    return MappingProxyType(frozen)


def _validate_contract(contract: ThermalFeatureContract) -> None:
    if contract.schema_version != 1:
        raise ThermalFeatureContractError("schema_version must be 1")

    status = contract.status
    if status.role != "formal_thermal_feature_contract":
        raise ThermalFeatureContractError("unexpected contract role")
    if status.design_status != "approved":
        raise ThermalFeatureContractError("design_status must be approved")
    if status.execution_status != "designed_not_executed":
        raise ThermalFeatureContractError(
            "execution_status must be designed_not_executed"
        )
    if status.formal_feature_extraction_enabled is not False:
        raise ThermalFeatureContractError(
            "formal_feature_extraction_enabled must remain false"
        )
    if status.formal_roi_generation_enabled is not False:
        raise ThermalFeatureContractError(
            "formal_roi_generation_enabled must remain false"
        )

    core_name_list = [feature.name for feature in contract.core_features]
    core_names = set(core_name_list)
    if len(contract.core_features) != _EXPECTED_CORE_COUNT:
        raise ThermalFeatureContractError(
            f"core feature count must be {_EXPECTED_CORE_COUNT}"
        )
    if len(core_name_list) != len(core_names):
        raise ThermalFeatureContractError("core feature names must be unique")
    if core_names != _EXPECTED_CORE_NAMES:
        missing = sorted(_EXPECTED_CORE_NAMES - core_names)
        extra = sorted(core_names - _EXPECTED_CORE_NAMES)
        raise ThermalFeatureContractError(
            "core feature names must match the approved set; "
            f"missing={missing}, extra={extra}"
        )

    _validate_sources(contract.sources)
    _validate_frame_and_temperature_blocks(contract)
    _validate_features(contract.all_features)
    _validate_output_contracts(contract)


def _validate_sources(sources: Mapping[str, str]) -> None:
    for key, value in sources.items():
        if value == "configs/default.yaml":
            raise ThermalFeatureContractError(
                f"{key} must not reference configs/default.yaml"
            )
    required = {
        "formal_pipeline_config",
        "roi_strategy_config",
        "physical_calibration_config",
        "xtherm_format_config",
        "feature_dictionary_document",
    }
    missing = sorted(required - set(sources))
    if missing:
        raise ThermalFeatureContractError(f"missing sources: {missing}")
    if sources["xtherm_format_config"] != _XTHERM_FORMAT_SOURCE:
        raise ThermalFeatureContractError(
            "xtherm_format_config must be configs/xtherm_format.yaml"
        )


def _validate_frame_and_temperature_blocks(contract: ThermalFeatureContract) -> None:
    frames = contract.frame_definitions
    effective = frames.get("effective_frame", {})
    if effective.get("python_slice") != "frames[1:]":
        raise ThermalFeatureContractError("effective frame slice must be frames[1:]")
    if (
        _required_int(
            effective,
            "startup_frames_excluded",
            "frame_definitions.effective_frame",
        )
        != 1
    ):
        raise ThermalFeatureContractError("startup_frames_excluded must be 1")
    rules = frames.get("aggregation_rules", {})
    if _required_bool(
        rules,
        "extra_start_or_end_frame_exclusion_enabled",
        "frame_definitions.aggregation_rules",
    ) is not False:
        raise ThermalFeatureContractError("extra frame exclusion must remain false")

    temp = contract.temperature_validity
    valid_min = _required_float(temp, "valid_min_C", "temperature_validity")
    valid_max = _required_float(temp, "valid_max_C", "temperature_validity")
    hard = _required_float(
        temp, "hard_saturation_threshold_C", "temperature_validity"
    )
    above_min = _required_float(
        temp, "above_range_min_C", "temperature_validity"
    )
    above_max = _required_float(
        temp, "above_range_max_C", "temperature_validity"
    )
    xtherm = load_xtherm_format(
        _resolve_repo_path(contract.sources["xtherm_format_config"])
    )
    _assert_equal(
        "temperature_validity.valid_min_C",
        valid_min,
        xtherm.camera_valid_temperature_min_C,
    )
    _assert_equal(
        "temperature_validity.valid_max_C",
        valid_max,
        xtherm.camera_valid_temperature_max_C,
    )
    _assert_equal(
        "temperature_validity.hard_saturation_threshold_C",
        hard,
        xtherm.hard_saturation_threshold_C,
    )
    if not (valid_min < valid_max < hard):
        raise ThermalFeatureContractError(
            "temperature validity must satisfy valid_min_C < valid_max_C "
            "< hard_saturation_threshold_C"
        )
    _assert_equal("temperature_validity.above_range_min_C", above_min, valid_max)
    _assert_equal("temperature_validity.above_range_max_C", above_max, hard)
    if _required_bool(
        temp,
        "invalid_pixel_interpolation_enabled",
        "temperature_validity",
    ) is not False:
        raise ThermalFeatureContractError("invalid interpolation must remain false")
    if _required_bool(
        temp,
        "above_range_quantitative_use_enabled",
        "temperature_validity",
    ) is not False:
        raise ThermalFeatureContractError("above_range quantitative use is forbidden")
    if _required_bool(
        temp,
        "hard_saturation_quantitative_use_enabled",
        "temperature_validity",
    ) is not False:
        raise ThermalFeatureContractError(
            "hard_saturation quantitative use is forbidden"
        )

    geometry = contract.geometry_mask_policy
    thresholds = _section(geometry, "main_region_thresholds_C")
    envelope = _required_float(
        thresholds, "envelope", "geometry_mask_policy.main_region_thresholds_C"
    )
    core = _required_float(
        thresholds, "core", "geometry_mask_policy.main_region_thresholds_C"
    )
    if not (valid_min <= envelope < core <= valid_max):
        raise ThermalFeatureContractError(
            "formal 700/800 C thresholds must be ordered and within the "
            "camera-valid range"
        )
    if envelope != 700.0 or core != 800.0:
        raise ThermalFeatureContractError("formal thresholds must be 700 C and 800 C")
    if _required_int(geometry, "connectivity", "geometry_mask_policy") != 8:
        raise ThermalFeatureContractError("geometry_mask_policy.connectivity must be 8")
    min_area = _required_int(
        geometry, "min_component_area_px", "geometry_mask_policy"
    )
    if min_area != 9:
        raise ThermalFeatureContractError(
            "geometry_mask_policy.min_component_area_px must be 9"
        )
    if _required_bool(
        geometry, "valid_hot_seed_required", "geometry_mask_policy"
    ) is not True:
        raise ThermalFeatureContractError("valid hot seed is required")
    candidates = set(
        _required_str_tuple(
            geometry, "expansion_candidates", "geometry_mask_policy"
        )
    )
    if candidates != {"above_range", "hard_saturation"}:
        raise ThermalFeatureContractError(
            "expansion_candidates must be above_range and hard_saturation"
        )
    if _required_bool(
        geometry,
        "expansion_requires_connection_to_valid_hot_seed",
        "geometry_mask_policy",
    ) is not True:
        raise ThermalFeatureContractError(
            "expansion must require connection to a valid hot seed"
        )
    if str(_required(
        geometry, "main_component_selection", "geometry_mask_policy"
    )) != "largest_seeded_connected_component":
        raise ThermalFeatureContractError(
            "main_component_selection must be largest_seeded_connected_component"
        )
    if _required_bool(
        geometry, "fill_internal_holes", "geometry_mask_policy"
    ) is not True:
        raise ThermalFeatureContractError("fill_internal_holes must be true")
    if _required_bool(
        geometry,
        "include_above_or_hard_only_if_connected_to_valid_hot_region",
        "geometry_mask_policy",
    ) is not True:
        raise ThermalFeatureContractError(
            "above/hard pixels require connection to a valid hot region"
        )
    if _required_bool(
        geometry,
        "isolated_above_range_can_form_main_region",
        "geometry_mask_policy",
    ) is not False:
        raise ThermalFeatureContractError(
            "isolated above_range cannot form the main region"
        )
    if _required_bool(
        geometry,
        "isolated_hard_saturation_can_form_main_region",
        "geometry_mask_policy",
    ) is not False:
        raise ThermalFeatureContractError(
            "isolated hard_saturation cannot form the main region"
        )
    if _required_bool(
        geometry,
        "isolated_saturation_can_form_main_region",
        "geometry_mask_policy",
    ) is not False:
        raise ThermalFeatureContractError(
            "isolated saturation cannot form the main region"
        )
    if str(_required(
        geometry, "no_valid_hot_seed_result", "geometry_mask_policy"
    )) != "empty_region":
        raise ThermalFeatureContractError(
            "no_valid_hot_seed_result must be empty_region"
        )

    coordinates = contract.coordinate_convention
    if _required_int(
        coordinates,
        "physical_to_array_y_sign",
        "coordinate_convention",
    ) != -1:
        raise ThermalFeatureContractError("physical_to_array_y_sign must be -1")
    if _required_bool(
        coordinates,
        "physical_left_right_calibrated",
        "coordinate_convention",
    ) is not False:
        raise ThermalFeatureContractError(
            "physical left/right calibration must remain false"
        )


def _validate_features(features: tuple[FeatureDefinition, ...]) -> None:
    names = [feature.name for feature in features]
    if len(names) != len(set(names)):
        raise ThermalFeatureContractError("feature names must be unique")

    for feature in features:
        _validate_feature_name(feature.name)
        if feature.role not in _VALID_ROLES:
            raise ThermalFeatureContractError(f"invalid role for {feature.name}")
        _validate_unit_suffix(feature)

        if feature.role == "qc_only":
            if feature.default_model_input or feature.default_rsm_response:
                raise ThermalFeatureContractError(
                    f"QC feature {feature.name} must be excluded by default"
                )
        if feature.signed or "signed" in feature.name:
            if feature.cv_applicable:
                raise ThermalFeatureContractError(
                    f"signed feature {feature.name} must have cv_applicable=false"
                )
        if feature.role == "rejected":
            if feature.enabled:
                raise ThermalFeatureContractError(
                    f"rejected feature {feature.name} must not be enabled"
                )
            if not feature.rejection_reason:
                raise ThermalFeatureContractError(
                    f"rejected feature {feature.name} needs rejection_reason"
                )
            if feature.default_model_input or feature.default_rsm_response:
                raise ThermalFeatureContractError(
                    f"rejected feature {feature.name} must be excluded"
                )

    core_names = {feature.name for feature in features if feature.role == "core"}
    if "single_pixel_max_temperature_C" in core_names:
        raise ThermalFeatureContractError("single-pixel max must not be Core")


def _validate_feature_name(name: str) -> None:
    if not _NAME_RE.match(name):
        raise ThermalFeatureContractError(f"invalid feature name: {name}")
    if "__" in name:
        raise ThermalFeatureContractError(f"feature name must not contain '__': {name}")


def _validate_unit_suffix(feature: FeatureDefinition) -> None:
    unit = feature.unit
    if unit in {"dimensionless", "boolean", "string_list", "not_applicable"}:
        return
    if unit == "C_s":
        if not feature.name.endswith("_C_s"):
            raise ThermalFeatureContractError(
                f"{feature.name} must end with _C_s for unit C_s"
            )
        return
    suffix = _UNIT_SUFFIXES.get(unit)
    if suffix is None:
        raise ThermalFeatureContractError(
            f"{feature.name} has unsupported unit {unit!r}"
        )
    if not feature.name.endswith(suffix):
        raise ThermalFeatureContractError(
            f"{feature.name} must end with {suffix} for unit {unit}"
        )


def _validate_output_contracts(contract: ThermalFeatureContract) -> None:
    track = contract.track_level_output_contract
    condition = contract.condition_level_output_contract
    if track.expected_rows != _EXPECTED_TRACK_ROWS:
        raise ThermalFeatureContractError("track-level expected rows must be 57")
    if condition.expected_rows != _EXPECTED_CONDITION_ROWS:
        raise ThermalFeatureContractError("condition-level expected rows must be 19")
    if track.creation_enabled or condition.creation_enabled:
        raise ThermalFeatureContractError("output table creation must remain disabled")
