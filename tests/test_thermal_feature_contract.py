import copy
from dataclasses import FrozenInstanceError
import os
import sys

import pytest
import yaml


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from config.thermal_feature_contract import (
    ThermalFeatureContractError,
    load_thermal_feature_contract,
)


CONTRACT = os.path.join(_ROOT, "configs", "thermal_feature_contract.yaml")
XTHERM_CONFIG = os.path.join(_ROOT, "configs", "xtherm_format.yaml")

EXPECTED_CORE_NAMES = {
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

EXPECTED_CORE_SEMANTICS = {
    "mean_active_frame_valid_temperature_C": (
        "C", "tracking_window", "active_700_frames",
        "arithmetic_mean_of_frame_means", True,
    ),
    "max_frame_p999_valid_temperature_C": (
        "C", "tracking_window", "active_700_frames",
        "maximum_of_frame_p999_values", True,
    ),
    "mean_main_area_above_700_C_mm2": (
        "mm2", "tracking_window", "active_700_frames", "arithmetic_mean", True,
    ),
    "mean_main_area_above_800_C_mm2": (
        "mm2", "tracking_window", "active_700_frames", "arithmetic_mean", True,
    ),
    "mean_main_transverse_width_above_700_C_mm": (
        "mm", "tracking_window", "active_700_frames", "arithmetic_mean", True,
    ),
    "mean_main_scan_length_above_700_C_mm": (
        "mm", "tracking_window", "active_700_frames", "arithmetic_mean", True,
    ),
    "centroid_path_length_mm": (
        "mm", "fixed_global_roi", "active_700_frames",
        "sum_distances_between_adjacent_frames_without_crossing_missing_centroid_gaps",
        True,
    ),
    "signed_scan_direction_displacement_mm": (
        "mm", "fixed_global_roi", "active_700_frames",
        "last_valid_y_mm_minus_first_valid_y_mm", False,
    ),
    "signed_transverse_drift_mm": (
        "mm", "fixed_global_roi", "active_700_frames",
        "last_valid_x_mm_minus_first_valid_x_mm", False,
    ),
    "centroid_transverse_jitter_mm": (
        "mm", "fixed_global_roi", "active_700_frames",
        "rms_residual_x_from_line_fit_x_equals_a_y_plus_b", True,
    ),
    "median_frame_p95_internal_gradient_magnitude_700_C_per_mm": (
        "C_per_mm", "tracking_window", "active_700_frames",
        "median_of_frame_p95_gradient_values", True,
    ),
    "mean_signed_thermal_centroid_offset_from_geometric_center_mm": (
        "mm", "tracking_window", "active_700_frames", "arithmetic_mean", False,
    ),
    "mean_left_right_excess_temperature_asymmetry_700_fraction": (
        "fraction", "tracking_window", "active_700_frames",
        "arithmetic_mean", True,
    ),
    "hot_core_presence_duration_800_C_s": (
        "s", "tracking_window", "effective_frames",
        "active_800_frame_count / 52.0", True,
    ),
    "main_area_above_700_C_temporal_cv": (
        "cv", "tracking_window", "active_700_frames",
        "sample_std_area700_ddof1_divided_by_mean_area700", True,
    ),
}


def _yaml(path=CONTRACT):
    with open(path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _write_variant(tmp_path, mutator):
    cfg = copy.deepcopy(_yaml())
    mutator(cfg)
    path = tmp_path / "thermal_feature_contract.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def test_load_formal_thermal_feature_contract():
    contract = load_thermal_feature_contract(CONTRACT)
    assert contract.schema_version == 1
    assert contract.status.role == "formal_thermal_feature_contract"
    assert contract.status.design_status == "approved"
    assert contract.status.execution_status == "designed_not_executed"


def test_core_count_is_exactly_15():
    contract = load_thermal_feature_contract(CONTRACT)
    assert len(contract.core_features) == 15


def test_feature_names_are_unique():
    contract = load_thermal_feature_contract(CONTRACT)
    names = [feature.name for feature in contract.all_features]
    assert len(names) == len(set(names))


def test_naming_and_unit_suffix_rules():
    contract = load_thermal_feature_contract(CONTRACT)
    suffixes = contract.feature_naming_rules["suffixes"]
    for feature in contract.all_features:
        assert " " not in feature.name
        assert "-" not in feature.name
        if feature.unit in {"dimensionless", "boolean", "string_list", "not_applicable"}:
            continue
        expected = "_C_s" if feature.unit == "C_s" else suffixes[feature.unit]
        assert feature.name.endswith(expected), (feature.name, feature.unit)


def test_status_gates_are_closed():
    contract = load_thermal_feature_contract(CONTRACT)
    assert contract.status.formal_feature_extraction_enabled is False
    assert contract.status.formal_roi_generation_enabled is False
    assert contract.track_level_output_contract.creation_enabled is False
    assert contract.condition_level_output_contract.creation_enabled is False


def test_qc_features_excluded_by_default():
    contract = load_thermal_feature_contract(CONTRACT)
    assert contract.qc_only_features
    for feature in contract.qc_only_features:
        assert feature.default_model_input is False
        assert feature.default_rsm_response is False


def test_signed_features_have_cv_disabled():
    contract = load_thermal_feature_contract(CONTRACT)
    signed = [f for f in contract.all_features if f.signed or "signed" in f.name]
    assert signed
    assert all(feature.cv_applicable is False for feature in signed)


def test_secondary_and_rejected_classification():
    contract = load_thermal_feature_contract(CONTRACT)
    secondary_names = {feature.name for feature in contract.secondary_features}
    rejected_names = {feature.name for feature in contract.rejected_features}
    assert "time_to_peak_s" in secondary_names
    assert "temperature_auc_C_s" in secondary_names
    assert "single_pixel_max_temperature_C" in rejected_names
    assert "concatenated_T1_T2_T3_sequence" in rejected_names
    assert all(feature.enabled is False for feature in contract.rejected_features)
    assert all(feature.rejection_reason for feature in contract.rejected_features)


def test_single_pixel_max_is_not_core():
    contract = load_thermal_feature_contract(CONTRACT)
    core_names = {feature.name for feature in contract.core_features}
    assert "single_pixel_max_temperature_C" not in core_names


def test_no_default_yaml_as_formal_source():
    contract = load_thermal_feature_contract(CONTRACT)
    assert "configs/default.yaml" not in set(contract.sources.values())


def test_expected_track_and_condition_rows():
    contract = load_thermal_feature_contract(CONTRACT)
    assert contract.track_level_output_contract.expected_rows == 57
    assert contract.condition_level_output_contract.expected_rows == 19


def test_exact_core_feature_name_set():
    contract = load_thermal_feature_contract(CONTRACT)
    assert {feature.name for feature in contract.core_features} == EXPECTED_CORE_NAMES


def test_each_core_feature_semantics_match_approved_contract():
    contract = load_thermal_feature_contract(CONTRACT)
    by_name = {feature.name: feature for feature in contract.core_features}
    for name, expected in EXPECTED_CORE_SEMANTICS.items():
        feature = by_name[name]
        assert (
            feature.unit,
            feature.region,
            feature.frame_population,
            feature.track_aggregation,
            feature.cv_applicable,
        ) == expected


def test_thresholds_match_xtherm_format():
    cfg = _yaml()
    xtherm = _yaml(XTHERM_CONFIG)
    temp = cfg["temperature_validity"]
    qc = xtherm["temperature_qc"]
    assert temp["valid_min_C"] == qc["camera_valid_temperature_min_C"]
    assert temp["valid_max_C"] == qc["camera_valid_temperature_max_C"]
    assert temp["hard_saturation_threshold_C"] == qc["hard_saturation_threshold_C"]


def test_valid_temperature_ordering():
    cfg = _yaml()
    temp = cfg["temperature_validity"]
    thresholds = cfg["geometry_mask_policy"]["main_region_thresholds_C"]
    assert temp["valid_min_C"] < temp["valid_max_C"] < temp["hard_saturation_threshold_C"]
    assert thresholds["envelope"] == 700.0
    assert thresholds["core"] == 800.0
    assert temp["valid_min_C"] <= thresholds["envelope"] < thresholds["core"]
    assert thresholds["core"] <= temp["valid_max_C"]


def test_geometry_connectivity_is_eight():
    contract = load_thermal_feature_contract(CONTRACT)
    assert contract.geometry_mask_policy["connectivity"] == 8


def test_min_component_area_is_nine():
    contract = load_thermal_feature_contract(CONTRACT)
    assert contract.geometry_mask_policy["min_component_area_px"] == 9


def test_min_component_area_rejects_bool(tmp_path):
    def mutate(cfg):
        cfg["geometry_mask_policy"]["min_component_area_px"] = True

    path = _write_variant(tmp_path, mutate)
    with pytest.raises(ThermalFeatureContractError, match="min_component_area_px"):
        load_thermal_feature_contract(path)


def test_seed_required_for_main_region():
    contract = load_thermal_feature_contract(CONTRACT)
    assert contract.geometry_mask_policy["valid_hot_seed_required"] is True


def test_expansion_requires_seed_connection():
    contract = load_thermal_feature_contract(CONTRACT)
    policy = contract.geometry_mask_policy
    assert set(policy["expansion_candidates"]) == {"above_range", "hard_saturation"}
    assert policy["expansion_requires_connection_to_valid_hot_seed"] is True


def test_main_component_selection_is_largest_seeded():
    contract = load_thermal_feature_contract(CONTRACT)
    assert (
        contract.geometry_mask_policy["main_component_selection"]
        == "largest_seeded_connected_component"
    )


def test_internal_hole_filling_is_enabled():
    contract = load_thermal_feature_contract(CONTRACT)
    assert contract.geometry_mask_policy["fill_internal_holes"] is True


def test_isolated_above_range_cannot_form_region():
    contract = load_thermal_feature_contract(CONTRACT)
    assert (
        contract.geometry_mask_policy["isolated_above_range_can_form_main_region"]
        is False
    )


def test_isolated_hard_saturation_cannot_form_region():
    contract = load_thermal_feature_contract(CONTRACT)
    assert (
        contract.geometry_mask_policy["isolated_hard_saturation_can_form_main_region"]
        is False
    )


def test_no_valid_seed_returns_empty_region():
    contract = load_thermal_feature_contract(CONTRACT)
    assert contract.geometry_mask_policy["no_valid_hot_seed_result"] == "empty_region"


def test_bad_status_raises(tmp_path):
    def mutate(cfg):
        cfg["status"]["execution_status"] = "executed"

    path = _write_variant(tmp_path, mutate)
    with pytest.raises(ThermalFeatureContractError):
        load_thermal_feature_contract(path)


def test_duplicate_feature_name_raises(tmp_path):
    def mutate(cfg):
        cfg["core_features"][1]["name"] = cfg["core_features"][0]["name"]

    path = _write_variant(tmp_path, mutate)
    with pytest.raises(ThermalFeatureContractError, match="unique"):
        load_thermal_feature_contract(path)


def test_illegal_unit_raises(tmp_path):
    def mutate(cfg):
        cfg["core_features"][0]["unit"] = "kelvin"

    path = _write_variant(tmp_path, mutate)
    with pytest.raises(ThermalFeatureContractError, match="unsupported unit"):
        load_thermal_feature_contract(path)


def test_bool_as_int_raises(tmp_path):
    def mutate(cfg):
        cfg["track_level_output_contract"]["expected_rows"] = True

    path = _write_variant(tmp_path, mutate)
    with pytest.raises(ThermalFeatureContractError, match="expected_rows"):
        load_thermal_feature_contract(path)


def test_numeric_string_raises(tmp_path):
    def mutate(cfg):
        cfg["temperature_validity"]["valid_min_C"] = "300.0"

    path = _write_variant(tmp_path, mutate)
    with pytest.raises(ThermalFeatureContractError, match="valid_min_C"):
        load_thermal_feature_contract(path)


def test_nan_raises(tmp_path):
    def mutate(cfg):
        cfg["temperature_validity"]["valid_min_C"] = float("nan")

    path = _write_variant(tmp_path, mutate)
    with pytest.raises(ThermalFeatureContractError, match="valid_min_C"):
        load_thermal_feature_contract(path)


def test_infinity_raises(tmp_path):
    def mutate(cfg):
        cfg["temperature_validity"]["valid_min_C"] = float("inf")

    path = _write_variant(tmp_path, mutate)
    with pytest.raises(ThermalFeatureContractError, match="valid_min_C"):
        load_thermal_feature_contract(path)


def test_loaded_contract_is_deeply_immutable():
    contract = load_thermal_feature_contract(CONTRACT)
    with pytest.raises(FrozenInstanceError):
        contract.core_features[0].name = "changed"
    with pytest.raises(AttributeError):
        contract.core_features.append(contract.core_features[0])
    with pytest.raises(TypeError):
        contract.sources["new"] = "value"
    with pytest.raises(TypeError):
        contract.frame_definitions["effective_frame"]["python_slice"] = "frames[:]"


def test_loading_has_no_data_results_or_numpy_side_effects(tmp_path, monkeypatch):
    before_modules = {
        name for name in sys.modules if name == "numpy" or name.startswith("numpy.")
    }
    work = tmp_path / "work"
    work.mkdir()
    before_files = {p.relative_to(work) for p in work.rglob("*")}

    monkeypatch.chdir(work)
    load_thermal_feature_contract()

    after_modules = {
        name for name in sys.modules if name == "numpy" or name.startswith("numpy.")
    }
    after_files = {p.relative_to(work) for p in work.rglob("*")}
    assert after_modules == before_modules
    assert after_files == before_files
    assert not (work / "data").exists()
    assert not (work / "results").exists()
