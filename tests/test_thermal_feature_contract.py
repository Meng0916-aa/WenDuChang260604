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


def _yaml():
    with open(CONTRACT, "r", encoding="utf-8") as stream:
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
