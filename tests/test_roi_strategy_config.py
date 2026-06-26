import copy
from dataclasses import FrozenInstanceError
import math
import os
import sys

import pytest
import yaml


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from config.roi_strategy import RoiStrategyError, load_roi_strategy


ROI_CONFIG = os.path.join(_ROOT, "configs", "roi_strategy.yaml")
FORMAL_PIPELINE = os.path.join(_ROOT, "configs", "formal_pipeline.yaml")


def _yaml(path):
    with open(path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _roi_yaml():
    return _yaml(ROI_CONFIG)


def _write_variant(tmp_path, mutator):
    cfg = copy.deepcopy(_roi_yaml())
    mutator(cfg)
    path = tmp_path / "roi_strategy.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def test_load_formal_roi_strategy():
    cfg = load_roi_strategy(ROI_CONFIG)
    assert cfg.schema_version == 1
    assert cfg.status.role == "formal_roi_strategy"
    assert cfg.status.strategy_name == "global_roi_plus_tracking_window"
    assert cfg.thresholds_C.envelope == 700.0
    assert cfg.thresholds_C.core == 800.0


def test_strategy_status_gates_closed():
    cfg = load_roi_strategy(ROI_CONFIG)
    assert cfg.status.evaluation_status == "completed"
    assert cfg.status.activation_status == "evaluated_not_activated"
    assert cfg.status.formal_roi_generation_enabled is False
    assert cfg.status.formal_feature_extraction_enabled is False


def test_fixed_global_roi_half_open_geometry():
    roi = load_roi_strategy(ROI_CONFIG).fixed_global_roi
    assert (roi.row_start, roi.row_stop) == (175, 495)
    assert (roi.col_start, roi.col_stop) == (86, 334)
    assert roi.height_px == roi.row_stop - roi.row_start == 320
    assert roi.width_px == roi.col_stop - roi.col_start == 248


def test_global_roi_within_source_image():
    cfg = load_roi_strategy(ROI_CONFIG)
    roi = cfg.fixed_global_roi
    coords = cfg.coordinate_system
    assert 0 <= roi.row_start < roi.row_stop <= coords.source_image_height_px
    assert 0 <= roi.col_start < roi.col_stop <= coords.source_image_width_px


def test_tracking_window_dimensions_and_coverage():
    window = load_roi_strategy(ROI_CONFIG).tracking_window
    assert window.width_px == 256
    assert window.height_px == 216
    assert window.coverage.envelope_700_fraction == 1.0
    assert window.coverage.core_800_fraction == 1.0
    assert window.clipped_frame_count == 0
    assert window.edge_adjusted_frame_count == 0


def test_tracking_window_physical_size_consistent():
    cfg = load_roi_strategy(ROI_CONFIG)
    window = cfg.tracking_window
    pixel = cfg.pixel_size
    assert math.isclose(
        window.width_mm,
        window.width_px * pixel.pixel_size_x_mm,
        rel_tol=1e-9,
        abs_tol=1e-9,
    )
    assert math.isclose(
        window.height_mm,
        window.height_px * pixel.pixel_size_y_mm,
        rel_tol=1e-9,
        abs_tol=1e-9,
    )


def test_effective_frame_rule_is_frames_1_onward():
    frames = load_roi_strategy(ROI_CONFIG).effective_frames
    assert frames.startup_frames_excluded == 1
    assert frames.start_index_0_based == 1
    assert frames.python_slice == "frames[1:]"


def test_current_strategy_evaluated_but_not_enabled():
    cfg = load_roi_strategy(ROI_CONFIG)
    assert cfg.fixed_global_roi.evaluation_accepted is True
    assert cfg.fixed_global_roi.formal_use_enabled is False
    assert cfg.tracking_window.evaluation_accepted is True
    assert cfg.tracking_window.formal_use_enabled is False


def test_legacy_candidates_rejected():
    legacy = load_roi_strategy(ROI_CONFIG).legacy_candidates
    assert legacy.fixed_roi.evaluation_accepted is False
    assert legacy.fixed_roi.formal_use_enabled is False
    assert legacy.fixed_roi.minimum_envelope_700_coverage_fraction == 0.9971
    assert legacy.tracking_window.width_px == 192
    assert legacy.tracking_window.height_px == 208
    assert legacy.tracking_window.evaluation_accepted is False
    assert legacy.tracking_window.formal_use_enabled is False


def test_formal_pipeline_references_roi_strategy_config():
    pipeline = _yaml(FORMAL_PIPELINE)
    roi_strategy = pipeline["roi_strategy"]
    assert roi_strategy["strategy_config"] == "configs/roi_strategy.yaml"
    assert roi_strategy["status"] == "evaluated_not_activated"
    assert roi_strategy["generate_formal_roi_matrices"] is False
    assert pipeline["processing"]["formal_feature_extraction_enabled"] is False


def test_invalid_geometry_raises(tmp_path):
    path = _write_variant(
        tmp_path,
        lambda cfg: cfg["fixed_global_roi"].update(row_stop=174),
    )
    with pytest.raises(RoiStrategyError):
        load_roi_strategy(path)


def test_invalid_coverage_raises(tmp_path):
    def mutate(cfg):
        cfg["tracking_window"]["coverage"]["envelope_700_fraction"] = 1.01

    path = _write_variant(tmp_path, mutate)
    with pytest.raises(RoiStrategyError):
        load_roi_strategy(path)


def test_invalid_frame_rule_raises(tmp_path):
    def mutate(cfg):
        cfg["effective_frames"]["python_slice"] = "frames[:]"

    path = _write_variant(tmp_path, mutate)
    with pytest.raises(RoiStrategyError):
        load_roi_strategy(path)


def test_enabled_roi_generation_raises(tmp_path):
    def mutate(cfg):
        cfg["status"]["formal_roi_generation_enabled"] = True

    path = _write_variant(tmp_path, mutate)
    with pytest.raises(RoiStrategyError):
        load_roi_strategy(path)


def test_enabled_feature_extraction_raises(tmp_path):
    def mutate(cfg):
        cfg["status"]["formal_feature_extraction_enabled"] = True

    path = _write_variant(tmp_path, mutate)
    with pytest.raises(RoiStrategyError):
        load_roi_strategy(path)


def test_enabled_formal_use_raises(tmp_path):
    def mutate(cfg):
        cfg["fixed_global_roi"]["formal_use_enabled"] = True

    path = _write_variant(tmp_path, mutate)
    with pytest.raises(RoiStrategyError):
        load_roi_strategy(path)


def test_accepted_legacy_candidate_raises(tmp_path):
    def mutate(cfg):
        cfg["legacy_candidates"]["fixed_roi"]["evaluation_accepted"] = True

    path = _write_variant(tmp_path, mutate)
    with pytest.raises(RoiStrategyError):
        load_roi_strategy(path)


def test_missing_local_results_does_not_block_loading(tmp_path):
    def mutate(cfg):
        cfg["sources"]["evaluation_summary_local_artifact"] = str(
            tmp_path / "missing" / "roi_strategy_summary.json"
        )

    path = _write_variant(tmp_path, mutate)
    cfg = load_roi_strategy(path)
    assert cfg.sources.required_for_config_loading is False


def test_loading_has_no_script_or_data_side_effects(tmp_path, monkeypatch):
    path = _write_variant(tmp_path, lambda cfg: None)
    work = tmp_path / "work"
    work.mkdir()
    before_files = {p.relative_to(work) for p in work.rglob("*")}
    before_modules = {
        name for name in sys.modules if name == "scripts" or name.startswith("scripts.")
    }

    monkeypatch.chdir(work)
    load_roi_strategy(path)

    after_files = {p.relative_to(work) for p in work.rglob("*")}
    after_modules = {
        name for name in sys.modules if name == "scripts" or name.startswith("scripts.")
    }
    assert after_files == before_files
    assert after_modules == before_modules
    assert not (work / "data").exists()
    assert not (work / "results").exists()


def test_bool_is_rejected_for_integer_field(tmp_path):
    def mutate(cfg):
        cfg["tracking_window"]["width_px"] = True
        cfg["tracking_window"]["width_mm"] = cfg["pixel_size"]["pixel_size_x_mm"]

    path = _write_variant(tmp_path, mutate)
    with pytest.raises(RoiStrategyError, match="tracking_window.width_px"):
        load_roi_strategy(path)


def test_numeric_string_is_rejected_for_float_field(tmp_path):
    def mutate(cfg):
        cfg["fixed_global_roi"]["coverage"]["envelope_700_fraction"] = "1.0"

    path = _write_variant(tmp_path, mutate)
    with pytest.raises(RoiStrategyError, match="fixed_global_roi.coverage.envelope_700_fraction"):
        load_roi_strategy(path)


def test_nan_is_rejected_for_float_field(tmp_path):
    def mutate(cfg):
        cfg["fixed_global_roi"]["coverage"]["envelope_700_fraction"] = float("nan")

    path = _write_variant(tmp_path, mutate)
    with pytest.raises(RoiStrategyError, match="must be finite"):
        load_roi_strategy(path)


def test_positive_infinity_is_rejected_for_float_field(tmp_path):
    def mutate(cfg):
        cfg["fixed_global_roi"]["coverage"]["envelope_700_fraction"] = float("inf")

    path = _write_variant(tmp_path, mutate)
    with pytest.raises(RoiStrategyError, match="must be finite"):
        load_roi_strategy(path)


def test_negative_infinity_is_rejected_for_float_field(tmp_path):
    def mutate(cfg):
        cfg["fixed_global_roi"]["coverage"]["envelope_700_fraction"] = float("-inf")

    path = _write_variant(tmp_path, mutate)
    with pytest.raises(RoiStrategyError, match="must be finite"):
        load_roi_strategy(path)


def test_loaded_config_is_deeply_immutable():
    cfg = load_roi_strategy(ROI_CONFIG)
    with pytest.raises(AttributeError):
        cfg.fixed_global_roi.purpose.append("test")
    with pytest.raises(FrozenInstanceError):
        cfg.fixed_global_roi.row_start = 1


def test_empty_yaml_raises(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    with pytest.raises(RoiStrategyError):
        load_roi_strategy(path)


def test_non_mapping_yaml_root_raises(tmp_path):
    path = tmp_path / "list.yaml"
    path.write_text("- item1\n- item2\n", encoding="utf-8")
    with pytest.raises(RoiStrategyError):
        load_roi_strategy(path)
