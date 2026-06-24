"""
Formal-pipeline configuration gate.

Loads/validates the active formal pipeline entry (``configs/formal_pipeline.yaml``)
and enforces the separation between the FORMAL 57-track pipeline and the LEGACY
pilot configuration (``configs/default.yaml``):

  * the formal config entry must be ``status: active`` and must NOT carry any
    machine-absolute path;
  * formal code must refuse to use ``configs/default.yaml`` (legacy_pilot), its
    legacy ROI, its legacy pixel size, or the early-test ``dataset`` path.

Pure stdlib + PyYAML; no numpy / torch.
"""

import os
import re

import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FORMAL_CALIBRATION_CONFIG = "configs/physical_calibration.yaml"

# Drive-letter (D:\ or D:/) or POSIX-absolute (/...) path values are "machine
# absolute"; relative repo paths like "configs/experiments.yaml" are fine.
_ABS_PATH = re.compile(r"^(?:[A-Za-z]:[\\/]|/)")


class FormalConfigError(RuntimeError):
    """Raised when a formal-processing rule is violated."""


def _read_yaml(path):
    abs_path = path if os.path.isabs(path) else os.path.join(_ROOT, path)
    if not os.path.isfile(abs_path):
        raise FormalConfigError(f"config not found: {abs_path}")
    with open(abs_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def find_absolute_path_values(obj, prefix=""):
    """Recursively collect ``(dotted_key, value)`` pairs whose string value looks
    like a machine-absolute path."""
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            found += find_absolute_path_values(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            found += find_absolute_path_values(v, f"{prefix}[{i}]")
    elif isinstance(obj, str) and _ABS_PATH.match(obj.strip()):
        found.append((prefix, obj))
    return found


def load_formal_pipeline(path="configs/formal_pipeline.yaml"):
    """Load + validate the active formal pipeline config.

    Raises FormalConfigError if status != active or any machine-absolute path is
    present.
    """
    cfg = _read_yaml(path)
    status = (cfg.get("pipeline") or {}).get("status")
    if status != "active":
        raise FormalConfigError(
            f"formal pipeline status must be 'active', got {status!r} in {path}")
    abs_found = find_absolute_path_values(cfg)
    if abs_found:
        raise FormalConfigError(
            f"formal pipeline config must not contain machine-absolute paths: "
            f"{abs_found}")
    return cfg


def formal_calibration_config(formal_cfg=None):
    """The calibration config the formal pipeline must use (physical_calibration)."""
    if formal_cfg is None:
        formal_cfg = load_formal_pipeline()
    cc = (formal_cfg.get("pipeline") or {}).get("calibration_config")
    if cc != FORMAL_CALIBRATION_CONFIG:
        raise FormalConfigError(
            f"formal calibration_config must be {FORMAL_CALIBRATION_CONFIG!r}, "
            f"got {cc!r}")
    return cc


# ---------------------------------------------------------------------------
# Legacy-isolation guards (used by formal code paths)
# ---------------------------------------------------------------------------

def is_formal_processing_enabled(default_cfg):
    """True iff configs/default.yaml declares formal processing enabled."""
    return bool((default_cfg.get("config_status") or {}).get(
        "formal_processing_enabled", False))


def assert_not_legacy_default(default_cfg):
    """Refuse to use configs/default.yaml in formal mode (it is legacy_pilot)."""
    if not is_formal_processing_enabled(default_cfg):
        role = (default_cfg.get("config_status") or {}).get("role", "legacy")
        raise FormalConfigError(
            f"configs/default.yaml is {role!r} (formal_processing_enabled: false); "
            "the formal 57-track pipeline must use configs/formal_pipeline.yaml + "
            "configs/physical_calibration.yaml, not default.yaml.")


def reject_legacy_roi(default_cfg):
    """Refuse the legacy default.yaml ROI block for formal processing."""
    roi = default_cfg.get("roi", {}) or {}
    if not roi.get("enabled", False) or roi.get("status") != "approved":
        raise FormalConfigError(
            f"default.yaml ROI is status={roi.get('status')!r} enabled="
            f"{roi.get('enabled')!r}; the formal pipeline must use the reviewed "
            "ROI strategy, never this legacy pilot ROI.")


def assert_no_legacy_dataset_path(path):
    """Refuse the early-test 'dataset' directory / dataset.npy in formal mode."""
    parts = {p.lower() for p in str(path).replace("\\", "/").split("/")}
    if "dataset" in parts or str(path).replace("\\", "/").lower().endswith("dataset.npy"):
        raise FormalConfigError(f"formal pipeline must not read legacy dataset path: {path}")


def assert_pixel_size_from_calibration(default_cfg):
    """Refuse to take the formal pixel size from default.yaml's legacy value.

    The formal spatial scale must come from configs/physical_calibration.yaml.
    """
    if not is_formal_processing_enabled(default_cfg):
        raise FormalConfigError(
            "formal pixel size must come from configs/physical_calibration.yaml; "
            "configs/default.yaml.thermal_field_features.pixel_size_mm is a legacy "
            "pilot value and must not be used for formal processing.")
