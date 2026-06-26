"""
Formal physical-calibration loader + validator.

Single, testable entry point for the spatial scale, frame rate, image geometry,
and physical process metadata of the 57-track experimental dataset. Formal ROI /
feature programs MUST obtain the pixel size from here, never from the legacy
``pixel_size_mm`` in configs/default.yaml. XTherm binary format and temperature
validity are authoritative in configs/xtherm_format.yaml; deprecated mirror
fields retained here are cross-checked against that authority.

Guarantees enforced on load (formal mode):
  - required spatial fields are present and positive;
  - pixel_area_mm2 is consistent with pixel_size_x_mm * pixel_size_y_mm;
  - if isotropic_scaling_assumed, x and y pixel sizes are equal;
  - NO dual active calibration: legacy_calibration must be disabled for formal
    processing;
  - spatial / geometric access works even when frame_rate_fps is null;
  - any TIME-domain feature that needs frame_rate_fps must call
    ``require_frame_rate()``, which raises a clear error while it is unconfirmed
    (it is NOT silently defaulted).
"""

import os
import yaml

from config.xtherm_format import load_xtherm_format

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_REQUIRED_SPATIAL = (
    "calibration_status", "calibration_source",
    "calibration_reference_length_mm", "calibration_reference_pixels",
    "pixel_size_x_mm", "pixel_size_y_mm", "pixel_area_mm2",
    "isotropic_scaling_assumed",
)

_AREA_REL_TOL = 1e-4   # pixel_area vs x*y
_ISO_ABS_TOL = 1e-9    # x vs y when isotropic
_XTherm_FORMAT_CONFIG = "configs/xtherm_format.yaml"


class CalibrationError(ValueError):
    """Raised when the formal calibration is missing, inconsistent, or a
    not-yet-confirmed value (e.g. frame rate) is requested."""


class PhysicalCalibration:
    """Validated view over configs/physical_calibration.yaml."""

    def __init__(self, cfg, path):
        self._cfg = cfg
        self.path = path

    # -- spatial --------------------------------------------------------
    @property
    def spatial(self):
        return self._cfg["spatial_calibration"]

    @property
    def pixel_size_x_mm(self):
        return float(self.spatial["pixel_size_x_mm"])

    @property
    def pixel_size_y_mm(self):
        return float(self.spatial["pixel_size_y_mm"])

    @property
    def pixel_area_mm2(self):
        return float(self.spatial["pixel_area_mm2"])

    @property
    def isotropic(self):
        return bool(self.spatial["isotropic_scaling_assumed"])

    @property
    def calibration_id(self):
        return self._cfg.get("calibration_id", "formal_150p2px_5mm")

    @property
    def calibration_reference_axis(self):
        return self.spatial.get("calibration_reference_axis")

    # -- geometry -------------------------------------------------------
    @property
    def geometry(self):
        return self._cfg.get("image_geometry", {})

    @property
    def image_height_px(self):
        return int(self.geometry["image_height_px"])

    @property
    def image_width_px(self):
        return int(self.geometry["image_width_px"])

    @property
    def scan_axis(self):
        return self.geometry.get("scan_axis")

    @property
    def transverse_axis(self):
        return self.geometry.get("transverse_axis")

    @property
    def image_scan_direction(self):
        return self.geometry.get("image_scan_direction")

    @property
    def array_scan_direction(self):
        return self.geometry.get("array_scan_direction")

    @property
    def physical_to_array_y_sign(self):
        sign = self.geometry.get("physical_to_array_y_sign")
        return None if sign is None else int(sign)

    @property
    def rotation_deg(self):
        return self.geometry.get("rotation_deg")

    @property
    def flip_horizontal(self):
        return self.geometry.get("flip_horizontal")

    @property
    def flip_vertical(self):
        return self.geometry.get("flip_vertical")

    @property
    def scan_direction(self):
        # Backward-compat: prefer the explicit image_scan_direction.
        return self.geometry.get("scan_direction") or self.image_scan_direction

    # -- temperature ----------------------------------------------------
    @property
    def temperature(self):
        return self._cfg["temperature_calibration"]

    @property
    def scale_C_per_count(self):
        return float(self.temperature["scale_C_per_count"])

    @property
    def saturation_value_C(self):
        return float(self.temperature["saturation_value_C"])

    @property
    def hard_saturation_threshold_C(self):
        return float(self.temperature["hard_saturation_threshold_C"])

    @property
    def valid_temperature_min_C(self):
        return float(self.temperature["valid_temperature_min_C"])

    @property
    def valid_temperature_max_C(self):
        return float(self.temperature["valid_temperature_max_C"])

    @property
    def measurement_range_status(self):
        return self.temperature.get("measurement_range_status")

    @property
    def robust_peak_quantile(self):
        return float(self.temperature["robust_peak_quantile"])

    @property
    def analysis_policy(self):
        """Temperature-analysis policy (mask/report rules; raw never modified)."""
        return self._cfg.get("temperature_analysis_policy", {})

    # -- acquisition / time ---------------------------------------------
    @property
    def acquisition(self):
        return self._cfg.get("acquisition", {})

    @property
    def working_distance_mm(self):
        wd = self.acquisition.get("working_distance_mm")
        return None if wd is None else float(wd)

    @property
    def working_distance_reference(self):
        return self.acquisition.get("working_distance_reference")

    @property
    def effective_frame_rule(self):
        return self._cfg.get("effective_frame_rule", {})

    @property
    def frame_rate_fps(self):
        """Returns the confirmed frame rate or None (not yet confirmed)."""
        return self.acquisition.get("frame_rate_fps")

    def has_frame_rate(self):
        return self.frame_rate_fps is not None

    def require_frame_rate(self):
        """Return a confirmed frame rate, or raise.

        Time-domain features (cooling rate C/s, dwell time s, temperature AUC,
        scan distance per frame, effective scan duration) must call this. While
        the frame rate is unconfirmed it raises instead of defaulting silently.
        """
        fps = self.frame_rate_fps
        if fps is None:
            raise CalibrationError(
                "frame_rate_fps is not confirmed yet (acquisition.frame_rate_fps "
                "is null). Time-domain features (cooling rate, dwell time, AUC, "
                "scan-distance/frame, scan duration) cannot be computed. Provide a "
                "verified frame rate in configs/physical_calibration.yaml first. "
                "Spatial/geometric features do not need it.")
        fps = float(fps)
        if fps <= 0:
            raise CalibrationError(f"frame_rate_fps must be > 0, got {fps}")
        return fps


def _validate(cfg, path, formal):
    if "spatial_calibration" not in cfg:
        raise CalibrationError(f"missing 'spatial_calibration' block in {path}")
    sp = cfg["spatial_calibration"]
    missing = [k for k in _REQUIRED_SPATIAL if k not in sp]
    if missing:
        raise CalibrationError(f"missing spatial fields {missing} in {path}")

    x = float(sp["pixel_size_x_mm"])
    y = float(sp["pixel_size_y_mm"])
    area = float(sp["pixel_area_mm2"])
    if not (x > 0 and y > 0 and area > 0):
        raise CalibrationError(
            f"pixel sizes/area must be positive (x={x}, y={y}, area={area})")

    if abs(area - x * y) > _AREA_REL_TOL * (x * y):
        raise CalibrationError(
            f"pixel_area_mm2 {area} inconsistent with x*y {x * y} in {path}")

    if bool(sp["isotropic_scaling_assumed"]) and abs(x - y) > _ISO_ABS_TOL:
        raise CalibrationError(
            f"isotropic_scaling_assumed=true but pixel_size_x ({x}) != "
            f"pixel_size_y ({y}) in {path}")

    # Geometry sanity (when a scan axis has been confirmed): the physical-Y to
    # array-row sign must be exactly +/-1. Null/unspecified geometry is allowed.
    geom = cfg.get("image_geometry", {})
    if geom.get("scan_axis") is not None:
        sign = geom.get("physical_to_array_y_sign")
        if sign is not None and int(sign) not in (-1, 1):
            raise CalibrationError(
                f"physical_to_array_y_sign must be -1 or +1, got {sign!r} in {path}")

    if formal:
        if str(sp["calibration_status"]) != "confirmed":
            raise CalibrationError(
                f"formal mode requires spatial calibration_status=confirmed, "
                f"got {sp['calibration_status']!r}")
        legacy = cfg.get("legacy_calibration", {})
        if legacy.get("enabled_for_formal_processing", False):
            raise CalibrationError(
                "dual active calibration: legacy_calibration."
                "enabled_for_formal_processing must be false in formal mode.")
        _validate_temperature_compatibility_mirror(cfg, path)


def _validate_temperature_compatibility_mirror(cfg, path):
    """Cross-check deprecated temperature mirror fields against xtherm_format."""
    if "temperature_calibration" not in cfg:
        return

    temp = cfg["temperature_calibration"]
    if temp.get("role") != "deprecated_compatibility_mirror":
        raise CalibrationError(
            "temperature_calibration.role must be "
            "'deprecated_compatibility_mirror'; formal temperature format "
            f"authority is {_XTherm_FORMAT_CONFIG}")
    if temp.get("authoritative_source") != _XTherm_FORMAT_CONFIG:
        raise CalibrationError(
            "temperature_calibration.authoritative_source must be "
            f"{_XTherm_FORMAT_CONFIG!r}")
    if temp.get("formal_consumers_must_use_authoritative_source") is not True:
        raise CalibrationError(
            "temperature_calibration.formal_consumers_must_use_authoritative_source "
            "must be true")

    xtherm_path = os.path.join(_ROOT, _XTherm_FORMAT_CONFIG)
    xtherm = load_xtherm_format(xtherm_path)
    geom = cfg.get("image_geometry", {})
    comparisons = (
        ("image_geometry.image_height_px", geom.get("image_height_px"),
         xtherm.height_px),
        ("image_geometry.image_width_px", geom.get("image_width_px"),
         xtherm.width_px),
        ("temperature_calibration.header_bytes", temp.get("header_bytes"),
         xtherm.header_bytes),
        ("temperature_calibration.raw_dtype", temp.get("raw_dtype"),
         xtherm.raw_dtype),
        ("temperature_calibration.byte_order", temp.get("byte_order"),
         xtherm.byte_order),
        ("temperature_calibration.scale_C_per_count",
         temp.get("scale_C_per_count"), xtherm.scale_C_per_count),
        ("temperature_calibration.valid_temperature_min_C",
         temp.get("valid_temperature_min_C"),
         xtherm.camera_valid_temperature_min_C),
        ("temperature_calibration.valid_temperature_max_C",
         temp.get("valid_temperature_max_C"),
         xtherm.camera_valid_temperature_max_C),
        ("temperature_calibration.hard_saturation_threshold_C",
         temp.get("hard_saturation_threshold_C"),
         xtherm.hard_saturation_threshold_C),
    )

    for field, mirror_value, authoritative_value in comparisons:
        if _normalized_scalar(mirror_value) != _normalized_scalar(authoritative_value):
            raise CalibrationError(
                f"{field} conflicts between {path} and {_XTherm_FORMAT_CONFIG}: "
                f"physical_calibration.yaml={mirror_value!r}, "
                f"xtherm_format.yaml={authoritative_value!r}")


def _normalized_scalar(value):
    if isinstance(value, float):
        return round(value, 12)
    if isinstance(value, int):
        return value
    try:
        return round(float(value), 12)
    except (TypeError, ValueError):
        return str(value)


def load_physical_calibration(path="configs/physical_calibration.yaml", formal=True):
    """Load + validate the formal physical calibration.

    Args:
        path: YAML path (relative to project root or absolute).
        formal: enforce formal-mode rules (confirmed status, no dual calibration).

    Returns:
        PhysicalCalibration
    """
    abs_path = path if os.path.isabs(path) else os.path.join(_ROOT, path)
    if not os.path.isfile(abs_path):
        raise CalibrationError(f"calibration file not found: {abs_path}")
    with open(abs_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    _validate(cfg, abs_path, formal)
    return PhysicalCalibration(cfg, abs_path)
