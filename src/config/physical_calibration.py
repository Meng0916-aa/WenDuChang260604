"""
Formal physical-calibration loader + validator.

Single, testable entry point for the spatial scale and temperature binary
calibration of the 57-track experimental dataset. Formal ROI / feature programs
MUST obtain the pixel size from here, never from the legacy ``pixel_size_mm`` in
configs/default.yaml.

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

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_REQUIRED_SPATIAL = (
    "calibration_status", "calibration_source",
    "calibration_reference_length_mm", "calibration_reference_pixels",
    "pixel_size_x_mm", "pixel_size_y_mm", "pixel_area_mm2",
    "isotropic_scaling_assumed",
)

_AREA_REL_TOL = 1e-4   # pixel_area vs x*y
_ISO_ABS_TOL = 1e-9    # x vs y when isotropic


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

    # -- geometry -------------------------------------------------------
    @property
    def image_height_px(self):
        return int(self._cfg["image_geometry"]["image_height_px"])

    @property
    def image_width_px(self):
        return int(self._cfg["image_geometry"]["image_width_px"])

    @property
    def scan_axis(self):
        return self._cfg["image_geometry"].get("scan_axis")

    @property
    def scan_direction(self):
        return self._cfg["image_geometry"].get("scan_direction")

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
    def robust_peak_quantile(self):
        return float(self.temperature["robust_peak_quantile"])

    # -- acquisition / time ---------------------------------------------
    @property
    def frame_rate_fps(self):
        """Returns the confirmed frame rate or None (not yet confirmed)."""
        return self._cfg.get("acquisition", {}).get("frame_rate_fps")

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
