# Formal Decision Log

This file is append-only in normal use. Do not overwrite history casually.

## 2026-06-25 Codex Takeover Migration

### Decision 1

Decision: `configs/xtherm_format.yaml` is the formal source for XTherm binary
format, temperature conversion, camera valid range, and conversion QC thresholds.

Reason: The formal XTherm migration moved verified format values out of the
legacy configuration.

Status: Active.

Impact files or stage: XTherm conversion and conversion QC.

### Decision 2

Decision: `configs/physical_calibration.yaml` is the formal source for spatial
calibration, frame rate, scan direction, optical information, and physical
process metadata.

Reason: Physical calibration and acquisition metadata are formal experiment
properties, not legacy pilot settings.

Status: Active.

Impact files or stage: Metadata, ROI evaluation, future feature extraction.

### Decision 3

Decision: Frame rate is 52 fps.

Reason: User-confirmed experimental setting.

Status: Active.

Impact files or stage: Time-domain features and metadata.

### Decision 4

Decision: Formal effective frames are `frames[1:]`.

Reason: The first frame is excluded from formal analysis.

Status: Active.

Impact files or stage: ROI evaluation and future feature extraction.

### Decision 5

Decision: Frame 1 is the startup frame.

Reason: User-confirmed effective-frame rule.

Status: Active.

Impact files or stage: Frame selection.

### Decision 6

Decision: Formal camera valid range is 300-1800 deg C.

Reason: User-confirmed camera quantitative measurement range.

Status: Active.

Impact files or stage: Masking, reporting, and feature definitions.

### Decision 7

Decision: 1800-6500 deg C is `above_range`.

Reason: Values above the camera valid maximum are not quantitative
temperatures.

Status: Active.

Impact files or stage: Temperature-state reporting.

### Decision 8

Decision: `>=6500` deg C is `hard_saturation`.

Reason: This range represents hard saturation near the uint16 ceiling.

Status: Active.

Impact files or stage: Temperature-state reporting.

### Decision 9

Decision: Original matrices must be preserved.

Reason: Raw and processed temperature data must remain auditable and unmodified.

Status: Active.

Impact files or stage: All processing stages.

### Decision 10

Decision: Invalid pixels must not be interpolated.

Reason: Current policy is mask-and-report only.

Status: Active.

Impact files or stage: ROI evaluation and future feature extraction.

### Decision 11

Decision: Spatial calibration is 150.2 px = 5 mm.

Reason: User-confirmed formal spatial scale.

Status: Active.

Impact files or stage: ROI geometry and physical feature units.

### Decision 12

Decision: X direction currently uses the isotropic assumption.

Reason: Y scale is measured; independent X/Y anisotropy has not been verified.

Status: Active.

Impact files or stage: Spatial measurements.

### Decision 13

Decision: Fixed global ROI is used to preserve absolute position and trajectory.

Reason: Full-position information is needed for offsets and track stability.

Status: Evaluated, not yet applied.

Impact files or stage: Future ROI strategy configuration.

### Decision 14

Decision: Moving tracking window is used for local temperature, morphology,
gradient, and asymmetry analysis.

Reason: The moving window isolates local melt-pool morphology while preserving
centroid and absolute-position metadata separately.

Status: Evaluated, not yet applied.

Impact files or stage: Future ROI strategy configuration and feature extraction.

### Decision 15

Decision: The accepted evaluation recommendation is global ROI plus a 256 x 216
tracking window.

Reason: The evaluated tracking window covers the 700 deg C envelope and 800 deg C
core with zero clipped frames.

Status: Evaluated, not yet applied.

Impact files or stage: Future ROI strategy configuration.

### Decision 16

Decision: ROI generation is currently closed.

Reason: The formal pipeline waits for an explicit activation step.

Status: Active gate.

Impact files or stage: Formal ROI outputs.

### Decision 17

Decision: Formal feature extraction is currently closed.

Reason: ROI strategy must be machine-readable and approved before formal feature
generation.

Status: Active gate.

Impact files or stage: Formal feature outputs and response-surface analysis.

### Decision 18

Decision: `scripts/02d_conversion_report.py` is not a version-controlled formal
pipeline component.

Reason: It is a protected local untracked file and must remain untouched.

Status: Active.

Impact files or stage: Documentation and conversion-report status.

### Decision 19

Decision: T1/T2/T3 are within-condition repeated tracks; later analysis must
aggregate mean, standard deviation, and CV.

Reason: The condition is the statistical aggregation unit.

Status: Active.

Impact files or stage: Feature aggregation and response-surface analysis.

### Decision 20

Decision: `configs/roi_strategy.yaml` is the machine-readable authoritative
record for the formal ROI strategy.

Reason: Future code must not infer formal ROI parameters from `results/`, local
JSON files, or prose documentation.

Status: accepted_but_not_activated.

Impact files or stage: Future ROI generation and feature extraction must read
from this config; current execution gates remain closed.

### Decision 21

Decision: `configs/thermal_feature_contract.yaml` is the machine-readable
contract for the first formal thermal-field feature set, and
`docs/formal_feature_dictionary.md` is the human-readable feature dictionary.

Reason: The formal feature definitions, units, ROI/window responsibilities,
invalid-pixel policy, QC-only fields, and condition aggregation rules must be
fixed before any extraction code reads the 57 matrices or writes feature tables.

Status: designed_not_executed.

Impact files or stage: Future formal feature computation must follow this
contract; current ROI generation and feature extraction gates remain closed.
