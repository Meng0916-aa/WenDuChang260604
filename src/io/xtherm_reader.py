"""
Reader for Xiris VXIR-3000 / WeldStudio Pro .xtherm files — INTERFACE ONLY.

The internal binary layout of .xtherm is NOT assumed or hand-decoded here.
Real parsing should be implemented later on top of the Xiris WeldSDK or an
official export interface. Until then these functions raise NotImplementedError
so callers cannot silently consume fabricated data.

Current supported path: export temperature matrices to .npy / .csv / .h5
(see src/io/export_loader.py) and run the pipeline from there.

Contract once implemented (must follow project conventions):
  - return float32 temperature in degrees Celsius
  - shape N x H x W (N frames, H height, W width)
  - raw digital counts -> Celsius via temperature = raw_value / 10.0
"""


def read_xtherm(filepath: str):
    """
    Read a single .xtherm file into a float32 Celsius array of shape (N, H, W).

    NOT IMPLEMENTED. Implement via Xiris WeldSDK / official export, then apply
    the raw->Celsius scale and N x H x W convention.
    """
    raise NotImplementedError(
        "xtherm parsing is not implemented. Export to .npy/.csv/.h5 and use "
        "src/io/export_loader.py, or implement this via the Xiris WeldSDK."
    )


def read_xtherm_metadata(filepath: str) -> dict:
    """
    Read acquisition metadata (frame rate, resolution, calibration, ...).

    NOT IMPLEMENTED — depends on the official .xtherm specification / SDK.
    """
    raise NotImplementedError(
        "xtherm metadata reading is not implemented (needs Xiris WeldSDK)."
    )
