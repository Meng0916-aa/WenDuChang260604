"""
Tests for script 02 N x H x W shape/axis-order validation.

SIMULATED/synthetic data only — for code validation, not experimental results.
"""

import os
import sys
import importlib.util

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))


def _load_script_02():
    """Load scripts/02_convert_exported_to_npy.py as a module (not a package)."""
    path = os.path.join(_ROOT, "scripts", "02_convert_exported_to_npy.py")
    spec = importlib.util.spec_from_file_location("script_02", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_m = _load_script_02()
_validate_shape = _m._validate_shape
ShapeContractError = _m.ShapeContractError


def test_valid_nhw_passes():
    data = np.zeros((10, 8, 6), dtype=np.float32)   # N=10, H=8, W=6
    out = _validate_shape(data, "ok.npy")
    assert out.shape == (10, 8, 6)


def test_single_frame_2d_expanded():
    data = np.zeros((8, 6), dtype=np.float32)        # single H x W frame
    out = _validate_shape(data, "single.npy", min_frames=2)
    assert out.shape == (1, 8, 6)                     # exempt from min_frames


def test_single_frame_n1_allowed():
    data = np.zeros((1, 8, 6), dtype=np.float32)      # already (1, H, W)
    out = _validate_shape(data, "single3d.npy", min_frames=2)
    assert out.shape == (1, 8, 6)


def test_too_few_frames_rejected():
    # N=1 is allowed, but an N below min_frames with N>1... use min_frames=5.
    data = np.zeros((3, 8, 6), dtype=np.float32)
    try:
        _validate_shape(data, "few.npy", min_frames=5)
        raise AssertionError("expected ShapeContractError for N < min_frames")
    except ShapeContractError as e:
        assert "got shape=" in str(e)


def test_hwn_axis_order_rejected():
    # Real frame is H=480, W=640, N=8 -> correct N x H x W = (8, 480, 640).
    # Software exported H x W x N = (480, 640, 8); validate sees N=480, H=640,
    # W=8. Pinning the true H/W catches the wrong axis order.
    transposed = np.zeros((480, 640, 8), dtype=np.float32)
    try:
        _validate_shape(transposed, "hwn.npy",
                        expected_height=480, expected_width=640)
        raise AssertionError("expected height/width mismatch error")
    except ShapeContractError as e:
        msg = str(e)
        assert "mismatch" in msg and "got shape=" in msg


def test_expected_height_width_match():
    data = np.zeros((10, 64, 48), dtype=np.float32)
    out = _validate_shape(data, "hw.npy", expected_height=64, expected_width=48)
    assert out.shape == (10, 64, 48)


def test_expected_height_mismatch():
    data = np.zeros((10, 64, 48), dtype=np.float32)
    try:
        _validate_shape(data, "hbad.npy", expected_height=100)
        raise AssertionError("expected height mismatch error")
    except ShapeContractError as e:
        assert "height mismatch" in str(e)


def test_expected_width_mismatch():
    data = np.zeros((10, 64, 48), dtype=np.float32)
    try:
        _validate_shape(data, "wbad.npy", expected_width=99)
        raise AssertionError("expected width mismatch error")
    except ShapeContractError as e:
        assert "width mismatch" in str(e)


def test_degenerate_spatial_rejected():
    data = np.zeros((50, 1, 1), dtype=np.float32)     # H=W=1 -> wrong axis order
    try:
        _validate_shape(data, "degen.npy")
        raise AssertionError("expected error for degenerate spatial dims")
    except ShapeContractError as e:
        assert "got shape=" in str(e)


def test_wrong_ndim_rejected():
    data = np.zeros((4, 5, 6, 7), dtype=np.float32)   # 4-D
    try:
        _validate_shape(data, "four.npy")
        raise AssertionError("expected error for 4-D input")
    except ShapeContractError as e:
        assert "N x H x W" in str(e)


def test_nonzero_frame_axis_rejected():
    data = np.zeros((10, 8, 6), dtype=np.float32)
    try:
        _validate_shape(data, "axis.npy", expected_frame_axis=2)
        raise AssertionError("expected error for non-zero frame axis")
    except ShapeContractError as e:
        assert "axis 0" in str(e)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("test_convert_shape_validation: OK (synthetic/code-validation only)")
