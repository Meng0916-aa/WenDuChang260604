"""
Tests for scripts/02b_convert_xtherm_binary_to_npy.py.

Uses tiny SYNTHETIC .xtherm files (56-byte header + 2x3 little-endian uint16
payload) written to a temp directory — no real data is touched.
"""

import os
import sys
import importlib.util

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

HEADER_BYTES = 56
WIDTH = 3
HEIGHT = 2
SCALE = 0.1


def _load_script_02b():
    path = os.path.join(_ROOT, "scripts", "02b_convert_xtherm_binary_to_npy.py")
    spec = importlib.util.spec_from_file_location("script_02b", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_m = _load_script_02b()


def _write_xtherm(path, raw_values):
    """Write header + HEIGHT x WIDTH little-endian uint16 payload."""
    payload = np.asarray(raw_values, dtype="<u2").reshape(HEIGHT, WIDTH)
    with open(path, "wb") as f:
        f.write(b"\x00" * HEADER_BYTES)
        f.write(payload.tobytes())


def _make_cfg(input_dir):
    return {
        "input_dir": str(input_dir),
        "sample_id": "test",
        "width": WIDTH,
        "height": HEIGHT,
        "header_bytes": HEADER_BYTES,
        "dtype": "uint16",
        "endian": "little",
        "scale_factor": SCALE,
    }


def _write_three_frames(tmp_path):
    # Created deliberately OUT of name order; raw value = frame tag.
    # Frame for 0001: raw 100..105 -> 10.0..10.5 C, 0002: 200..205, 0003: 300..305.
    for name, base in (("0003.xtherm", 300), ("0001.xtherm", 100),
                       ("0002.xtherm", 200)):
        _write_xtherm(tmp_path / name, np.arange(base, base + WIDTH * HEIGHT))


def test_output_shape(tmp_path):
    _write_three_frames(tmp_path)
    data, files = _m.convert_xtherm_dir(_make_cfg(tmp_path))
    assert data.shape == (3, HEIGHT, WIDTH)
    assert data.dtype == np.float32


def test_temperature_scaling(tmp_path):
    _write_three_frames(tmp_path)
    data, _ = _m.convert_xtherm_dir(_make_cfg(tmp_path))
    expected_first = (np.arange(100, 106, dtype=np.float32)
                      .reshape(HEIGHT, WIDTH) * np.float32(SCALE))
    np.testing.assert_allclose(data[0], expected_first, rtol=0, atol=1e-6)
    assert abs(float(data[0, 0, 0]) - 10.0) < 1e-6
    assert abs(float(data[2, 1, 2]) - 30.5) < 1e-6


def test_files_sorted_by_name(tmp_path):
    _write_three_frames(tmp_path)
    data, files = _m.convert_xtherm_dir(_make_cfg(tmp_path))
    names = [os.path.basename(f) for f in files]
    assert names == ["0001.xtherm", "0002.xtherm", "0003.xtherm"]
    # Stacking must follow that order: frame i carries raw base (i+1)*100.
    np.testing.assert_allclose(data[:, 0, 0],
                               np.array([10.0, 20.0, 30.0], dtype=np.float32))


def test_size_mismatch_raises_with_filename(tmp_path):
    _write_three_frames(tmp_path)
    bad = tmp_path / "0004.xtherm"
    with open(bad, "wb") as f:
        f.write(b"\x00" * (HEADER_BYTES + 5))   # truncated payload
    with pytest.raises(_m.XthermSizeError, match="0004.xtherm"):
        _m.convert_xtherm_dir(_make_cfg(tmp_path))


def test_recursive_subfolder_pickup(tmp_path):
    sub = tmp_path / "runA"
    sub.mkdir()
    _write_xtherm(sub / "0001.xtherm", np.arange(100, 106))
    data, files = _m.convert_xtherm_dir(_make_cfg(tmp_path))
    assert data.shape == (1, HEIGHT, WIDTH)
    assert os.path.basename(files[0]) == "0001.xtherm"
