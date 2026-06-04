"""
Test raw -> Celsius conversion (temperature = raw_value / 10.0).

SIMULATED data only — for code validation, not experimental results.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import importlib.util as _ilu
_PATH = os.path.join(os.path.dirname(__file__), "..", "src", "io", "export_loader.py")
_spec = _ilu.spec_from_file_location("export_loader", _PATH)
_export_loader = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_export_loader)
raw_to_celsius = _export_loader.raw_to_celsius


def test_scale_default_is_divide_by_ten():
    raw = np.array([0, 10, 250, 12000], dtype=np.float32)
    out = raw_to_celsius(raw)            # default scale 0.1
    assert np.allclose(out, raw / 10.0)


def test_scale_dtype_is_float32():
    raw = np.array([100, 200], dtype=np.int32)
    out = raw_to_celsius(raw, scale=0.1)
    assert out.dtype == np.float32


def test_scale_custom_factor():
    raw = np.array([100.0], dtype=np.float32)
    assert np.isclose(raw_to_celsius(raw, scale=0.05)[0], 5.0)


if __name__ == "__main__":
    test_scale_default_is_divide_by_ten()
    test_scale_dtype_is_float32()
    test_scale_custom_factor()
    print("test_temperature_scale: OK (simulated/code-validation only)")
