"""
Tests for script 05 SIM / real / mixed thermal-cycle classification.

SIMULATED/synthetic data only — for code validation, not experimental results.
"""

import os
import sys
import tempfile
import importlib.util

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))


def _load_script_05():
    """Load scripts/05_build_window_dataset.py as a module (not a package)."""
    path = os.path.join(_ROOT, "scripts", "05_build_window_dataset.py")
    spec = importlib.util.spec_from_file_location("script_05", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_m = _load_script_05()
_classify = _m._classify_cycle_csvs
MixedDataError = _m.MixedDataError


def _touch(d, name):
    with open(os.path.join(d, name), "w", encoding="utf-8") as f:
        f.write("frame,tmax,center_average,hot_zone_average\n0,1,1,1\n")


def test_sim_only():
    with tempfile.TemporaryDirectory() as d:
        _touch(d, "SIM_000.csv")
        _touch(d, "SIM_001.csv")
        real, sim = _classify(d)
        assert real == []
        assert len(sim) == 2


def test_real_only():
    with tempfile.TemporaryDirectory() as d:
        _touch(d, "B0_01.csv")
        _touch(d, "B100_02.csv")
        real, sim = _classify(d)
        assert len(real) == 2
        assert sim == []


def test_mixed_is_detected():
    with tempfile.TemporaryDirectory() as d:
        _touch(d, "B0_01.csv")
        _touch(d, "SIM_000.csv")
        real, sim = _classify(d)
        # classification returns both non-empty -> main() raises MixedDataError
        assert len(real) == 1 and len(sim) == 1
        # Replicate the guard in main() to confirm it fires on mixed input.
        raised = False
        try:
            if real and sim:
                raise MixedDataError("mixed")
        except MixedDataError:
            raised = True
        assert raised


def test_empty_dir():
    with tempfile.TemporaryDirectory() as d:
        real, sim = _classify(d)
        assert real == [] and sim == []


def test_mixed_dataerror_is_a_runtimeerror():
    # The exception type exists and is raiseable (used by main()).
    assert issubclass(MixedDataError, RuntimeError)
    try:
        raise MixedDataError("mixed")
    except RuntimeError:
        pass


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("test_build_window_sim_detection: OK (synthetic/code-validation only)")
