"""Pytest configuration: make src/ importable for all tests."""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

print("NOTE: tests use small SIMULATED data for code validation only — "
      "results do NOT represent experimental conclusions.")
