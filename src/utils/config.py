"""
Configuration loader.

Reads a YAML config file, resolves paths relative to the project root,
and provides dict-style access plus dot-notation for nested keys.
"""

import os
import yaml
from copy import deepcopy

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_project_root() -> str:
    """Return the absolute project root directory."""
    return _PROJECT_ROOT


def load_config(config_path: str) -> dict:
    """
    Load a YAML configuration file and resolve relative paths.

    Args:
        config_path: path to a .yaml config file (relative to project root or absolute).

    Returns:
        A nested dict with all values from the YAML file.
    """
    if not os.path.isabs(config_path):
        config_path = os.path.join(_PROJECT_ROOT, config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Resolve directory paths that are relative to the project root
    cfg = _resolve_paths(cfg, _PROJECT_ROOT)

    return cfg


def _resolve_paths(cfg: dict, root: str) -> dict:
    """Recursively resolve string values that look like relative paths under root."""
    path_keys: set[str] = set()
    _collect_path_keys(cfg, path_keys, prefix="")

    resolved = deepcopy(cfg)
    for key_path in path_keys:
        parts = key_path.split(".")
        node = resolved
        for p in parts[:-1]:
            node = node[p]
        val = node[parts[-1]]
        if isinstance(val, str) and not os.path.isabs(val):
            # Resolve relative paths under the 'paths' block
            node[parts[-1]] = os.path.normpath(os.path.join(root, val))

    return resolved


def _collect_path_keys(cfg: dict, keys: set, prefix: str) -> None:
    """Collect dotted-key paths for string values in the 'paths' block."""
    for k, v in cfg.items():
        full = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            _collect_path_keys(v, keys, full)


def merge_configs(base: dict, override: dict) -> dict:
    """
    Deep-merge two config dicts. Values from ``override`` win.

    Args:
        base: base configuration.
        override: override configuration.

    Returns:
        A new merged dict.
    """
    result = deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = merge_configs(result[k], v)
        else:
            result[k] = deepcopy(v)
    return result
