"""
Portable data-root resolution for the formal 57-track pipeline.

The repository must NOT hard-code a machine-absolute raw-data path. The raw-data
root is resolved by priority:

  1. explicit CLI argument
  2. environment variable ``WENDUCHANG_DATA_ROOT``
  3. ``configs/local.yaml`` (git-ignored)  -> ``paths.raw_data_root``
  4. ``configs/experiments.yaml``           -> ``raw_data_root`` (relative or null)
  5. otherwise: a clear error (never a silent guess)

Windows and POSIX paths are both accepted. The legacy ``dataset`` pilot path is
never returned. Unit tests can call this with ``require_exists=False`` and/or an
explicit path so they do not depend on any real D-drive directory.
"""

import os
from collections import namedtuple

import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ENV_VAR = "WENDUCHANG_DATA_ROOT"

ResolvedDataRoot = namedtuple("ResolvedDataRoot", ["path", "source"])


class DataRootError(RuntimeError):
    """Raised when no valid raw-data root can be resolved."""


def normalize_path(path):
    """Normalize a Windows or POSIX path string (no existence check)."""
    p = os.path.expanduser(str(path).strip())
    p = p.replace("\\", "/")            # accept backslashes on any platform
    return os.path.normpath(p)


def _read_yaml(path):
    abs_path = path if os.path.isabs(path) else os.path.join(_ROOT, path)
    if not os.path.isfile(abs_path):
        return None
    with open(abs_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _local_value(local_config):
    cfg = _read_yaml(local_config)
    if not cfg:
        return None
    return (cfg.get("paths") or {}).get("raw_data_root")


def _experiments_value(experiments_config):
    cfg = _read_yaml(experiments_config)
    if not cfg:
        return None
    val = cfg.get("raw_data_root")
    if val is None:
        val = (cfg.get("paths") or {}).get("raw_data_root")
    return val


def _is_set(value):
    return value is not None and str(value).strip() != ""


def resolve_data_root(cli_arg=None, env=None, local_config="configs/local.yaml",
                      experiments_config="configs/experiments.yaml",
                      require_exists=True):
    """Resolve the raw-data root by the documented priority.

    Returns a ``ResolvedDataRoot(path, source)``. Raises ``DataRootError`` when
    nothing is configured, or (when ``require_exists``) when the resolved path is
    not an existing directory.
    """
    env = os.environ if env is None else env

    source, raw = None, None
    if _is_set(cli_arg):
        source, raw = "cli", cli_arg
    elif _is_set(env.get(ENV_VAR)):
        source, raw = f"env:{ENV_VAR}", env.get(ENV_VAR)
    elif _is_set(_local_value(local_config)):
        source, raw = "configs/local.yaml", _local_value(local_config)
    elif _is_set(_experiments_value(experiments_config)):
        source, raw = "configs/experiments.yaml", _experiments_value(experiments_config)

    if not _is_set(raw):
        raise DataRootError(
            "could not resolve the raw-data root. Provide one of (in priority "
            "order):\n"
            "  1. --raw-data-root <path> on the command line,\n"
            f"  2. the {ENV_VAR} environment variable,\n"
            "  3. configs/local.yaml -> paths.raw_data_root (copy "
            "configs/local.example.yaml; it is git-ignored),\n"
            "  4. configs/experiments.yaml -> raw_data_root (currently null).")

    path = normalize_path(raw)
    if require_exists and not os.path.isdir(path):
        raise DataRootError(
            f"raw-data root resolved from {source} does not exist as a directory: "
            f"{path}")
    return ResolvedDataRoot(path=path, source=source)
