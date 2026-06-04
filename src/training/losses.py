"""
Loss functions for thermal cycle prediction.

The baseline uses MSE as the main training loss. A small factory keeps
the choice configurable from YAML while staying trivial to extend later.
"""

import torch.nn as nn


def get_loss(name: str = "mse") -> nn.Module:
    """
    Return a loss module by name.

    Args:
        name: one of {"mse", "mae"} (case-insensitive).
              "mse" -> nn.MSELoss (main baseline loss)
              "mae" -> nn.L1Loss

    Returns:
        An instantiated torch loss module.

    Raises:
        ValueError: if the name is not recognised.
    """
    key = str(name).lower()
    if key in ("mse", "l2", "mseloss"):
        return nn.MSELoss()
    if key in ("mae", "l1", "l1loss"):
        return nn.L1Loss()
    raise ValueError(f"Unknown loss '{name}'. Supported: 'mse', 'mae'.")
