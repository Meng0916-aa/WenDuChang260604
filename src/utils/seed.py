"""
Reproducibility utilities.

Set random seeds for Python, NumPy, and PyTorch to ensure
deterministic and reproducible results.
"""

import random
import os
import numpy as np


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """
    Set all random seeds for reproducibility.

    Args:
        seed: integer seed value.
        deterministic: if True, request deterministic cuDNN algorithms
                       (may be slower).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True
    except ImportError:
        pass
