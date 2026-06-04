"""
06_train_model.py

Train the LSTM baseline on the windowed samples produced by
05_build_window_dataset.py. All parameters come from configs/default.yaml.

Outputs:
    results/logs/training_log.csv
    results/checkpoints/best_lstm.pt

Usage:
    python scripts/06_train_model.py [--config configs/default.yaml]
"""

import os
import sys
import argparse

import numpy as np
from torch.utils.data import Dataset

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from utils.config import load_config
from training.train import train_model


class ArrayWindowDataset(Dataset):
    """
    Minimal Dataset over pre-windowed arrays loaded from the .npz bundle.

    Exposes `.feature_dim` so train_model can size the model dynamically,
    matching the WindowDataset contract: X (input_len, feature_dim),
    y (pred_len, feature_dim).
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)
        self.feature_dim = int(self.X.shape[-1])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].copy(), self.y[idx].copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    bundle = os.path.join(
        config["paths"]["processed_samples"],
        config["dataset"].get("samples_file", "window_samples.npz"),
    )
    if not os.path.exists(bundle):
        raise FileNotFoundError(
            f"Samples bundle not found: {bundle}. "
            "Run scripts/05_build_window_dataset.py first."
        )

    data = np.load(bundle, allow_pickle=True)
    if bool(data["simulated"]):
        print("[06] NOTE: training on SIMULATED samples — code-chain test only.")

    train_ds = ArrayWindowDataset(data["X_train"], data["y_train"])
    val_ds = ArrayWindowDataset(data["X_val"], data["y_val"])

    summary = train_model(train_ds, val_ds, config)

    print(f"[06] best val_loss={summary['best_val_loss']:.6f} "
          f"@ epoch {summary['best_epoch']} ({summary['device']})")
    print(f"[06] checkpoint: {summary['checkpoint_path']}")
    print(f"[06] normalizer: {summary['normalizer_path']}")
    print(f"[06] log:        {summary['log_path']}")
    print(f"[06] used config: {summary['used_config_path']}")


if __name__ == "__main__":
    main()
