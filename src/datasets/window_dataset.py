"""
Sliding-window dataset for thermal cycle prediction.

Constructs (X, y) pairs from a multi-frame temperature sequence, where:
  X = past input_len frames  (input_len, feature_dim)
  y = next pred_len frames   (pred_len, feature_dim)

Supports per-experiment process parameters (laser_power, scan_speed, etc.)
which are tiled and concatenated as extra input features.

Train/val/test splits are done by experiment ID — adjacent frames from
the same experiment are never split across partitions.
"""

import numpy as np
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Standalone: build sliding windows (numpy)
# ---------------------------------------------------------------------------

def build_windows(data: np.ndarray,
                  input_len: int,
                  pred_len: int,
                  step: int = 1) -> tuple:
    """
    Build sliding-window samples from a 1-D or 2-D sequence.

    Args:
        data: (total_frames,) or (total_frames, feature_dim) in float32.
        input_len: number of input (past) frames.
        pred_len: number of output (future) frames.
        step: stride between consecutive windows.

    Returns:
        X: (num_samples, input_len, feature_dim)
        y: (num_samples, pred_len, feature_dim)
    """
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 1:
        data = data[:, np.newaxis]  # (N,) -> (N, 1)
    if data.ndim != 2:
        raise ValueError(f"Expected 1D or 2D data, got shape {data.shape}")

    total = data.shape[0]
    window_size = input_len + pred_len
    max_start = total - window_size

    if max_start < 0:
        raise ValueError(
            f"Data length {total} is too short for "
            f"input_len={input_len} + pred_len={pred_len}"
        )

    starts = np.arange(0, max_start + 1, step)
    n_samples = len(starts)
    feat_dim = data.shape[1]

    X = np.zeros((n_samples, input_len, feat_dim), dtype=np.float32)
    y = np.zeros((n_samples, pred_len, feat_dim), dtype=np.float32)

    for idx, s in enumerate(starts):
        X[idx] = data[s:s + input_len]
        y[idx] = data[s + input_len:s + window_size]

    return X, y


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class WindowDataset(Dataset):
    """
    PyTorch Dataset wrapping sliding-window samples.

    Args:
        data: (total_frames, feature_dim) float32 array, OR
              list of (N_i, feature_dim) arrays (multi-experiment).
        input_len: number of input (history) frames.
        pred_len: number of output (future) frames.
        step: stride between windows (default 1).
        process_params: optional. Can be:
            - (D,) array — same params appended to every sample
            - (N, D) array — per-sample params (used internally by split)
            - dict mapping experiment index -> (D,) array
              (only used when data is a list)
    """

    def __init__(self,
                 data,
                 input_len: int,
                 pred_len: int,
                 step: int = 1,
                 process_params=None):
        self.input_len = input_len
        self.pred_len = pred_len
        self.step = step

        # Normalize data to list of (N_i, D) arrays
        if isinstance(data, np.ndarray):
            data = np.asarray(data, dtype=np.float32)
            if data.ndim == 1:
                data = data[:, np.newaxis]
            self._data_list = [data]
        else:
            self._data_list = [
                np.asarray(d, dtype=np.float32) if d.ndim == 2
                else np.asarray(d, dtype=np.float32)[:, np.newaxis]
                for d in data
            ]

        # feature_dim is uniform across all experiments
        self.feature_dim = self._data_list[0].shape[1]

        # Build windows per experiment
        self.X_list = []
        self.y_list = []
        self._exp_boundaries = [0]  # cumulative sample counts

        for exp_data in self._data_list:
            X_i, y_i = build_windows(exp_data, input_len, pred_len, step)
            self.X_list.append(X_i)
            self.y_list.append(y_i)
            self._exp_boundaries.append(self._exp_boundaries[-1] + len(X_i))

        self._X = np.concatenate(self.X_list, axis=0) if self.X_list else np.array([])
        self._y = np.concatenate(self.y_list, axis=0) if self.y_list else np.array([])

        # Process params
        self._process_params = None
        if process_params is not None:
            self._process_params = self._build_process_params(process_params)
            # Tile params across input_len and concatenate as extra features
            if self._process_params is not None:
                param_dim = self._process_params.shape[-1]
                tiled = np.tile(self._process_params[:, np.newaxis, :],
                                (1, input_len, 1))
                self._X = np.concatenate([self._X, tiled], axis=-1)
                self.feature_dim += param_dim

    def _build_process_params(self, process_params):
        """
        Build per-sample process parameter array of shape (num_samples, D).

        Handles three forms:
          - (D,)  array  → tile to every sample
          - (N, D) array → already per-sample, use directly
          - dict[exp_idx -> (D,)] → build per-experiment, then concatenate
        """
        if isinstance(process_params, np.ndarray):
            pp = np.asarray(process_params, dtype=np.float32)
            if pp.ndim == 2 and pp.shape[0] == len(self):
                # Already per-sample (e.g. from split_by_experiment)
                return pp
            # 1-D: tile to all samples
            return np.tile(pp.ravel(), (len(self), 1))

        if isinstance(process_params, dict):
            all_params = []
            for exp_idx, X_i in enumerate(self.X_list):
                p = process_params.get(exp_idx)
                if p is None:
                    raise KeyError(
                        f"process_params missing for experiment index {exp_idx}"
                    )
                p = np.asarray(p, dtype=np.float32).ravel()
                all_params.append(np.tile(p, (len(X_i), 1)))
            return np.concatenate(all_params, axis=0)

        raise TypeError(
            f"process_params must be np.ndarray or dict, got {type(process_params)}"
        )

    def __len__(self) -> int:
        return len(self._X)

    def __getitem__(self, idx: int):
        return self._X[idx].copy(), self._y[idx].copy()

    def get_raw_data(self):
        """Return X, y as numpy arrays (for inspection)."""
        return self._X.copy(), self._y.copy()


# ---------------------------------------------------------------------------
# Train / val / test split by experiment
# ---------------------------------------------------------------------------

def split_by_experiment(dataset: WindowDataset,
                        val_ratio: float = 0.15,
                        test_ratio: float = 0.15,
                        seed: int = 42):
    """
    Split a multi-experiment WindowDataset into train / val / test
    by experiment index, NOT by shuffling individual samples.

    Args:
        dataset: a WindowDataset built from a list of per-experiment arrays.
        val_ratio: fraction of experiments for validation.
        test_ratio: fraction of experiments for testing.
        seed: random seed for reproducibility.

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    n_exp = len(dataset._data_list)
    if n_exp < 3:
        raise ValueError(
            f"Need at least 3 experiments for train/val/test split, got {n_exp}"
        )

    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_exp)

    n_val = max(1, int(n_exp * val_ratio))
    n_test = max(1, int(n_exp * test_ratio))
    n_train = n_exp - n_val - n_test
    if n_train < 1:
        raise ValueError(
            f"Not enough experiments: {n_exp} total, "
            f"val={n_val}, test={n_test}"
        )

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    def _make_subset(subset_indices):
        subset_data = [dataset._data_list[i] for i in subset_indices]
        # Carry over per-sample process params if present
        pp = None
        if dataset._process_params is not None:
            pp = np.concatenate([
                dataset._process_params[
                    dataset._exp_boundaries[i]:dataset._exp_boundaries[i + 1]
                ]
                for i in subset_indices
            ], axis=0)
        return WindowDataset(
            subset_data,
            input_len=dataset.input_len,
            pred_len=dataset.pred_len,
            step=dataset.step,
            process_params=pp,
        )

    return (
        _make_subset(train_idx),
        _make_subset(val_idx),
        _make_subset(test_idx),
    )


# ---------------------------------------------------------------------------
# Helpers for experiment-aware sample bookkeeping
# ---------------------------------------------------------------------------

def split_experiment_indices(n_exp: int,
                             val_ratio: float = 0.15,
                             test_ratio: float = 0.15,
                             seed: int = 42) -> tuple:
    """
    Return (train_idx, val_idx, test_idx) experiment-index arrays using the
    SAME shuffling logic as split_by_experiment. Lets callers (e.g. the build
    script) know which experiment landed in which split so they can emit
    per-sample experiment-id arrays.
    """
    if n_exp < 3:
        raise ValueError(f"Need at least 3 experiments, got {n_exp}")
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_exp)
    n_val = max(1, int(n_exp * val_ratio))
    n_test = max(1, int(n_exp * test_ratio))
    n_train = n_exp - n_val - n_test
    if n_train < 1:
        raise ValueError(f"Not enough experiments: {n_exp} total")
    return (indices[:n_train],
            indices[n_train:n_train + n_val],
            indices[n_train + n_val:])


def count_windows(n_frames: int, input_len: int, pred_len: int,
                  step: int = 1) -> int:
    """Number of sliding windows produced from a sequence of n_frames."""
    max_start = n_frames - (input_len + pred_len)
    if max_start < 0:
        return 0
    return len(range(0, max_start + 1, step))
