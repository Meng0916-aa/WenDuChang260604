"""
Feature normalization for thermal-cycle prediction.

StandardNormalizer applies per-channel z-score normalization:

    z = (x - mean) / std

Statistics are computed per feature channel (the LAST axis of the input)
and MUST be fit on the training split only — never on validation or test.
The normalizer works on both numpy arrays and torch tensors, and on both
1-D sequences (treated as a single channel) and 2-D / N-D feature
sequences whose last axis is the channel dimension.

Because the prediction target is feature channel 0 (the primary thermal-
cycle curve), `transform_target` / `inverse_transform_target` apply the
channel-0 statistics so predictions can be mapped back to Celsius before
metrics are computed.

Statistics persist to / from .npz (default) or .json.
"""

import json

import numpy as np


def _is_torch(x):
    """True if x is a torch.Tensor, without importing torch unless needed."""
    return type(x).__module__.startswith("torch")


class StandardNormalizer:
    """
    Per-channel standard (z-score) normalizer.

    Attributes:
        mean_: (C,) float32 array of per-channel means.
        std_:  (C,) float32 array of per-channel std devs (floored by eps).
        fitted: whether fit() has been called.
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = float(eps)
        self.mean_ = None
        self.std_ = None
        self.fitted = False

    # ------------------------------------------------------------------ fit
    def fit(self, X) -> "StandardNormalizer":
        """
        Compute per-channel statistics from TRAINING data only.

        Args:
            X: numpy array. 1-D (N,) -> single channel; otherwise the last
               axis is the channel dimension, e.g. (N, input_len, C).

        Returns:
            self.
        """
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        n_channels = arr.shape[-1]
        flat = arr.reshape(-1, n_channels)  # (M, C)

        self.mean_ = flat.mean(axis=0).astype(np.float32)
        std = flat.std(axis=0).astype(np.float32)
        self.std_ = np.maximum(std, self.eps).astype(np.float32)
        self.fitted = True
        return self

    # ------------------------------------------------------------- transform
    def _check(self):
        if not self.fitted:
            raise RuntimeError("StandardNormalizer used before fit().")

    def transform(self, X):
        """Normalize all channels. Accepts numpy or torch; returns same type."""
        self._check()
        return self._apply(X, self.mean_, self.std_, invert=False)

    def inverse_transform(self, X):
        """Invert normalization on all channels."""
        self._check()
        return self._apply(X, self.mean_, self.std_, invert=True)

    def transform_target(self, y):
        """Normalize the target using channel-0 statistics."""
        self._check()
        return self._apply_scalar(y, float(self.mean_[0]), float(self.std_[0]),
                                  invert=False)

    def inverse_transform_target(self, y):
        """Invert normalization on the target using channel-0 statistics."""
        self._check()
        return self._apply_scalar(y, float(self.mean_[0]), float(self.std_[0]),
                                  invert=True)

    # ------------------------------------------------------------- internals
    def _apply(self, X, mean, std, invert):
        """Per-channel apply, broadcasting (C,) over the last axis."""
        if _is_torch(X):
            import torch
            m = torch.as_tensor(mean, dtype=X.dtype, device=X.device)
            s = torch.as_tensor(std, dtype=X.dtype, device=X.device)
            return X * s + m if invert else (X - m) / s
        arr = np.asarray(X, dtype=np.float32)
        squeeze = arr.ndim == 1
        if squeeze:
            arr = arr[:, np.newaxis]
        out = arr * std + mean if invert else (arr - mean) / std
        out = out.astype(np.float32)
        return out[:, 0] if squeeze else out

    def _apply_scalar(self, y, mean, std, invert):
        """Apply a single (mean, std) to every element (target channel)."""
        if _is_torch(y):
            return y * std + mean if invert else (y - mean) / std
        arr = np.asarray(y, dtype=np.float32)
        return (arr * std + mean if invert else (arr - mean) / std).astype(np.float32)

    # --------------------------------------------------------------- persist
    def save(self, path: str) -> None:
        """Save statistics to .npz (default) or .json by extension."""
        self._check()
        if path.endswith(".json"):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    {"mean": self.mean_.tolist(),
                     "std": self.std_.tolist(),
                     "eps": self.eps,
                     "method": "standard"},
                    f, indent=2,
                )
        else:
            np.savez(path, mean=self.mean_, std=self.std_,
                     eps=np.float32(self.eps))

    @classmethod
    def load(cls, path: str) -> "StandardNormalizer":
        """Load statistics from .npz or .json."""
        obj = cls()
        if path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
            obj.mean_ = np.asarray(d["mean"], dtype=np.float32)
            obj.std_ = np.asarray(d["std"], dtype=np.float32)
            obj.eps = float(d.get("eps", 1e-8))
        else:
            d = np.load(path)
            obj.mean_ = np.asarray(d["mean"], dtype=np.float32)
            obj.std_ = np.asarray(d["std"], dtype=np.float32)
            obj.eps = float(d["eps"]) if "eps" in d else 1e-8
        obj.fitted = True
        return obj


def build_normalizer(config: dict) -> StandardNormalizer:
    """
    Construct a normalizer from config['normalization'].

    Only the "standard" method is implemented; anything else raises.
    Returns None semantics are handled by callers via the `enabled` flag.
    """
    ncfg = config.get("normalization", {})
    method = str(ncfg.get("method", "standard")).lower()
    if method != "standard":
        raise NotImplementedError(
            f"normalization.method='{method}' not implemented; use 'standard'."
        )
    return StandardNormalizer(eps=float(ncfg.get("eps", 1e-8)))
