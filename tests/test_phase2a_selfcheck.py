"""
Minimal self-test for Phase 2A: WindowDataset + LSTMForecastModel.

Validates:
  1. build_windows shapes (1-D and 2-D input)
  2. WindowDataset with single experiment (1-D, 2-D)
  3. WindowDataset with multi-experiment data
  4. process_params (ndarray and dict forms)
  5. split_by_experiment
  6. LSTMForecastModel forward pass
  7. build_lstm_from_config
"""

import sys
import os
import numpy as np
import torch

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from datasets.window_dataset import build_windows, WindowDataset, split_by_experiment
from models.lstm import LSTMForecastModel, build_lstm_from_config, count_parameters

PASS, FAIL = 0, 0


def check(cond, msg):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {msg}")
    else:
        FAIL += 1
        print(f"  FAIL  {msg}")


# ---------------------------------------------------------------------------
# 1. build_windows
# ---------------------------------------------------------------------------
print("=" * 60)
print("1. build_windows")
print("=" * 60)

# 1a: 1-D input (thermal cycle)
data_1d = np.sin(np.linspace(0, 10 * np.pi, 500)).astype(np.float32)
X, y = build_windows(data_1d, input_len=30, pred_len=10, step=2)
check(X.shape == (231, 30, 1), f"X shape (1D input): {X.shape}")  # (500-40)//2 + 1 = 231
check(y.shape == (231, 10, 1), f"y shape (1D input): {y.shape}")

# 1b: 2-D input (feature sequence)
data_2d = np.random.randn(500, 3).astype(np.float32)
X2, y2 = build_windows(data_2d, input_len=30, pred_len=10, step=1)
check(X2.shape == (461, 30, 3), f"X shape (2D input): {X2.shape}")
check(y2.shape == (461, 10, 3), f"y shape (2D input): {y2.shape}")

# 1c: edge case — exact window
data_exact = np.random.randn(40, 2).astype(np.float32)
X3, y3 = build_windows(data_exact, input_len=30, pred_len=10, step=1)
check(len(X3) == 1, f"exact window: 1 sample, got {len(X3)}")

# 1d: too-short data raises
try:
    build_windows(np.random.randn(20, 1), input_len=30, pred_len=10)
    check(False, "too-short data should raise ValueError")
except ValueError:
    check(True, "too-short data raises ValueError")

# ---------------------------------------------------------------------------
# 2. WindowDataset — single experiment
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("2. WindowDataset — single experiment")
print("=" * 60)

# 2a: 1-D data
ds = WindowDataset(data_1d, input_len=30, pred_len=10, step=2)
x0, y0 = ds[0]
check(isinstance(ds, torch.utils.data.Dataset), "is PyTorch Dataset")
check(ds.feature_dim == 1, f"feature_dim=1, got {ds.feature_dim}")
check(x0.shape == (30, 1), f"X[0] shape: {x0.shape}")
check(y0.shape == (10, 1), f"y[0] shape: {y0.shape}")
check(x0.dtype == np.float32, f"dtype: {x0.dtype}")

# 2b: 2-D data
ds2 = WindowDataset(data_2d, input_len=30, pred_len=10, step=2)
check(ds2.feature_dim == 3, f"feature_dim=3, got {ds2.feature_dim}")
x1, y1 = ds2[0]
check(x1.shape == (30, 3), f"X[0] shape (2D): {x1.shape}")
check(y1.shape == (10, 3), f"y[0] shape (2D): {y1.shape}")

# ---------------------------------------------------------------------------
# 3. WindowDataset — process_params
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("3. WindowDataset — process_params")
print("=" * 60)

# 3a: ndarray params (laser_power=1500W, scan_speed=5mm/s)
params = np.array([1500.0, 5.0], dtype=np.float32)
ds_p = WindowDataset(data_1d, input_len=30, pred_len=10, step=2,
                     process_params=params)
check(ds_p.feature_dim == 3, f"feature_dim after params: {ds_p.feature_dim}")
x_p, y_p = ds_p[0]
check(x_p.shape == (30, 3), f"X shape with params: {x_p.shape}")
# Last 2 features should be the params
check(np.allclose(x_p[0, 1:], [1500.0, 5.0]), "params tiled correctly")
check(y_p.shape == (10, 1), f"y unchanged: {y_p.shape}")

# 3b: dict params (multi-experiment)
data_list = [np.random.randn(200, 2).astype(np.float32) for _ in range(4)]
pp_dict = {0: [100.0, 1.0], 1: [200.0, 2.0],
           2: [300.0, 3.0], 3: [400.0, 4.0]}
ds_dict = WindowDataset(data_list, input_len=30, pred_len=10, step=1,
                        process_params=pp_dict)
check(ds_dict.feature_dim == 4, f"feature_dim with dict params: {ds_dict.feature_dim}")
check(len(ds_dict) > 0, f"samples: {len(ds_dict)}")

# ---------------------------------------------------------------------------
# 4. split_by_experiment
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("4. split_by_experiment")
print("=" * 60)

ds_multi = WindowDataset(data_list, input_len=30, pred_len=10, step=1,
                         process_params=pp_dict)
train_ds, val_ds, test_ds = split_by_experiment(ds_multi, val_ratio=0.25, test_ratio=0.25,
                                                 seed=42)
check(len(train_ds._data_list) == 2, f"train experiments: {len(train_ds._data_list)}")
check(len(val_ds._data_list) == 1, f"val experiments: {len(val_ds._data_list)}")
check(len(test_ds._data_list) == 1, f"test experiments: {len(test_ds._data_list)}")

# Verify no sample overlap between splits
total_split = len(train_ds) + len(val_ds) + len(test_ds)
check(total_split == len(ds_multi), f"no sample loss: {total_split} == {len(ds_multi)}")

# Verify process_params carried through split correctly
check(train_ds._process_params is not None, "train has process_params")
check(val_ds._process_params is not None, "val has process_params")
check(test_ds._process_params is not None, "test has process_params")
check(train_ds.feature_dim == ds_multi.feature_dim,
      f"train feature_dim matches: {train_ds.feature_dim}")

# Verify no double-tiling: each split's _process_params should be (n_samples, D)
check(train_ds._process_params.shape == (len(train_ds), 2),
      f"train params shape: {train_ds._process_params.shape}")
check(val_ds._process_params.shape == (len(val_ds), 2),
      f"val params shape: {val_ds._process_params.shape}")

# ---------------------------------------------------------------------------
# 5. LSTMForecastModel forward pass
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("5. LSTMForecastModel forward pass")
print("=" * 60)

# 5a: basic forward
model = LSTMForecastModel(input_dim=1, hidden_dim=32, num_layers=2, pred_len=10)
batch = torch.randn(4, 30, 1)  # (batch=4, input_len=30, feature_dim=1)
out = model(batch)
check(out.shape == (4, 10), f"output shape: {out.shape}")
check(out.dtype == torch.float32, f"output dtype: {out.dtype}")

# 5b: with process_params concatenated (feature_dim=3)
model3 = LSTMForecastModel(input_dim=3, hidden_dim=32, num_layers=2, pred_len=10)
batch3 = torch.randn(4, 30, 3)
out3 = model3(batch3)
check(out3.shape == (4, 10), f"output shape (dim=3): {out3.shape}")

# 5c: bidirectional vs unidirectional
model_uni = LSTMForecastModel(input_dim=1, hidden_dim=32, num_layers=2,
                              pred_len=10, bidirectional=False)
out_uni = model_uni(batch)
check(out_uni.shape == (4, 10), f"unidirectional output: {out_uni.shape}")

n_uni = count_parameters(model_uni)
n_bi = count_parameters(model)
check(n_bi > n_uni, f"bidirectional has more params: {n_bi} > {n_uni}")

# 5d: grad flow
loss = out.mean()
loss.backward()
for name, p in model.named_parameters():
    if p.requires_grad and p.grad is None:
        check(False, f"no grad for {name}")
        break
else:
    check(True, "gradients flow through all parameters")

# ---------------------------------------------------------------------------
# 6. build_lstm_from_config
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("6. build_lstm_from_config")
print("=" * 60)

config = {
    "model": {
        "name": "lstm",
        "input_dim": 2,
        "hidden_dim": 48,
        "num_layers": 3,
        "dropout": 0.3,
        "lstm": {"bidirectional": False},
    },
    "dataset": {
        "predict_window": 15,
    },
}
m = build_lstm_from_config(config)
check(m.input_dim == 2, f"input_dim=2: {m.input_dim}")
check(m.hidden_dim == 48, f"hidden_dim=48: {m.hidden_dim}")
check(m.num_layers == 3, f"num_layers=3: {m.num_layers}")
check(m.pred_len == 15, f"pred_len=15: {m.pred_len}")
check(m.bidirectional == False, f"bidirectional=False: {m.bidirectional}")
n_params = count_parameters(m)
check(n_params > 0, f"parameter count: {n_params}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
total = PASS + FAIL
print(f"RESULTS: {PASS}/{total} passed, {FAIL}/{total} failed")
if FAIL == 0:
    print("All tests passed!")
else:
    print(f"{FAIL} test(s) FAILED!")
    sys.exit(1)
