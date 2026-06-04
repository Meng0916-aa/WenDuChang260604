# Model Design

## Runnable scope

Only the **LSTM baseline** is implemented and trainable. The TCN, Transformer, and
LSTM-TCN models are guarded skeletons (`NotImplementedError` in `__init__`) and
`src/training/train.py::build_model` raises `NotImplementedError` for any `model.name`
other than `lstm`.

## LSTM baseline (`src/models/lstm.py`)

`LSTMForecastModel` performs multi-step forecasting of a thermal-cycle curve.

```
input:  (batch, input_len, input_dim)
        -> LSTM(input_dim -> hidden_dim, num_layers, optional bidirectional)
        -> take last time step
        -> LayerNorm -> Dropout
        -> Linear(hidden_dim * num_directions, pred_len)
output: (batch, pred_len)
```

### Input / output

- `input_len` = `dataset.input_window` (history length).
- `pred_len` = `dataset.predict_window` (forecast horizon).
- `input_dim` is determined **dynamically** at build time from the data:
  `input_dim = feature_dim = len(dataset.feature_columns) + (process params if enabled)`.
- Output is one scalar per future step (`pred_len` values). The predicted quantity is the
  **first** feature in `dataset.feature_columns` (the primary curve, e.g. `tmax`).
  Process parameters are auxiliary **inputs only**, never targets.

### Hyperparameters (from `configs/default.yaml`)

| Config key | Meaning |
|------------|---------|
| `model.hidden_dim` | LSTM hidden size |
| `model.num_layers` | stacked LSTM layers |
| `model.dropout` | dropout between layers / before head |
| `model.lstm.bidirectional` | bidirectional LSTM |
| `dataset.input_window` / `predict_window` | input / output length |

### Training

- Loss: **MSE** (`training.loss = mse`; MAE also available).
- Optimizer: Adam (`training.learning_rate`, `training.weight_decay`).
- Early stopping on validation loss (`training.patience`).
- Gradient clipping (`training.gradient_clip`).
- Device auto-selection (`training.device = auto` -> CUDA if available, else CPU).
- Random seed fixed via `src/utils/seed.py` (`seed`).

### Normalization

When `normalization.enabled = true` (default, method `standard`):

- A per-channel `StandardNormalizer` is fit on the **training split only** and saved to
  `results/checkpoints/normalizer.npz`.
- Training/validation run in normalized space.
- At evaluation, predictions are **inverse-transformed back to Celsius** before metrics are
  computed, so RMSE / MAE / waveform_similarity are all in physical units.

## Future models (skeletons)

`tcn.py`, `transformer.py`, `lstm_tcn.py` document their intended interface (same
`(B, input_len, input_dim) -> (B, pred_len)` contract). To enable one: implement the body,
remove the guard, and register the name in `train.py::build_model`.
