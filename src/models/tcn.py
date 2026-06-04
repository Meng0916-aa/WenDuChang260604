"""
Temporal Convolutional Network (TCN) — SKELETON ONLY.

This model is intentionally not implemented yet. Only the LSTM baseline
(src/models/lstm.py) is runnable in the current project state. The class
below documents the intended interface and guards against accidental use by
raising NotImplementedError in __init__.

To enable later:
  1. Implement the dilated causal-convolution stack in __init__.
  2. Implement forward() to map (B, input_len, input_dim) -> (B, pred_len).
  3. Remove the NotImplementedError guard.
  4. Register the name in src/training/train.py::build_model.
"""

import torch.nn as nn


class TCNForecastModel(nn.Module):
    """
    Planned: dilated causal TCN for multi-step thermal-cycle forecasting.

    Intended interface (matches LSTMForecastModel):
        input:  (batch, input_len, input_dim)
        output: (batch, pred_len)

    Config (config['model']['tcn']): kernel_size, num_channels (list).
    """

    def __init__(self,
                 input_dim: int,
                 pred_len: int = 10,
                 kernel_size: int = 3,
                 num_channels=(64, 64, 64, 64),
                 dropout: float = 0.2):
        super().__init__()
        raise NotImplementedError(
            "TCNForecastModel is a skeleton. Only the LSTM baseline is "
            "implemented; implement the TCN stack before using this model."
        )

    def forward(self, x):
        raise NotImplementedError("TCNForecastModel.forward is not implemented.")
