"""
Hybrid LSTM-TCN forecaster — SKELETON ONLY.

Not implemented yet. Only the LSTM baseline (src/models/lstm.py) is runnable.
The class guards against accidental use by raising NotImplementedError.

To enable later:
  1. Combine a TCN feature extractor with an LSTM head (or vice versa).
  2. Implement forward() -> (B, pred_len).
  3. Remove the guard and register in train.py::build_model.
"""

import torch.nn as nn


class LSTMTCNForecastModel(nn.Module):
    """
    Planned: hybrid LSTM + TCN forecaster.

    Intended interface (matches LSTMForecastModel):
        input:  (batch, input_len, input_dim)
        output: (batch, pred_len)
    """

    def __init__(self,
                 input_dim: int,
                 pred_len: int = 10,
                 hidden_dim: int = 64,
                 num_channels=(64, 64),
                 kernel_size: int = 3,
                 dropout: float = 0.2):
        super().__init__()
        raise NotImplementedError(
            "LSTMTCNForecastModel is a skeleton. Only the LSTM baseline is "
            "implemented; implement the hybrid before using this model."
        )

    def forward(self, x):
        raise NotImplementedError(
            "LSTMTCNForecastModel.forward is not implemented.")
