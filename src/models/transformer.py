"""
Transformer encoder forecaster — SKELETON ONLY.

Not implemented yet. Only the LSTM baseline (src/models/lstm.py) is runnable.
The class guards against accidental use by raising NotImplementedError.

To enable later:
  1. Add positional encoding + nn.TransformerEncoder in __init__.
  2. Implement forward() -> (B, pred_len).
  3. Remove the guard and register in train.py::build_model.
"""

import torch.nn as nn


class TransformerForecastModel(nn.Module):
    """
    Planned: Transformer-encoder forecaster.

    Intended interface (matches LSTMForecastModel):
        input:  (batch, input_len, input_dim)
        output: (batch, pred_len)

    Config (config['model']['transformer']): nhead, dim_feedforward.
    """

    def __init__(self,
                 input_dim: int,
                 pred_len: int = 10,
                 d_model: int = 64,
                 nhead: int = 4,
                 num_layers: int = 2,
                 dim_feedforward: int = 256,
                 dropout: float = 0.2):
        super().__init__()
        raise NotImplementedError(
            "TransformerForecastModel is a skeleton. Only the LSTM baseline "
            "is implemented; implement the encoder before using this model."
        )

    def forward(self, x):
        raise NotImplementedError(
            "TransformerForecastModel.forward is not implemented.")
