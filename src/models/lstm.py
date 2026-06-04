"""
Basic LSTM forecast model for thermal cycle prediction.

Input:  (batch, input_len, feature_dim)
Output: (batch, pred_len)
"""

import torch
import torch.nn as nn


class LSTMForecastModel(nn.Module):
    """
    Multi-layer LSTM for multi-step time-series forecasting.

    Architecture:
        LSTM(input_dim -> hidden_dim, num_layers, optional bidirectional)
        → LayerNorm
        → Linear(hidden_dim * D, pred_len)

    where D = 2 if bidirectional else 1.

    Args:
        input_dim: number of input features per time step.
        hidden_dim: LSTM hidden state size.
        num_layers: number of stacked LSTM layers.
        pred_len: number of future time steps to predict.
        dropout: dropout rate applied between LSTM layers.
        bidirectional: if True, use bidirectional LSTM.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 pred_len: int = 10,
                 dropout: float = 0.2,
                 bidirectional: bool = True):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pred_len = pred_len
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.norm = nn.LayerNorm(hidden_dim * self.num_directions)
        self.dropout = nn.Dropout(dropout)

        self.head = nn.Linear(hidden_dim * self.num_directions, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_len, input_dim)

        Returns:
            y: (batch, pred_len)
        """
        # LSTM: output (batch, seq_len, hidden_dim * num_directions)
        lstm_out, _ = self.lstm(x)

        # Take the last time step's output
        last_out = lstm_out[:, -1, :]  # (batch, hidden_dim * D)

        out = self.norm(last_out)
        out = self.dropout(out)
        out = self.head(out)  # (batch, pred_len)

        return out


def build_lstm_from_config(config: dict) -> LSTMForecastModel:
    """
    Build an LSTMForecastModel from a configuration dictionary.

    Expected keys under config['model']:
        input_dim, hidden_dim, num_layers, dropout, pred_len (from dataset)
        and lstm.bidirectional.

    Args:
        config: full project config dict.

    Returns:
        LSTMForecastModel instance.
    """
    m = config["model"]
    d = config["dataset"]

    lstm_cfg = m.get("lstm", {})

    return LSTMForecastModel(
        input_dim=m.get("input_dim", 1),
        hidden_dim=m.get("hidden_dim", 64),
        num_layers=m.get("num_layers", 2),
        pred_len=d.get("pred_len", d.get("predict_window", 10)),
        dropout=m.get("dropout", 0.2),
        bidirectional=lstm_cfg.get("bidirectional", True),
    )


def count_parameters(model: nn.Module) -> int:
    """Return the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
