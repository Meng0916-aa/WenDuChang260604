"""
Test LSTM forward and backward passes, and the LSTM-only build guard.

SIMULATED data only — for code validation, not experimental results.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models.lstm import LSTMForecastModel, count_parameters
from training.train import build_model


def test_forward_shape():
    model = LSTMForecastModel(input_dim=1, hidden_dim=16, num_layers=1, pred_len=10)
    out = model(torch.randn(4, 30, 1))
    assert out.shape == (4, 10)


def test_forward_multi_feature():
    model = LSTMForecastModel(input_dim=5, hidden_dim=16, num_layers=2, pred_len=8)
    out = model(torch.randn(3, 25, 5))
    assert out.shape == (3, 8)


def test_backward_grads_flow():
    model = LSTMForecastModel(input_dim=2, hidden_dim=16, num_layers=1, pred_len=6)
    out = model(torch.randn(4, 20, 2))
    out.mean().backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_build_model_lstm_ok():
    cfg = {"model": {"name": "lstm", "hidden_dim": 8, "num_layers": 1,
                     "dropout": 0.0, "lstm": {"bidirectional": False}},
           "dataset": {"predict_window": 5}}
    m = build_model(cfg, input_dim=3)
    assert count_parameters(m) > 0
    assert m(torch.randn(2, 12, 3)).shape == (2, 5)


def test_build_model_rejects_non_lstm():
    for bad in ("tcn", "transformer", "lstm_tcn"):
        cfg = {"model": {"name": bad}, "dataset": {"predict_window": 5}}
        try:
            build_model(cfg, input_dim=1)
            raise AssertionError(f"expected NotImplementedError for {bad}")
        except NotImplementedError:
            pass


if __name__ == "__main__":
    for fn in [test_forward_shape, test_forward_multi_feature,
               test_backward_grads_flow, test_build_model_lstm_ok,
               test_build_model_rejects_non_lstm]:
        fn()
    print("test_lstm_forward: OK (simulated/code-validation only)")
