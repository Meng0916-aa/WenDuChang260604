"""
Training loop for the LSTM thermal-cycle-prediction baseline.

Responsibilities:
  - device selection (CPU / CUDA auto)
  - fit a StandardNormalizer on the TRAINING split only and save it
  - per-epoch train / validation passes (in normalized space) with
    gradient clipping and early stopping
  - CSV training log   -> results/logs/training_log.csv
  - used config copy   -> results/logs/used_config.yaml
  - best checkpoint    -> results/checkpoints/best_lstm.pt
  - normalizer stats   -> results/checkpoints/normalizer.npz

Only the LSTM baseline is supported; any other model.name raises
NotImplementedError. Every function is callable in isolation so the loop
can be unit-tested without the pipeline scripts.
"""

import os
import csv

import yaml
import torch
from torch.utils.data import DataLoader

from models.lstm import LSTMForecastModel, count_parameters
from training.losses import get_loss
from preprocess.normalize import build_normalizer
from utils.seed import set_seed


# ---------------------------------------------------------------------------
# Device / target helpers
# ---------------------------------------------------------------------------

def select_device(config: dict) -> torch.device:
    """
    Resolve the compute device from config['training']['device'].

    "auto"/"cuda" -> cuda if available else cpu; "cpu" -> cpu.
    """
    requested = str(config.get("training", {}).get("device", "auto")).lower()
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_target(y: torch.Tensor) -> torch.Tensor:
    """
    Reduce a windowed target (B, pred_len, feature_dim) to the model's
    output shape (B, pred_len) by taking the FIRST feature channel
    (the primary thermal-cycle curve). Process-params are never targets.
    """
    if y.dim() == 3:
        return y[..., 0]
    return y


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 15, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.best_epoch = -1
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float, epoch: int) -> bool:
        """Update with the latest val loss; return True if it is a new best."""
        improved = val_loss < (self.best - self.min_delta)
        if improved:
            self.best = val_loss
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return improved


# ---------------------------------------------------------------------------
# Epoch passes (operate in NORMALIZED space when a normalizer is given)
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, loss_fn, optimizer, device,
                    grad_clip: float = 0.0, normalizer=None) -> float:
    """Run one training epoch; return mean batch loss (normalized space)."""
    model.train()
    total, n_batches = 0.0, 0
    for X, y in loader:
        X = X.to(device)
        target = make_target(y).to(device)
        if normalizer is not None:
            X = normalizer.transform(X)
            target = normalizer.transform_target(target)

        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, target)
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total += float(loss.item())
        n_batches += 1
    return total / max(1, n_batches)


@torch.no_grad()
def evaluate_loss(model, loader, loss_fn, device, normalizer=None) -> float:
    """Run one validation pass (no grad); return mean batch loss."""
    model.eval()
    total, n_batches = 0.0, 0
    for X, y in loader:
        X = X.to(device)
        target = make_target(y).to(device)
        if normalizer is not None:
            X = normalizer.transform(X)
            target = normalizer.transform_target(target)
        pred = model(X)
        total += float(loss_fn(pred, target).item())
        n_batches += 1
    return total / max(1, n_batches)


# ---------------------------------------------------------------------------
# Model construction (LSTM-only guard)
# ---------------------------------------------------------------------------

def build_model(config: dict, input_dim: int) -> LSTMForecastModel:
    """
    Build the LSTM baseline with input_dim determined dynamically from the
    dataset (curve features + any process params). Other model names raise.
    """
    name = str(config["model"].get("name", "lstm")).lower()
    if name != "lstm":
        raise NotImplementedError(
            f"Only the LSTM baseline is runnable, got model.name='{name}'. "
            "TCN / Transformer / LSTM-TCN are skeletons only."
        )
    m = config["model"]
    d = config["dataset"]
    lstm_cfg = m.get("lstm", {})
    return LSTMForecastModel(
        input_dim=input_dim,
        hidden_dim=m.get("hidden_dim", 64),
        num_layers=m.get("num_layers", 2),
        pred_len=d.get("predict_window", d.get("pred_len", 10)),
        dropout=m.get("dropout", 0.2),
        bidirectional=lstm_cfg.get("bidirectional", True),
    )


# ---------------------------------------------------------------------------
# Normalizer fitting
# ---------------------------------------------------------------------------

def _stack_dataset_X(dataset):
    """Collect all X windows from a dataset into one array for fitting."""
    if hasattr(dataset, "X"):
        return dataset.X
    xs = [dataset[i][0] for i in range(len(dataset))]
    import numpy as np
    return np.stack(xs, axis=0)


def fit_normalizer(train_ds, config):
    """
    Fit a StandardNormalizer on the TRAINING split only (or return None when
    normalization is disabled).
    """
    if not config.get("normalization", {}).get("enabled", True):
        return None
    normalizer = build_normalizer(config)
    normalizer.fit(_stack_dataset_X(train_ds))
    return normalizer


# ---------------------------------------------------------------------------
# Full training run
# ---------------------------------------------------------------------------

def train_model(train_ds, val_ds, config: dict) -> dict:
    """
    Train the LSTM baseline end-to-end.

    Saves training_log.csv, used_config.yaml, best_lstm.pt and (when
    normalization is enabled) normalizer.npz.

    Returns a summary dict.
    """
    tcfg = config["training"]
    paths = config["paths"]

    set_seed(config.get("seed", 42))
    device = select_device(config)

    input_dim = int(getattr(train_ds, "feature_dim"))
    model = build_model(config, input_dim).to(device)
    loss_fn = get_loss(tcfg.get("loss", "mse"))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(tcfg.get("learning_rate", 1e-3)),
        weight_decay=float(tcfg.get("weight_decay", 0.0)),
    )

    # Fit + save normalizer (train split only)
    normalizer = fit_normalizer(train_ds, config)
    os.makedirs(paths["results_checkpoints"], exist_ok=True)
    os.makedirs(paths["results_logs"], exist_ok=True)
    normalizer_path = os.path.join(paths["results_checkpoints"], "normalizer.npz")
    if normalizer is not None:
        normalizer.save(normalizer_path)

    train_loader = DataLoader(
        train_ds, batch_size=int(tcfg.get("batch_size", 32)),
        shuffle=bool(config["dataset"].get("shuffle_train", True)), drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=int(tcfg.get("batch_size", 32)),
        shuffle=False, drop_last=False,
    )

    log_path = os.path.join(paths["results_logs"], "training_log.csv")
    ckpt_path = os.path.join(paths["results_checkpoints"], "best_lstm.pt")
    used_config_path = os.path.join(paths["results_logs"], "used_config.yaml")

    # Persist the exact config used for this run
    with open(used_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    epochs = int(tcfg.get("epochs", 100))
    grad_clip = float(tcfg.get("gradient_clip", 0.0) or 0.0)
    stopper = EarlyStopping(patience=int(tcfg.get("patience", 15)))

    print(f"[train] device={device} input_dim={input_dim} "
          f"params={count_parameters(model)} "
          f"train={len(train_ds)} val={len(val_ds)} "
          f"normalize={'on' if normalizer is not None else 'off'}")

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "lr", "is_best"])

        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(
                model, train_loader, loss_fn, optimizer, device, grad_clip, normalizer
            )
            val_loss = evaluate_loss(model, val_loader, loss_fn, device, normalizer)
            lr = optimizer.param_groups[0]["lr"]

            is_best = stopper.step(val_loss, epoch)
            if is_best:
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "input_dim": input_dim,
                        "pred_len": model.pred_len,
                        "model_config": config["model"],
                        "feature_columns": config["dataset"].get(
                            "feature_columns", ["tmax"]),
                        "normalized": normalizer is not None,
                        "epoch": epoch,
                        "val_loss": val_loss,
                    },
                    ckpt_path,
                )

            writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
                             f"{lr:.6g}", int(is_best)])
            f.flush()
            if epoch == 1 or is_best or epoch % 10 == 0 or epoch == epochs:
                flag = " *" if is_best else ""
                print(f"[epoch {epoch:3d}/{epochs}] "
                      f"train={train_loss:.5f} val={val_loss:.5f}{flag}")

            if stopper.should_stop:
                print(f"[train] early stop @ epoch {epoch} "
                      f"(best epoch {stopper.best_epoch}, val={stopper.best:.5f})")
                break

    return {
        "best_val_loss": stopper.best,
        "best_epoch": stopper.best_epoch,
        "checkpoint_path": ckpt_path,
        "normalizer_path": normalizer_path if normalizer is not None else None,
        "log_path": log_path,
        "used_config_path": used_config_path,
        "input_dim": input_dim,
        "num_params": count_parameters(model),
        "device": str(device),
    }
