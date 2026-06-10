"""
Tests for the traditional ML quality models (src/models/ml_quality.py).

SIMULATED/synthetic data only — for code validation, not experimental results.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models.ml_quality import (
    build_model, get_cv_splitter, evaluate_model, run_models,
    compute_rf_feature_importance,
)
from sklearn.model_selection import LeaveOneOut


def _toy_classification(n_per_class=8, seed=0):
    """Two well-separated clusters -> easy 2-class problem."""
    rng = np.random.RandomState(seed)
    a = rng.normal(0.0, 0.3, size=(n_per_class, 3))
    b = rng.normal(3.0, 0.3, size=(n_per_class, 3))
    X = np.vstack([a, b]).astype(np.float64)
    y = np.array(["Bad"] * n_per_class + ["Good"] * n_per_class)
    names = ["f0", "f1", "f2"]
    return X, y, names


def _toy_regression(n=16, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.normal(0.0, 1.0, size=(n, 3))
    y = 2.0 * X[:, 0] - 1.0 * X[:, 1] + 0.5 + rng.normal(0, 0.05, n)
    return X.astype(np.float64), y.astype(np.float64), ["f0", "f1", "f2"]


def test_build_all_classification_models():
    for name in ("rf", "svm", "knn", "logistic_regression"):
        pipe = build_model(name, "classification")
        assert hasattr(pipe, "fit")
        # pipeline has a scaler + model
        assert "scaler" in dict(pipe.named_steps)


def test_build_all_regression_models():
    for name in ("rf", "svm", "knn", "ridge"):
        pipe = build_model(name, "regression")
        assert hasattr(pipe, "fit")


def test_leave_one_out_splitter():
    X, y, _ = _toy_classification()
    splitter = get_cv_splitter("leave_one_out", "classification", y)
    assert isinstance(splitter, LeaveOneOut)
    assert splitter.get_n_splits(X) == len(y)


def test_evaluate_classification_loo_runs():
    X, y, _ = _toy_classification()
    splitter = get_cv_splitter("leave_one_out", "classification", y)
    for name in ("rf", "svm", "knn", "logistic_regression"):
        est = build_model(name, "classification")
        res = evaluate_model(est, X, y, splitter, "classification")
        for k in ("accuracy", "precision", "recall", "f1", "confusion_matrix",
                  "labels", "y_pred"):
            assert k in res
        assert 0.0 <= res["accuracy"] <= 1.0
        # well-separated -> should classify perfectly
        assert res["accuracy"] > 0.9


def test_evaluate_regression_runs():
    X, y, _ = _toy_regression()
    splitter = get_cv_splitter("leave_one_out", "regression", y)
    est = build_model("rf", "regression")
    res = evaluate_model(est, X, y, splitter, "regression")
    for k in ("rmse", "mae", "r2", "y_pred"):
        assert k in res
    assert res["rmse"] >= 0.0


def test_feature_importance_sums_to_one():
    X, y, names = _toy_classification()
    imp = compute_rf_feature_importance(X, y, names, "classification")
    assert len(imp) == len(names)
    total = sum(d["importance"] for d in imp)
    assert np.isclose(total, 1.0, atol=1e-6)
    # sorted descending
    vals = [d["importance"] for d in imp]
    assert vals == sorted(vals, reverse=True)


def test_run_models_classification_bundle():
    X, y, names = _toy_classification()
    out = run_models(X, y, names, "classification",
                     ["rf", "svm", "knn", "logistic_regression"],
                     cv="leave_one_out")
    assert set(out["metrics"]) == {"rf", "svm", "knn", "logistic_regression"}
    assert set(out["predictions"]) == set(out["metrics"])
    assert out["labels"] == ["Bad", "Good"]
    assert len(out["feature_importance"]) == 3


def test_run_models_regression_bundle():
    X, y, names = _toy_regression()
    out = run_models(X, y, names, "regression",
                     ["rf", "svm", "knn", "ridge"], cv="leave_one_out")
    assert set(out["metrics"]) == {"rf", "svm", "knn", "ridge"}
    assert "rmse" in out["metrics"]["rf"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("test_ml_quality_model: OK (synthetic/code-validation only)")
