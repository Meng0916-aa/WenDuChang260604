"""
Tests for group-aware section ML splitting / training (src/models/section_quality_ml.py).

The critical leakage guard: sections of the same experiment_id must NEVER appear
in both the train and test side of any fold.

Synthetic data only — for code validation, not experimental results.
"""

import os
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from models.section_quality_ml import (
    get_group_cv_splitter, unique_group_count, has_multiple_classes,
    build_section_model, run_section_models,
)


def _grouped_data(n_groups=4, per_group=5, seed=0):
    """Each experiment is one group of correlated sections."""
    rng = np.random.RandomState(seed)
    X, y, groups = [], [], []
    for g in range(n_groups):
        center = rng.uniform(0, 5, size=3)
        for _ in range(per_group):
            X.append(center + rng.normal(0, 0.1, size=3))
            y.append(float(center[0]))         # regression target
            groups.append(f"R{g:02d}")
    return (np.asarray(X, dtype=np.float64),
            np.asarray(y, dtype=np.float64),
            np.asarray(groups))


def test_unique_group_count():
    _, _, groups = _grouped_data()
    assert unique_group_count(groups) == 4


def test_group_kfold_no_experiment_leakage():
    X, y, groups = _grouped_data()
    splitter, n_groups, method = get_group_cv_splitter("group_kfold", groups, n_splits=4)
    assert n_groups == 4 and "group_kfold" in method
    seen_test_groups = set()
    for tr, te in splitter.split(X, y, groups):
        tr_groups = set(groups[tr])
        te_groups = set(groups[te])
        # No experiment appears in both train and test of a fold.
        assert tr_groups.isdisjoint(te_groups)
        seen_test_groups |= te_groups
    # Every experiment is tested exactly once across folds.
    assert seen_test_groups == set(groups)


def test_leave_one_group_out_no_leakage():
    X, y, groups = _grouped_data()
    splitter, n_groups, method = get_group_cv_splitter(
        "leave_one_group_out", groups, n_splits=5)
    assert method == "leave_one_group_out"
    folds = list(splitter.split(X, y, groups))
    assert len(folds) == n_groups
    for tr, te in folds:
        assert set(groups[tr]).isdisjoint(set(groups[te]))
        # exactly one experiment held out per fold
        assert len(set(groups[te])) == 1


def test_fewer_than_two_groups_rejected():
    groups = np.array(["R00"] * 6)
    with pytest.raises(ValueError, match=">= 2"):
        get_group_cv_splitter("group_kfold", groups, n_splits=5)


def test_has_multiple_classes():
    assert has_multiple_classes(np.array(["Good", "Bad", "Good"]))
    assert not has_multiple_classes(np.array(["Good", "Good"]))


def test_run_section_models_regression_bundle():
    X, y, groups = _grouped_data()
    out = run_section_models(X, y, groups, ["f0", "f1", "f2"], "regression",
                             ["ridge", "svr", "random_forest", "knn"],
                             cv="group_kfold", n_splits=4)
    assert set(out["metrics"]) == {"ridge", "svr", "random_forest", "knn"}
    assert "rmse" in out["metrics"]["ridge"]
    assert out["n_groups"] == 4
    assert len(out["feature_importance"]) == 3


def test_run_section_models_classification_bundle():
    X, yreg, groups = _grouped_data()
    # Make a 2-class target correlated with the first feature.
    y = np.where(yreg > np.median(yreg), "Good", "Bad")
    out = run_section_models(X, y, groups, ["f0", "f1", "f2"], "classification",
                             ["logistic_regression", "svm", "random_forest", "knn"],
                             cv="group_kfold", n_splits=4)
    assert set(out["metrics"]) == {"logistic_regression", "svm",
                                   "random_forest", "knn"}
    for m in out["metrics"].values():
        for k in ("accuracy", "precision", "recall", "f1", "confusion_matrix"):
            assert k in m
    assert out["labels"] == ["Bad", "Good"]


def test_build_section_model_name_mapping():
    # svr -> SVR (regression), svm -> SVC (classification)
    assert "scaler" in dict(build_section_model("svr", "regression").named_steps)
    assert "scaler" in dict(build_section_model("svm", "classification").named_steps)
    with pytest.raises(ValueError):
        build_section_model("svr", "classification")   # svr is regression-only


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("test_section_group_split: OK (synthetic/code-validation only)")
