"""
Group-aware traditional ML for SECTION-LEVEL cladding quality prediction.

The ML sample unit is a cross-section position. Multiple sections share an
``experiment_id``; those sections are highly correlated (same run, adjacent
frames), so they MUST stay together across CV folds. This module therefore
splits with GroupKFold / LeaveOneGroupOut on ``experiment_id`` and NEVER does a
random shuffle of sections — that would leak information and inflate scores.

Design (mirrors src/models/ml_quality.py, but group-aware):
  - Each model is a Pipeline(StandardScaler -> estimator); the scaler is fit
    inside every training fold only (no leakage).
  - Metrics are computed on out-of-fold predictions via cross_val_predict with
    the group splitter, which predicts each sample exactly once.
  - Model names follow the section_ml config:
        regression:     ridge | svr | random_forest | knn
        classification: logistic_regression | svm | random_forest | knn
    They are normalized to the names understood by ml_quality.build_model.

scikit-learn only — no extra dependencies.
"""

import numpy as np

from sklearn.model_selection import (
    GroupKFold, LeaveOneGroupOut, cross_val_predict,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score,
)

from models.ml_quality import build_model as _build_quality_model


REGRESSION_MODELS = ("ridge", "svr", "random_forest", "knn")
CLASSIFICATION_MODELS = ("logistic_regression", "svm", "random_forest", "knn")

# Map section_ml model names -> ml_quality.build_model names (per task).
_NAME_MAP = {
    "regression": {
        "ridge": "ridge",
        "svr": "svm",            # SVR
        "random_forest": "rf",
        "knn": "knn",
    },
    "classification": {
        "logistic_regression": "logistic_regression",
        "svm": "svm",            # SVC
        "random_forest": "rf",
        "knn": "knn",
    },
}


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_section_model(name: str, task: str, random_state: int = 42,
                        n_neighbors: int = 3):
    """Build a Pipeline(StandardScaler -> estimator) for a section_ml model.

    Args:
        name: section_ml model name (see REGRESSION_MODELS / CLASSIFICATION_MODELS).
        task: "regression" or "classification".
        random_state: seed for stochastic models.
        n_neighbors: k for KNN.

    Returns:
        An unfitted sklearn Pipeline.
    """
    task = task.lower()
    key = name.lower()
    if task not in _NAME_MAP:
        raise ValueError(f"task must be 'regression' or 'classification', got {task!r}")
    if key not in _NAME_MAP[task]:
        valid = REGRESSION_MODELS if task == "regression" else CLASSIFICATION_MODELS
        raise ValueError(f"Unknown {task} model '{name}'. Choose from {valid}.")
    return _build_quality_model(_NAME_MAP[task][key], task,
                                random_state=random_state, n_neighbors=n_neighbors)


# ---------------------------------------------------------------------------
# Group-aware CV splitter
# ---------------------------------------------------------------------------

def unique_group_count(groups) -> int:
    """Number of distinct experiment groups."""
    return int(len(np.unique(np.asarray(groups))))


def get_group_cv_splitter(cv: str, groups, n_splits: int = 5):
    """Build a GROUP-aware CV splitter on experiment groups.

    Args:
        cv: "group_kfold" or "leave_one_group_out".
        groups: per-sample group labels (experiment_id).
        n_splits: requested folds for GroupKFold (clamped to n_groups).

    Returns:
        (splitter, n_groups, cv_method_str).

    Raises:
        ValueError if fewer than 2 distinct groups exist (cannot hold out an
        experiment for testing).
    """
    n_groups = unique_group_count(groups)
    if n_groups < 2:
        raise ValueError(
            f"section-level CV needs >= 2 distinct experiment groups, got "
            f"{n_groups}. Add sections from more experiments before training.")

    cv = str(cv).lower()
    if cv in ("leave_one_group_out", "logo", "leave_one_experiment_out", "loeo"):
        return LeaveOneGroupOut(), n_groups, "leave_one_group_out"
    if cv in ("group_kfold", "groupkfold", "gkf"):
        k = max(2, min(int(n_splits), n_groups))
        return GroupKFold(n_splits=k), n_groups, f"group_kfold(k={k})"
    raise ValueError(
        f"Unknown cv '{cv}'. Use 'group_kfold' or 'leave_one_group_out'.")


def has_multiple_classes(y) -> bool:
    """True if the target has at least two distinct classes."""
    return len(np.unique(np.asarray(y))) >= 2


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_group_model(estimator, X, y, groups, splitter, task: str) -> dict:
    """Cross-validate one model with a group splitter; score out-of-fold preds.

    Returns:
        regression:     {mae, rmse, r2, y_pred(list)}
        classification: {accuracy, precision, recall, f1, confusion_matrix(list),
                         labels(list), y_pred(list)}
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    groups = np.asarray(groups)
    y_pred = cross_val_predict(estimator, X, y, groups=groups, cv=splitter)

    if task == "classification":
        labels = sorted(np.unique(y).tolist())
        cm = confusion_matrix(y, y_pred, labels=labels)
        return {
            "accuracy": float(accuracy_score(y, y_pred)),
            "precision": float(precision_score(y, y_pred, average="macro",
                                               zero_division=0)),
            "recall": float(recall_score(y, y_pred, average="macro",
                                         zero_division=0)),
            "f1": float(f1_score(y, y_pred, average="macro", zero_division=0)),
            "confusion_matrix": cm.tolist(),
            "labels": labels,
            "y_pred": y_pred.tolist(),
        }

    rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
    return {
        "mae": float(mean_absolute_error(y, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y, y_pred)),
        "y_pred": [float(v) for v in y_pred],
    }


def compute_rf_feature_importance(X, y, feature_names, task: str,
                                  random_state: int = 42) -> list:
    """Fit a Random Forest on the full data; return sorted feature importances."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    if task == "classification":
        rf = RandomForestClassifier(n_estimators=300, random_state=random_state)
    else:
        rf = RandomForestRegressor(n_estimators=300, random_state=random_state)
    rf.fit(X, y)
    pairs = sorted(zip(feature_names, rf.feature_importances_),
                   key=lambda t: t[1], reverse=True)
    return [{"feature": f, "importance": float(imp)} for f, imp in pairs]


def run_section_models(X, y, groups, feature_names, task: str, model_names,
                       cv: str, n_splits: int = 5, random_state: int = 42) -> dict:
    """Train + group-CV a set of section_ml models; collect everything 15 needs.

    Returns:
        {
          "metrics": {model: metrics_dict, ...},
          "predictions": {model: [...], ...},
          "feature_importance": [{feature, importance}, ...],  # RF only
          "labels": [...] | None,
          "n_groups": int,
          "cv_method": str,
        }

    Raises:
        ValueError if fewer than 2 experiment groups (via get_group_cv_splitter).
    """
    splitter, n_groups, cv_method = get_group_cv_splitter(cv, groups, n_splits)

    metrics, predictions = {}, {}
    labels = None
    for name in model_names:
        est = build_section_model(name, task, random_state=random_state)
        res = evaluate_group_model(est, X, y, groups, splitter, task)
        predictions[name] = res.pop("y_pred")
        if task == "classification":
            labels = res.get("labels", labels)
        metrics[name] = res

    importance = compute_rf_feature_importance(
        X, y, feature_names, task, random_state=random_state)

    return {
        "metrics": metrics,
        "predictions": predictions,
        "feature_importance": importance,
        "labels": labels,
        "n_groups": n_groups,
        "cv_method": cv_method,
    }
