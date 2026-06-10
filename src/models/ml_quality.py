"""
Traditional machine-learning models for cladding quality assessment.

The project's main line under small-sample conditions: predict cladding
quality (classification) or geometric ratios (regression) from per-experiment
thermal-field features. Uses scikit-learn only — NO xgboost / lightgbm or other
extra dependencies.

Design notes:
  - Each model is a Pipeline(StandardScaler -> estimator) so feature scaling is
    fit inside every CV fold (no leakage).
  - Metrics are computed on out-of-fold predictions via cross_val_predict, which
    is the correct way to score LeaveOneOut (each sample is predicted exactly
    once; per-fold precision/recall on a single test point is undefined).
  - Random Forest feature importances are fit on the full feature matrix for
    interpretation (which thermal-field features matter most).
"""

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    LeaveOneOut, StratifiedKFold, cross_val_predict,
)

# Classifiers
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, Ridge

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score,
)


CLASSIFICATION_MODELS = ("rf", "svm", "knn", "logistic_regression")
REGRESSION_MODELS = ("rf", "svm", "knn", "ridge")


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(name: str, task: str, random_state: int = 42,
                n_neighbors: int = 3) -> Pipeline:
    """
    Build a Pipeline(StandardScaler -> estimator) for one model.

    Args:
        name: one of {"rf","svm","knn","logistic_regression","ridge"}.
        task: "classification" or "regression".
        random_state: seed for stochastic models.
        n_neighbors: k for KNN (kept small for small samples).

    Returns:
        An unfitted sklearn Pipeline.
    """
    name = name.lower()
    task = task.lower()

    if task == "classification":
        if name == "rf":
            est = RandomForestClassifier(n_estimators=200,
                                         random_state=random_state)
        elif name == "svm":
            est = SVC(kernel="rbf", probability=False,
                      random_state=random_state)
        elif name == "knn":
            est = KNeighborsClassifier(n_neighbors=n_neighbors)
        elif name == "logistic_regression":
            est = LogisticRegression(max_iter=1000,
                                     random_state=random_state)
        else:
            raise ValueError(
                f"Unknown classification model '{name}'. "
                f"Choose from {CLASSIFICATION_MODELS}.")
    elif task == "regression":
        if name == "rf":
            est = RandomForestRegressor(n_estimators=200,
                                        random_state=random_state)
        elif name == "svm":
            est = SVR(kernel="rbf")
        elif name == "knn":
            est = KNeighborsRegressor(n_neighbors=n_neighbors)
        elif name == "ridge":
            est = Ridge(random_state=random_state)
        else:
            raise ValueError(
                f"Unknown regression model '{name}'. "
                f"Choose from {REGRESSION_MODELS}.")
    else:
        raise ValueError(f"task must be 'classification' or 'regression', got {task!r}")

    return Pipeline([("scaler", StandardScaler()), ("model", est)])


# ---------------------------------------------------------------------------
# Cross-validation splitter
# ---------------------------------------------------------------------------

def get_cv_splitter(cv: str, task: str, y: np.ndarray, n_splits: int = 5):
    """
    Build a CV splitter. For small samples, leave_one_out is recommended.

    Args:
        cv: "leave_one_out" or "stratified_kfold".
        task: "classification" or "regression".
        y: label array (used to bound n_splits for stratified k-fold).
        n_splits: folds for stratified k-fold.

    Returns:
        An sklearn splitter instance.
    """
    cv = str(cv).lower()
    if cv in ("leave_one_out", "loo"):
        return LeaveOneOut()
    if cv in ("stratified_kfold", "skf", "kfold"):
        if task == "classification":
            # n_splits cannot exceed the smallest class count.
            _, counts = np.unique(y, return_counts=True)
            k = int(min(n_splits, counts.min())) if counts.size else n_splits
            k = max(2, k)
            return StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        from sklearn.model_selection import KFold
        k = max(2, min(n_splits, len(y)))
        return KFold(n_splits=k, shuffle=True, random_state=42)
    raise ValueError(f"Unknown cv '{cv}'. Use 'leave_one_out' or 'stratified_kfold'.")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(estimator, X: np.ndarray, y: np.ndarray, splitter,
                   task: str) -> dict:
    """
    Cross-validate one model and compute metrics on out-of-fold predictions.

    Returns a dict:
      classification: {accuracy, precision, recall, f1, confusion_matrix(list),
                       labels(list), y_pred(list)}
      regression:     {rmse, mae, r2, y_pred(list)}
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    y_pred = cross_val_predict(estimator, X, y, cv=splitter)

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

    # regression
    rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
    return {
        "rmse": rmse,
        "mae": float(mean_absolute_error(y, y_pred)),
        "r2": float(r2_score(y, y_pred)),
        "y_pred": [float(v) for v in y_pred],
    }


def compute_rf_feature_importance(X: np.ndarray, y: np.ndarray,
                                  feature_names: list, task: str,
                                  random_state: int = 42) -> list:
    """
    Fit a Random Forest on the full data and return feature importances.

    Returns a list of {feature, importance} dicts sorted by importance desc.
    """
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


def run_models(X, y, feature_names, task: str, model_names, cv: str,
               n_splits: int = 5, random_state: int = 42) -> dict:
    """
    Train + cross-validate a set of models and collect everything script 12
    needs.

    Returns:
        {
          "metrics": {model_name: metrics_dict, ...},
          "predictions": {model_name: [...], ...},
          "feature_importance": [{feature, importance}, ...],   # from RF
          "labels": [...]   # class labels (classification only)
        }
    """
    splitter = get_cv_splitter(cv, task, y, n_splits=n_splits)
    metrics, predictions = {}, {}
    labels = None
    for name in model_names:
        est = build_model(name, task, random_state=random_state)
        res = evaluate_model(est, X, y, splitter, task)
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
    }
