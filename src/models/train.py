from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import resolve_path
from data.split import split_train_test
from evaluation.metrics import calibration_table, evaluate_probabilistic_model

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - optional dependency
    XGBClassifier = None


class EloBaseline:
    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.classes_ = np.array([0, 1, 2])
        self.draw_rate_ = float((y == 1).mean())
        self.draw_rate_ = min(max(self.draw_rate_, 0.12), 0.32)
        return self

    def predict_proba(self, X: pd.DataFrame):
        elo_diff = X["elo_diff"].astype(float).to_numpy()
        team_strength = 1 / (1 + 10 ** (-elo_diff / 400))
        non_draw = 1 - self.draw_rate_
        team_win = team_strength * non_draw
        team_loss = (1 - team_strength) * non_draw
        draw = np.full_like(team_win, self.draw_rate_, dtype=float)
        return np.column_stack([team_win, draw, team_loss])

    def predict(self, X: pd.DataFrame):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def make_preprocessor(categorical_columns: list[str], numeric_columns: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_columns),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_columns),
        ],
        remainder="drop",
    )


def make_logistic_pipeline(config: dict[str, Any], categorical_columns: list[str], numeric_columns: list[str]) -> Pipeline:
    params = config["model"].get("logistic", {})
    return Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(categorical_columns, numeric_columns)),
            (
                "classifier",
                LogisticRegression(
                    C=float(params.get("C", 0.5)),
                    max_iter=int(params.get("max_iter", 1000)),
                    class_weight="balanced",
                    random_state=int(config["project"].get("random_state", 42)),
                ),
            ),
        ]
    )


def make_xgb_pipeline(config: dict[str, Any], categorical_columns: list[str], numeric_columns: list[str]) -> Pipeline | None:
    if XGBClassifier is None:
        return None
    params = config["model"].get("xgboost", {})
    return Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(categorical_columns, numeric_columns)),
            (
                "classifier",
                XGBClassifier(
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    random_state=int(config["project"].get("random_state", 42)),
                    n_estimators=int(params.get("n_estimators", 50)),
                    max_depth=int(params.get("max_depth", 2)),
                    learning_rate=float(params.get("learning_rate", 0.03)),
                    subsample=float(params.get("subsample", 0.8)),
                    colsample_bytree=float(params.get("colsample_bytree", 0.8)),
                    reg_lambda=float(params.get("reg_lambda", 5.0)),
                    reg_alpha=float(params.get("reg_alpha", 1.0)),
                ),
            ),
        ]
    )


def calibrate_model(base_model: Pipeline, X: pd.DataFrame, y: pd.Series, cv_splits: int):
    cv_splits = max(2, min(cv_splits, int(y.value_counts().min())))
    try:
        model = CalibratedClassifierCV(estimator=base_model, method="sigmoid", cv=cv_splits)
    except TypeError:  # sklearn < 1.2
        model = CalibratedClassifierCV(base_estimator=base_model, method="sigmoid", cv=cv_splits)
    model.fit(X, y)
    return model


def train_and_save(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    numeric_columns: list[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    train_df, test_df = split_train_test(dataset, config["data"]["split_date"])
    X_train = train_df[feature_columns]
    y_train = train_df["y_target"]
    X_test = test_df[feature_columns]
    y_test = test_df["y_target"]

    tscv = TimeSeriesSplit(n_splits=min(int(config["model"].get("timeseries_cv_splits", 3)), max(2, len(train_df) // 10)))
    base_logistic = make_logistic_pipeline(config, categorical_columns, numeric_columns)
    calibrated_logistic = calibrate_model(
        base_logistic,
        X_train,
        y_train,
        int(config["model"].get("calibration_cv_splits", 3)),
    )

    candidates = {"logistic_calibrated": calibrated_logistic}
    candidates["elo_baseline"] = EloBaseline().fit(X_train, y_train)
    xgb = make_xgb_pipeline(config, categorical_columns, numeric_columns)
    if xgb is not None:
        xgb.fit(X_train, y_train)
        candidates["xgb_shallow"] = xgb

    metrics = {}
    for name, model in candidates.items():
        metrics[name] = evaluate_probabilistic_model(model, X_test, y_test)

    cv_scores = cross_val_score(base_logistic, X_train, y_train, cv=tscv, scoring="neg_log_loss")
    metrics["logistic_timeseries_cv"] = {
        "mean_log_loss": float(-cv_scores.mean()),
        "std_log_loss": float(cv_scores.std()),
    }

    selected_name = config["model"].get("selected_model", "logistic_calibrated")
    selected_model = candidates.get(selected_name, calibrated_logistic)

    artifact_dir = resolve_path(config["model"]["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    probabilities = selected_model.predict_proba(X_test)
    calibration = calibration_table(y_test, probabilities, selected_model.classes_)
    calibration.to_csv(artifact_dir / "calibration_table.csv", index=False)

    artifact = {
        "model": selected_model,
        "feature_columns": feature_columns,
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
        "config": config,
        "metrics": metrics,
        "classes": list(selected_model.classes_),
    }
    joblib.dump(artifact, artifact_dir / "model.joblib")
    joblib.dump(metrics, artifact_dir / "metrics.joblib")

    manifest = {
        "feature_columns": feature_columns,
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
        "selected_model": selected_name,
        "classes": [int(c) for c in artifact["classes"]],
        "split_date": str(config["data"]["split_date"]),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
    }
    (artifact_dir / "feature_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return artifact


def load_artifact(path: str | Path):
    p = Path(path)
    if not p.is_absolute():
        p = resolve_path(p)
    return joblib.load(p)
