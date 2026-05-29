from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from config import resolve_path
from features.pipeline import build_dataset
from models.train import load_artifact, train_and_save


def test_train_and_save_writes_joblib_artifacts(sample_matches, tmp_config):
    result = build_dataset(sample_matches, tmp_config)
    artifact = train_and_save(
        result.dataset,
        result.feature_columns,
        result.categorical_columns,
        result.numeric_columns,
        tmp_config,
    )

    artifact_dir = resolve_path(tmp_config["model"]["artifact_dir"])
    model_path = artifact_dir / "model.joblib"
    metrics_path = artifact_dir / "metrics.joblib"
    manifest_path = artifact_dir / "feature_manifest.json"

    assert model_path.exists()
    assert metrics_path.exists()
    assert manifest_path.exists()
    assert artifact["model"] is not None
    assert "logistic_calibrated" in artifact["metrics"]


def test_loaded_artifact_predicts_valid_probabilities(sample_matches, tmp_config):
    result = build_dataset(sample_matches, tmp_config)
    train_and_save(
        result.dataset,
        result.feature_columns,
        result.categorical_columns,
        result.numeric_columns,
        tmp_config,
    )
    model_path = resolve_path(tmp_config["model"]["artifact_dir"]) / "model.joblib"
    loaded = load_artifact(model_path)

    X = result.dataset[loaded["feature_columns"]].iloc[:2]
    probs = loaded["model"].predict_proba(X)
    assert probs.shape == (2, 3)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_feature_manifest_matches_artifact(sample_matches, tmp_config):
    result = build_dataset(sample_matches, tmp_config)
    train_and_save(
        result.dataset,
        result.feature_columns,
        result.categorical_columns,
        result.numeric_columns,
        tmp_config,
    )
    manifest_path = resolve_path(tmp_config["model"]["artifact_dir"]) / "feature_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    loaded = load_artifact(resolve_path(tmp_config["model"]["artifact_dir"]) / "model.joblib")

    assert manifest["feature_columns"] == loaded["feature_columns"]
    assert manifest["selected_model"] == tmp_config["model"]["selected_model"]


def test_identity_features_disabled_by_default(sample_matches, config):
    result = build_dataset(sample_matches, config)
    assert "Team" not in result.feature_columns
    assert "Opponent" not in result.feature_columns
    assert "Stage" not in result.feature_columns


@pytest.mark.parametrize("name", ["logistic_calibrated", "elo_baseline"])
def test_holdout_metrics_present(sample_matches, tmp_config, name):
    result = build_dataset(sample_matches, tmp_config)
    artifact = train_and_save(
        result.dataset,
        result.feature_columns,
        result.categorical_columns,
        result.numeric_columns,
        tmp_config,
    )
    metrics = artifact["metrics"][name]
    assert "log_loss" in metrics
    assert metrics["log_loss"] >= 0.0
