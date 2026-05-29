"""Chargement des prédictions et formatage pour l'UI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import load_config, resolve_path
from data.loaders import load_json, load_matches
from evaluation.metrics import bookmaker_probabilities_without_margin
from features.pipeline import build_final_vector
from models.train import load_artifact

OUTCOME_LABELS = {
    0: ("PSG", "Victoire PSG", "#004170"),
    1: ("Nul", "Match nul", "#64748b"),
    2: ("Arsenal", "Victoire Arsenal", "#EF0107"),
}


@dataclass(frozen=True)
class PredictionBundle:
    artifact: dict[str, Any]
    final_vector: pd.DataFrame
    feature_columns: list[str]
    ordered: np.ndarray
    predicted_class: int
    bookmaker: dict[str, float]
    config: dict[str, Any]
    artifact_path: Path


def load_prediction_bundle() -> PredictionBundle | None:
    config = load_config()
    artifact_path = resolve_path(config["model"]["artifact_dir"]) / "model.joblib"
    if not artifact_path.exists():
        return None

    artifact = load_artifact(artifact_path)
    raw = load_matches(config["data"]["raw_matches_path"])
    injuries = load_json(config["data"]["injuries_path"], default={})
    final_vector = build_final_vector(raw, artifact["config"], injuries)
    feature_columns = artifact["feature_columns"]
    X_final = final_vector[feature_columns]

    probabilities = artifact["model"].predict_proba(X_final)[0]
    by_class = dict(zip(artifact["model"].classes_, probabilities))
    ordered = np.array([by_class.get(label, 0.0) for label in [0, 1, 2]])
    predicted_class = int(np.argmax(ordered))
    odds = config["final_match"]["bookmaker_odds"]
    bookmaker = bookmaker_probabilities_without_margin(odds)

    return PredictionBundle(
        artifact=artifact,
        final_vector=final_vector,
        feature_columns=feature_columns,
        ordered=ordered,
        predicted_class=predicted_class,
        bookmaker=bookmaker,
        config=config,
        artifact_path=artifact_path,
    )


def confidence_info(probabilities: np.ndarray) -> tuple[str, str, str]:
    """Retourne (label, classe CSS, conseil court)."""
    top_two = np.sort(probabilities)[-2:]
    leader = float(top_two[-1])
    margin = float(top_two[-1] - top_two[-2])

    if leader >= 0.75:
        return (
            "Confiance élevée — à valider",
            "conf-high",
            "Probabilité leader > 75 % : vérifier la calibration et le contexte jour J.",
        )
    if margin < 0.08:
        return (
            "Match très ouvert",
            "conf-low",
            "Écart faible entre les deux issues principales — forte incertitude.",
        )
    if leader < 0.55:
        return ("Confiance modérée", "conf-med", "Aucun favori net selon le modèle calibré.")
    return ("Confiance moyenne", "conf-med", "Favori identifié avec marge raisonnable.")


def feature_rows(X_final: pd.DataFrame) -> list[dict[str, Any]]:
    labels = {
        "elo_diff": "Écart Elo",
        "form_diff": "Forme (5 matchs)",
        "attack_diff": "Attaque",
        "defense_diff": "Défense",
        "injuries_diff": "Blessures (vs adv.)",
        "ucl_experience_diff": "Expérience LDC",
        "bookmaker_prob_diff": "Signal cotes",
        "rest_days_diff": "Jours de repos",
        "xg_diff": "xG (proxy)",
        "possession_diff": "Possession",
        "h2h_win_rate_diff": "H2H victoires",
    }
    rows = []
    for key, label in labels.items():
        if key in X_final.columns:
            val = float(X_final[key].iloc[0])
            rows.append({"feature": label, "key": key, "value": val})
    return rows
