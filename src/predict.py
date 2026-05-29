from __future__ import annotations

import sys

import numpy as np

from config import load_config, resolve_path
from console import (
    configure_console,
    print_footer,
    print_header,
    print_kv,
    print_probability_table,
    print_section,
    print_status,
    print_table,
)
from data.loaders import load_json, load_matches
from evaluation.metrics import bookmaker_probabilities_without_margin
from features.pipeline import TARGET_LABELS, build_final_vector
from models.train import load_artifact


DISPLAY_LABELS = {
    0: "PSG gagne",
    1: "Match nul",
    2: "Arsenal gagne",
}


def pct(value: float) -> str:
    return f"{value * 100:6.2f}%"


def signed_pp(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value * 100:.2f} pp"


def confidence_label(probabilities: np.ndarray) -> str:
    top_two = np.sort(probabilities)[-2:]
    leader = float(top_two[-1])
    margin = float(top_two[-1] - top_two[-2])
    if leader >= 0.75:
        return "High but verify calibration"
    if margin < 0.08:
        return "Low - close market"
    if leader < 0.55:
        return "Moderate"
    return "Medium"


def final_value(frame, column: str, default: float = 0.0) -> float:
    if column not in frame.columns:
        return default
    return float(frame[column].iloc[0])


def main() -> None:
    configure_console()
    config = load_config()
    print_header("Prediction finale PSG vs Arsenal")

    artifact_path = resolve_path(config["model"]["artifact_dir"]) / "model.joblib"
    if not artifact_path.exists():
        print("Modele introuvable. Lance d'abord: python src/model.py")
        sys.exit(1)

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

    print_section("Match")
    print_table(
        ["Team", "Opponent", "Competition", "Venue"],
        [[config["final_match"]["team"], config["final_match"]["opponent"], config["final_match"]["competition"], X_final["Venue"].iloc[0]]],
    )

    print_section("Key feature checks")
    key_features = [
        "elo_diff",
        "form_diff",
        "attack_diff",
        "defense_diff",
        "injuries_diff",
        "ucl_experience_diff",
        "bookmaker_prob_diff",
        "rest_days_diff",
    ]
    print_table(
        ["Feature", "Value"],
        [[feature, f"{float(X_final[feature].iloc[0]):.4f}"] for feature in key_features if feature in X_final.columns],
    )

    print_section("Prediction calibree")
    print_kv("Resultat predit", DISPLAY_LABELS[predicted_class])
    print_probability_table([(DISPLAY_LABELS[class_id], ordered[class_id]) for class_id in [0, 1, 2]])

    print_section("Bookmakers sans marge")
    print_probability_table(
        [
            ("PSG", bookmaker["team"]),
            ("Nul", bookmaker["draw"]),
            ("Arsenal", bookmaker["opponent"]),
        ]
    )

    print_section("Modele vs marche")
    market_ordered = np.array([bookmaker["team"], bookmaker["draw"], bookmaker["opponent"]])
    comparison_rows = []
    for class_id, market_prob in zip([0, 1, 2], market_ordered):
        model_prob = ordered[class_id]
        comparison_rows.append(
            [
                DISPLAY_LABELS[class_id],
                pct(model_prob),
                pct(market_prob),
                signed_pp(model_prob - market_prob),
            ]
        )
    print_table(["Outcome", "Model", "Market", "Edge"], comparison_rows)

    print_section("Controle qualite")
    missing_signal_checks = {
        "xG data available": abs(final_value(X_final, "xg_diff")) > 0,
        "Possession data available": abs(final_value(X_final, "possession_diff")) > 0,
        "H2H history available": abs(final_value(X_final, "h2h_win_rate_diff")) > 0
        or abs(final_value(X_final, "h2h_draw_rate")) > 0,
    }
    print_status("Model artifact", artifact_path.exists(), str(artifact_path))
    print_status("Final venue neutral", str(X_final["Venue"].iloc[0]) == "Neutral")
    print_status("Feature alignment", list(X_final.columns) == feature_columns)
    print_status("Probabilities sum", abs(float(ordered.sum()) - 1.0) < 1e-6, f"{ordered.sum():.6f}")
    print_status("Confidence level", True, confidence_label(ordered))
    for label, ok in missing_signal_checks.items():
        print_status(label, ok, "neutral fallback used" if not ok else "")
    if "metrics" in artifact:
        selected = artifact["config"]["model"].get("selected_model", "unknown")
        metrics = artifact["metrics"].get(selected, {})
        if metrics:
            print_kv("Selected model", selected)
            print_kv("Holdout LogLoss", f"{metrics.get('log_loss', 0):.4f}")
            print_kv("Holdout Brier", f"{metrics.get('brier_score', 0):.4f}")

    print_section("Interpretation (lecture prudente)")
    max_prob = float(ordered.max())
    print_kv(
        "Confiance",
        "Probas calibrees (sigmoid CV sur train uniquement) — pas de re-fit sur le test.",
    )
    if max_prob >= 0.65:
        print_kv(
            "Surconfiance",
            "Classe leader > 65 % : valider sur cotes et contexte jour J avant decision.",
        )
    print_kv(
        "Variance finale UCL",
        "Match unique a fort alea ; ecarts modele/marche peuvent refleter donnees partielles (xG/H2H).",
    )
    print_kv(
        "Value bet",
        "Comparer uniquement apres calibration ; bookmakers = reference de marche.",
    )
    print_kv("Direction", DISPLAY_LABELS[predicted_class])

    print_kv("Classes", str({k: TARGET_LABELS[k] for k in [0, 1, 2]}))
    print_footer("Prediction terminee — indicative, pas conseil de pari")


if __name__ == "__main__":
    main()
