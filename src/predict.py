import os
import sys

import pandas as pd

from console import configure_console, print_footer, print_header, print_kv, print_section

configure_console()

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError:
    print("XGBoost non installe. Installez-le avec `pip install xgboost`.")
    sys.exit(1)


RESULT_LABELS = {
    0: "Victoire PSG",
    1: "Match nul",
    2: "Victoire Arsenal",
}


def load_all_data():
    train_path = os.path.join("data", "processed", "train_dataset.csv")
    test_path = os.path.join("data", "processed", "test_dataset.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("Fichiers de donnees introuvables. Lancez src/dataset.py d'abord.")
        sys.exit(1)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return pd.concat([train_df, test_df], ignore_index=True)


def train_final_model(df):
    drop_cols = [
        "Date",
        "Home",
        "Away",
        "Home_Goals",
        "Away_Goals",
        "Buts_Pour",
        "Buts_Contre",
        "Resultat",
        "Comp_Code",
        "Group",
    ]

    X = df.drop(columns=drop_cols + ["y_target"], errors="ignore")
    y = df["y_target"]

    categorical_cols = ["Competition", "Team", "Opponent", "Venue", "Stage"]
    numeric_cols = ["Matchday"]

    categorical_cols = [c for c in categorical_cols if c in X.columns]
    numeric_cols = [c for c in numeric_cols if c in X.columns]

    X[categorical_cols] = X[categorical_cols].fillna("Missing").astype(str)
    X[numeric_cols] = X[numeric_cols].fillna(0)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ]
    )

    xgb_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                XGBClassifier(
                    use_label_encoder=False,
                    eval_metric="mlogloss",
                    random_state=42,
                    learning_rate=0.1,
                    max_depth=3,
                    n_estimators=50,
                ),
            ),
        ]
    )

    xgb_pipeline.fit(X, y)
    return xgb_pipeline


def get_probability(probability_by_class: dict, class_id: int) -> float:
    return probability_by_class.get(class_id, 0.0)


def main():
    print_header("Etape 6 : Simulation et inference (finale)")
    print()

    df = load_all_data()
    model = train_final_model(df)

    final_match = pd.DataFrame(
        [
            {
                "Competition": "Champions League",
                "Team": "Paris Saint-Germain FC",
                "Opponent": "Arsenal FC",
                "Venue": "Neutral",
                "Stage": "FINAL",
                "Matchday": 1,
            }
        ]
    )

    print_section("Feature vector - finale")
    for col in final_match.columns:
        print_kv(col, str(final_match[col].iloc[0]))

    probabilities = model.predict_proba(final_match)[0]
    probability_by_class = dict(zip(model.classes_, probabilities))

    psg_prob = get_probability(probability_by_class, 0)
    draw_prob = get_probability(probability_by_class, 1)
    arsenal_prob = get_probability(probability_by_class, 2)

    predicted_class = int(model.predict(final_match)[0])
    predicted_result = RESULT_LABELS.get(predicted_class, f"Classe {predicted_class}")

    print_section("Prediction XGBoost - resultat 1N2")
    print_kv("Resultat predit", predicted_result)
    print_kv("PSG gagne", f"{psg_prob * 100:.2f}%")
    print_kv("Match nul", f"{draw_prob * 100:.2f}%")
    print_kv("Arsenal gagne", f"{arsenal_prob * 100:.2f}%")

    cotes_bookmakers = {
        "PSG": 2.20,
        "Nul": 3.40,
        "Arsenal": 2.90,
    }

    implied_probs = {key: (1 / odd) * 100 for key, odd in cotes_bookmakers.items()}
    model_probs = {
        "PSG": psg_prob * 100,
        "Nul": draw_prob * 100,
        "Arsenal": arsenal_prob * 100,
    }

    print_section("Bookmakers vs modele")
    for key in ["PSG", "Nul", "Arsenal"]:
        print_kv(
            f"Cote {key}",
            f"{cotes_bookmakers[key]} (implicite {implied_probs[key]:.2f}%, modele {model_probs[key]:.2f}%)",
        )

    print_section("Analyse value bet")
    values = {
        key: model_probs[key] - implied_probs[key]
        for key in ["PSG", "Nul", "Arsenal"]
    }
    best_pick = max(values, key=values.get)

    if values[best_pick] > 0:
        print(f"  Value sur {best_pick} (+{values[best_pick]:.2f}% vs bookmakers)")
        print(f"  Recommandation : {best_pick}")
    else:
        print("  Pas de value bet claire - cotes alignees avec le modele")

    print_footer("Simulation terminee")


if __name__ == "__main__":
    main()
