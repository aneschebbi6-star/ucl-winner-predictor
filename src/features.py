"""
══════════════════════════════════════════════════════════════════════
  📊 FEATURE ENGINEERING — PSG vs ARSENAL
  Transforme les données brutes en features prédictives
  
  Utilisation :
    1. Exécuter d'abord : python src/scrapper.py
    2. Puis : python src/features.py
══════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
import os
import json
from dotenv import load_dotenv
from datetime import datetime
import sys

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()

INPUT_DIR = os.getenv("OUTPUT_DIR", "data/raw")
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# FONCTIONS DE CALCUL DE FEATURES
# ══════════════════════════════════════════════════════════════

def compute_form(df: pd.DataFrame, window: int = 5) -> float:
    if df.empty or len(df) == 0:
        return 0.0
    recent = df.tail(window).copy()
    n = len(recent)
    weights = np.array([0.5 + 0.5 * (i / (n - 1)) if n > 1 else 1.0 for i in range(n)])
    weights = weights / weights.sum()
    points = recent["Resultat"].map({"W": 3, "D": 1, "L": 0}).fillna(0).values
    return round(np.dot(points, weights), 3)

def compute_rolling_stats(df: pd.DataFrame, window: int = 5) -> dict:
    if df.empty or len(df) < 1:
        return {f"avg_goals_scored_{window}": 0, f"avg_goals_conceded_{window}": 0}
    recent = df.tail(window)
    return {
        f"avg_goals_scored_{window}": round(recent["Buts_Pour"].mean(), 3),
        f"avg_goals_conceded_{window}": round(recent["Buts_Contre"].mean(), 3),
    }

def compute_competition_stats(df: pd.DataFrame, competition: str) -> dict:
    comp_df = df[df["Competition"].str.contains(competition, case=False, na=False)]
    if comp_df.empty:
        return {f"{competition.lower().replace(' ', '_')}_win_rate": 0}
    wins = (comp_df["Resultat"] == "W").sum()
    total = len(comp_df)
    prefix = competition.lower().replace(" ", "_")
    return {f"{prefix}_win_rate": round(wins / total, 3) if total > 0 else 0}

def compute_clean_sheets(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"clean_sheet_pct": 0}
    cs = (df["Buts_Contre"] == 0).sum()
    return {"clean_sheet_pct": round(cs / len(df), 3)}

def build_team_features(team_name: str, matches_df: pd.DataFrame, injuries_data: dict) -> dict:
    features = {"team": team_name}
    features["form_5"] = compute_form(matches_df, window=5)
    features["form_10"] = compute_form(matches_df, window=10)
    features.update(compute_rolling_stats(matches_df, window=5))
    
    total = len(matches_df)
    if total > 0:
        wins = (matches_df["Resultat"] == "W").sum()
        features["season_win_rate"] = round(wins / total, 3)
    
    features.update(compute_competition_stats(matches_df, "Champions League"))
    features.update(compute_clean_sheets(matches_df))
    
    # Intégration des blessures
    team_injuries = injuries_data.get(team_name, {})
    features["injury_impact_score"] = team_injuries.get("impact_score", 0)
    
    return features


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  📊 FEATURE ENGINEERING — PSG vs ARSENAL")
    print("=" * 70 + "\n")

    try:
        psg_df = pd.read_csv(os.path.join(INPUT_DIR, "psg_matches.csv"), parse_dates=["Date"])
        arsenal_df = pd.read_csv(os.path.join(INPUT_DIR, "arsenal_matches.csv"), parse_dates=["Date"])
    except FileNotFoundError:
        print("  ❌ Données brutes manquantes. Exécutez d'abord python src/scrapper.py")
        sys.exit(1)

    injuries_path = os.path.join(PROCESSED_DIR, "injuries_impact.json")
    if os.path.exists(injuries_path):
        with open(injuries_path, "r", encoding="utf-8") as f:
            injuries_data = json.load(f)
        print("  ✅ Données de blessures chargées.")
    else:
        injuries_data = {}
        print("  ⚠️  Aucune donnée de blessure trouvée.")

    psg_features = build_team_features("PSG", psg_df, injuries_data)
    arsenal_features = build_team_features("Arsenal", arsenal_df, injuries_data)

    print("\n  " + "═" * 50)
    print(f"  {'FEATURE':25s} {'PSG':>10s} {'ARSENAL':>10s}")
    print("  " + "═" * 50)
    
    for key in [k for k in psg_features.keys() if k != "team"]:
        p_val = f"{psg_features[key]:.3f}" if isinstance(psg_features[key], float) else str(psg_features[key])
        a_val = f"{arsenal_features[key]:.3f}" if isinstance(arsenal_features[key], float) else str(arsenal_features[key])
        print(f"  {key:25s} {p_val:>10s} {a_val:>10s}")
    print("  " + "═" * 50)

    features_json = {
        "PSG": psg_features,
        "Arsenal": arsenal_features,
        "generated_at": datetime.now().isoformat()
    }
    with open(os.path.join(PROCESSED_DIR, "team_features.json"), "w", encoding="utf-8") as f:
        json.dump(features_json, f, indent=2, ensure_ascii=False)
    print(f"\n  💾 Features sauvegardées dans {PROCESSED_DIR}/team_features.json\n")
