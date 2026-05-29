from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


TARGET_LABELS = {
    0: "Team win",
    1: "Draw",
    2: "Team loss",
}

DIFF_FEATURES = [
    "elo_diff",
    "form_diff",
    "xg_diff",
    "attack_diff",
    "defense_diff",
    "injuries_diff",
    "possession_diff",
    "ucl_experience_diff",
    "bookmaker_prob_diff",
    "rest_days_diff",
    "h2h_win_rate_diff",
    "h2h_draw_rate",
]

BASE_NUMERIC_FEATURES = [
    "is_home",
    "is_away",
    "is_neutral",
    "team_form_5",
    "team_form_10",
    "team_attack_5",
    "team_defense_5",
    "team_win_rate_before",
    "team_clean_sheet_rate_before",
    "team_ucl_win_rate_before",
    "team_ucl_matches_before",
    "team_rest_days",
    "team_elo_pre",
    "opponent_form_5",
    "opponent_form_10",
    "opponent_attack_5",
    "opponent_defense_5",
    "opponent_win_rate_before",
    "opponent_clean_sheet_rate_before",
    "opponent_ucl_win_rate_before",
    "opponent_ucl_matches_before",
    "opponent_rest_days",
    "opponent_elo_pre",
] + DIFF_FEATURES


@dataclass(frozen=True)
class FeatureBuildResult:
    dataset: pd.DataFrame
    feature_columns: list[str]
    categorical_columns: list[str]
    numeric_columns: list[str]


def team_key(name: str) -> str:
    text = str(name).lower()
    if "paris" in text or "psg" in text:
        return "PSG"
    if "arsenal" in text:
        return "Arsenal"
    return str(name)


def _result_from_scores(goals_for, goals_against) -> float:
    if pd.isna(goals_for) or pd.isna(goals_against):
        return np.nan
    if goals_for > goals_against:
        return 0
    if goals_for == goals_against:
        return 1
    return 2


def make_fixture_events(raw_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "Date",
        "Competition",
        "Comp_Code",
        "Matchday",
        "Home",
        "Away",
        "Home_Goals",
        "Away_Goals",
        "Venue",
        "Stage",
        "Group",
    ]
    available = [c for c in cols if c in raw_df.columns]
    events = raw_df[available].drop_duplicates(
        subset=["Date", "Competition", "Home", "Away"], keep="first"
    ).copy()
    events["Date"] = pd.to_datetime(events["Date"])
    events = events.sort_values(["Date", "Home", "Away"]).reset_index(drop=True)
    events["event_id"] = np.arange(len(events))
    return events


def make_perspective_rows(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in events.to_dict("records"):
        for is_home in [True, False]:
            team = row["Home"] if is_home else row["Away"]
            opponent = row["Away"] if is_home else row["Home"]
            goals_for = row.get("Home_Goals") if is_home else row.get("Away_Goals")
            goals_against = row.get("Away_Goals") if is_home else row.get("Home_Goals")
            venue = row.get("Venue")
            if not venue or pd.isna(venue):
                venue = "Home" if is_home else "Away"
            elif venue == "Neutral":
                venue = "Neutral"
            target = _result_from_scores(goals_for, goals_against)
            rows.append(
                {
                    **row,
                    "Team": team,
                    "Opponent": opponent,
                    "Venue": venue,
                    "Buts_Pour": goals_for,
                    "Buts_Contre": goals_against,
                    "Resultat": np.nan if pd.isna(target) else {0: "W", 1: "D", 2: "L"}[int(target)],
                    "y_target": target,
                    "target_label": np.nan if pd.isna(target) else TARGET_LABELS[int(target)],
                }
            )
    return pd.DataFrame(rows).sort_values(["Date", "event_id", "Team"]).reset_index(drop=True)


def add_elo_features(events: pd.DataFrame, k_factor: float = 24.0, start_elo: float = 1500.0) -> pd.DataFrame:
    ratings: dict[str, float] = {}
    rows = []
    for row in events.sort_values("Date").to_dict("records"):
        home = row["Home"]
        away = row["Away"]
        home_elo = ratings.get(home, start_elo)
        away_elo = ratings.get(away, start_elo)
        rows.append({"event_id": row["event_id"], "Home": home, "Away": away, "home_elo_pre": home_elo, "away_elo_pre": away_elo})

        if pd.isna(row.get("Home_Goals")) or pd.isna(row.get("Away_Goals")):
            continue
        actual_home = 1.0 if row["Home_Goals"] > row["Away_Goals"] else 0.5 if row["Home_Goals"] == row["Away_Goals"] else 0.0
        expected_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
        ratings[home] = home_elo + k_factor * (actual_home - expected_home)
        ratings[away] = away_elo + k_factor * ((1 - actual_home) - (1 - expected_home))

    return pd.DataFrame(rows)


def add_h2h_features(events: pd.DataFrame) -> pd.DataFrame:
    history: dict[tuple[str, str], dict[str, float]] = {}
    rows = []
    for row in events.sort_values("Date").to_dict("records"):
        home = row["Home"]
        away = row["Away"]
        key = tuple(sorted([home, away]))
        stats = history.get(key, {"matches": 0, "draws": 0, "wins": {home: 0, away: 0}})
        matches = stats["matches"]
        home_wins = stats["wins"].get(home, 0)
        away_wins = stats["wins"].get(away, 0)
        rows.append(
            {
                "event_id": row["event_id"],
                "h2h_matches_prior": matches,
                "h2h_home_win_rate_prior": home_wins / matches if matches else 0.0,
                "h2h_away_win_rate_prior": away_wins / matches if matches else 0.0,
                "h2h_draw_rate_prior": stats["draws"] / matches if matches else 0.0,
            }
        )
        if pd.isna(row.get("Home_Goals")) or pd.isna(row.get("Away_Goals")):
            continue
        stats["matches"] += 1
        stats["wins"].setdefault(home, 0)
        stats["wins"].setdefault(away, 0)
        if row["Home_Goals"] > row["Away_Goals"]:
            stats["wins"][home] += 1
        elif row["Home_Goals"] < row["Away_Goals"]:
            stats["wins"][away] += 1
        else:
            stats["draws"] += 1
        history[key] = stats
    return pd.DataFrame(rows)


def add_shifted_team_features(perspective: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    df = perspective.sort_values(["Team", "Date", "event_id"]).copy()
    df["points"] = df["Resultat"].map({"W": 3, "D": 1, "L": 0})
    df["is_win"] = (df["Resultat"] == "W").astype(float)
    df["is_clean_sheet"] = (df["Buts_Contre"] == 0).astype(float)
    df["is_ucl"] = df["Competition"].str.contains("Champions League", case=False, na=False).astype(float)
    df["is_ucl_win"] = df["is_ucl"] * df["is_win"]

    grouped = df.groupby("Team", group_keys=False)
    for window in windows:
        df[f"team_form_{window}"] = grouped["points"].transform(lambda s: s.shift().rolling(window, min_periods=1).mean())
    df["team_attack_5"] = grouped["Buts_Pour"].transform(lambda s: s.shift().rolling(5, min_periods=1).mean())
    df["team_defense_5"] = grouped["Buts_Contre"].transform(lambda s: s.shift().rolling(5, min_periods=1).mean())
    df["team_win_rate_before"] = grouped["is_win"].transform(lambda s: s.shift().expanding(min_periods=1).mean())
    df["team_clean_sheet_rate_before"] = grouped["is_clean_sheet"].transform(lambda s: s.shift().expanding(min_periods=1).mean())
    df["team_ucl_matches_before"] = grouped["is_ucl"].transform(lambda s: s.shift().cumsum())
    df["team_ucl_win_rate_before"] = grouped["is_ucl_win"].transform(lambda s: s.shift().cumsum()) / df["team_ucl_matches_before"]
    df["team_rest_days"] = grouped["Date"].diff().dt.days

    defaults = {
        "team_form_5": 1.0,
        "team_form_10": 1.0,
        "team_attack_5": df["Buts_Pour"].mean(),
        "team_defense_5": df["Buts_Contre"].mean(),
        "team_win_rate_before": df["is_win"].mean(),
        "team_clean_sheet_rate_before": df["is_clean_sheet"].mean(),
        "team_ucl_matches_before": 0,
        "team_ucl_win_rate_before": df.loc[df["is_ucl"] == 1, "is_win"].mean(),
        "team_rest_days": df["Date"].diff().dt.days.median(),
    }
    df = df.replace([np.inf, -np.inf], np.nan).fillna(defaults)
    return df.drop(columns=["points", "is_win", "is_clean_sheet", "is_ucl", "is_ucl_win"])


def add_external_features(df: pd.DataFrame, injuries: dict[str, Any] | None = None) -> pd.DataFrame:
    injuries = injuries or {}
    out = df.copy()
    out["team_injuries"] = out["Team"].map(lambda name: float(injuries.get(team_key(name), {}).get("impact_score", 0) or 0))
    out["xg_for_before"] = 0.0
    out["xg_against_before"] = 0.0
    out["possession_before"] = 0.0
    out["bookmaker_prob"] = 0.0
    return out


def add_opponent_and_diff_features(df: pd.DataFrame) -> pd.DataFrame:
    base_cols = [
        "event_id",
        "Team",
        "team_form_5",
        "team_form_10",
        "team_attack_5",
        "team_defense_5",
        "team_win_rate_before",
        "team_clean_sheet_rate_before",
        "team_ucl_win_rate_before",
        "team_ucl_matches_before",
        "team_rest_days",
        "team_elo_pre",
        "team_injuries",
        "xg_for_before",
        "xg_against_before",
        "possession_before",
        "bookmaker_prob",
    ]
    opponent = df[base_cols].rename(
        columns={c: c.replace("team_", "opponent_") for c in base_cols if c.startswith("team_")}
        | {
            "Team": "Opponent",
            "xg_for_before": "opponent_xg_for_before",
            "xg_against_before": "opponent_xg_against_before",
            "possession_before": "opponent_possession_before",
            "bookmaker_prob": "opponent_bookmaker_prob",
        }
    )
    out = df.merge(opponent, on=["event_id", "Opponent"], how="left")

    out["elo_diff"] = out["team_elo_pre"] - out["opponent_elo_pre"]
    out["form_diff"] = out["team_form_5"] - out["opponent_form_5"]
    out["xg_diff"] = (out["xg_for_before"] - out["xg_against_before"]) - (
        out["opponent_xg_for_before"] - out["opponent_xg_against_before"]
    )
    out["attack_diff"] = out["team_attack_5"] - out["opponent_attack_5"]
    out["defense_diff"] = out["opponent_defense_5"] - out["team_defense_5"]
    out["injuries_diff"] = out["opponent_injuries"] - out["team_injuries"]
    out["possession_diff"] = out["possession_before"] - out["opponent_possession_before"]
    out["ucl_experience_diff"] = out["team_ucl_matches_before"] - out["opponent_ucl_matches_before"]
    out["bookmaker_prob_diff"] = out["bookmaker_prob"] - out["opponent_bookmaker_prob"]
    out["rest_days_diff"] = out["team_rest_days"] - out["opponent_rest_days"]
    out["h2h_win_rate_diff"] = np.where(
        out["Team"] == out["Home"],
        out["h2h_home_win_rate_prior"] - out["h2h_away_win_rate_prior"],
        out["h2h_away_win_rate_prior"] - out["h2h_home_win_rate_prior"],
    )
    out["h2h_draw_rate"] = out["h2h_draw_rate_prior"]
    return out


def build_dataset(raw_df: pd.DataFrame, config: dict[str, Any], injuries: dict[str, Any] | None = None) -> FeatureBuildResult:
    windows = config["features"].get("rolling_windows", [5, 10])
    events = make_fixture_events(raw_df)
    elo = add_elo_features(events)
    h2h = add_h2h_features(events)
    perspective = make_perspective_rows(events)
    perspective = perspective.merge(elo, on=["event_id", "Home", "Away"], how="left")
    perspective["team_elo_pre"] = np.where(perspective["Team"] == perspective["Home"], perspective["home_elo_pre"], perspective["away_elo_pre"])
    perspective = perspective.merge(h2h, on="event_id", how="left")
    perspective = add_shifted_team_features(perspective, windows)
    perspective = add_external_features(perspective, injuries)
    perspective["is_home"] = (perspective["Venue"] == "Home").astype(int)
    perspective["is_away"] = (perspective["Venue"] == "Away").astype(int)
    perspective["is_neutral"] = (perspective["Venue"] == "Neutral").astype(int)
    dataset = add_opponent_and_diff_features(perspective)

    categorical = config["features"].get("allowed_categorical", ["Competition", "Venue"]).copy()
    if config["features"].get("use_identity_features", False):
        categorical.extend(["Team", "Opponent", "Stage"])
    categorical = [c for c in categorical if c in dataset.columns]
    numeric = [c for c in BASE_NUMERIC_FEATURES if c in dataset.columns]
    features = categorical + numeric
    dataset[numeric] = dataset[numeric].replace([np.inf, -np.inf], np.nan).fillna(0)
    dataset[categorical] = dataset[categorical].fillna("Missing").astype(str)
    return FeatureBuildResult(dataset=dataset, feature_columns=features, categorical_columns=categorical, numeric_columns=numeric)


def build_final_vector(raw_df: pd.DataFrame, config: dict[str, Any], injuries: dict[str, Any] | None = None) -> pd.DataFrame:
    final_cfg = config["final_match"]
    events = make_fixture_events(raw_df)
    final_event = {
        "Date": pd.to_datetime(final_cfg["date"]),
        "Competition": final_cfg["competition"],
        "Comp_Code": "CL",
        "Matchday": pd.to_numeric(events.get("Matchday"), errors="coerce").max() + 1,
        "Home": final_cfg["team"],
        "Away": final_cfg["opponent"],
        "Home_Goals": np.nan,
        "Away_Goals": np.nan,
        "Venue": "Neutral",
        "Stage": final_cfg.get("stage", "FINAL"),
        "Group": np.nan,
    }
    combined = pd.concat([events.drop(columns=["event_id"]), pd.DataFrame([final_event])], ignore_index=True)
    result = build_dataset(combined, config, injuries)
    final_row = result.dataset[
        (result.dataset["Date"] == pd.to_datetime(final_cfg["date"]))
        & (result.dataset["Team"] == final_cfg["team"])
        & (result.dataset["Opponent"] == final_cfg["opponent"])
    ].copy()
    odds = final_cfg.get("bookmaker_odds", {})
    if odds:
        total = sum(1 / float(v) for v in odds.values())
        team_prob = (1 / float(odds["team"])) / total
        opponent_prob = (1 / float(odds["opponent"])) / total
        final_row["bookmaker_prob"] = team_prob
        final_row["opponent_bookmaker_prob"] = opponent_prob
        final_row["bookmaker_prob_diff"] = team_prob - opponent_prob
    return final_row
