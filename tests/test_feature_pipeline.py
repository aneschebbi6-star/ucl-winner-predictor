from __future__ import annotations

import pandas as pd

from config import load_config
from features.pipeline import build_dataset, build_final_vector


def sample_matches() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Date": "2025-01-01",
                "Competition": "Champions League",
                "Comp_Code": "CL",
                "Matchday": 1,
                "Home": "Paris Saint-Germain FC",
                "Away": "Arsenal FC",
                "Home_Goals": 2,
                "Away_Goals": 0,
                "Stage": "LEAGUE_STAGE",
                "Group": None,
            },
            {
                "Date": "2025-01-08",
                "Competition": "Champions League",
                "Comp_Code": "CL",
                "Matchday": 2,
                "Home": "Arsenal FC",
                "Away": "Paris Saint-Germain FC",
                "Home_Goals": 1,
                "Away_Goals": 1,
                "Stage": "LEAGUE_STAGE",
                "Group": None,
            },
        ]
    )


def test_target_mapping_is_team_perspective():
    config = load_config()
    result = build_dataset(sample_matches(), config)
    psg_first = result.dataset[
        (result.dataset["Date"] == pd.Timestamp("2025-01-01"))
        & (result.dataset["Team"] == "Paris Saint-Germain FC")
    ].iloc[0]
    arsenal_first = result.dataset[
        (result.dataset["Date"] == pd.Timestamp("2025-01-01"))
        & (result.dataset["Team"] == "Arsenal FC")
    ].iloc[0]

    assert int(psg_first["y_target"]) == 0
    assert int(arsenal_first["y_target"]) == 2


def test_shift_correctness_uses_only_previous_matches():
    config = load_config()
    result = build_dataset(sample_matches(), config)
    psg_rows = result.dataset[result.dataset["Team"] == "Paris Saint-Germain FC"].sort_values("Date")

    assert psg_rows.iloc[0]["team_form_5"] == 1.0
    assert psg_rows.iloc[1]["team_form_5"] == 3.0


def test_no_leakage_columns_in_features():
    config = load_config()
    result = build_dataset(sample_matches(), config)
    leakage = set(config["features"]["leakage_columns"])

    assert leakage.isdisjoint(result.feature_columns)


def test_feature_alignment_contains_differential_features():
    config = load_config()
    result = build_dataset(sample_matches(), config)

    for column in ["elo_diff", "form_diff", "attack_diff", "defense_diff", "rest_days_diff", "h2h_win_rate_diff"]:
        assert column in result.feature_columns
        assert column in result.dataset.columns


def test_final_vector_consistency():
    config = load_config()
    final_vector = build_final_vector(sample_matches(), config)
    result = build_dataset(sample_matches(), config)

    assert len(final_vector) == 1
    assert final_vector.iloc[0]["Team"] == config["final_match"]["team"]
    assert final_vector.iloc[0]["Opponent"] == config["final_match"]["opponent"]
    assert set(result.feature_columns).issubset(final_vector.columns)
