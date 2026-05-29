from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import load_config  # noqa: E402


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def sample_matches() -> pd.DataFrame:
    rows = []
    fixtures = [
        ("2025-01-01", "Paris Saint-Germain FC", "Arsenal FC", 2, 0),
        ("2025-01-08", "Arsenal FC", "Paris Saint-Germain FC", 1, 1),
        ("2025-02-01", "Paris Saint-Germain FC", "Arsenal FC", 0, 1),
        ("2025-02-08", "Arsenal FC", "Paris Saint-Germain FC", 2, 2),
        ("2025-03-01", "Paris Saint-Germain FC", "Arsenal FC", 3, 1),
        ("2025-03-08", "Arsenal FC", "Paris Saint-Germain FC", 1, 0),
        ("2025-04-01", "Paris Saint-Germain FC", "Arsenal FC", 1, 1),
        ("2025-04-08", "Arsenal FC", "Paris Saint-Germain FC", 0, 2),
        ("2025-05-01", "Paris Saint-Germain FC", "Arsenal FC", 2, 1),
        ("2025-05-08", "Arsenal FC", "Paris Saint-Germain FC", 1, 3),
        ("2025-06-01", "Paris Saint-Germain FC", "Arsenal FC", 4, 0),
        ("2025-06-08", "Arsenal FC", "Paris Saint-Germain FC", 2, 2),
    ]
    for idx, (date, home, away, hg, ag) in enumerate(fixtures, start=1):
        rows.append(
            {
                "Date": date,
                "Competition": "Champions League",
                "Comp_Code": "CL",
                "Matchday": idx,
                "Home": home,
                "Away": away,
                "Home_Goals": hg,
                "Away_Goals": ag,
                "Stage": "LEAGUE_STAGE",
                "Group": None,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def tmp_config(tmp_path, config):
    cfg = dict(config)
    cfg["data"] = dict(config["data"])
    cfg["model"] = dict(config["model"])
    cfg["data"]["split_date"] = "2025-04-01"
    cfg["model"]["artifact_dir"] = str(tmp_path / "artifacts")
    cfg["model"]["calibration_cv_splits"] = 2
    cfg["model"]["timeseries_cv_splits"] = 2
    return cfg
