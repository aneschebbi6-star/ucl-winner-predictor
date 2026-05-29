from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from config import resolve_path


def load_matches(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(resolve_path(path))
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def load_json(path: str | Path, default=None):
    file_path = resolve_path(path)
    if not file_path.exists():
        return {} if default is None else default
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_dataset(df: pd.DataFrame, path: str | Path) -> None:
    file_path = resolve_path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=False)
