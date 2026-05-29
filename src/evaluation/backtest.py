from __future__ import annotations

import pandas as pd


def backtest_finals(dataset: pd.DataFrame) -> pd.DataFrame:
    """Return historical final rows when multi-season UCL data is available."""
    if "Stage" not in dataset.columns:
        return pd.DataFrame()
    finals = dataset[dataset["Stage"].astype(str).str.upper().eq("FINAL")].copy()
    return finals.sort_values("Date")
