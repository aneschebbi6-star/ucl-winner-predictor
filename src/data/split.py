from __future__ import annotations

import pandas as pd


def split_train_test(dataset: pd.DataFrame, split_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = dataset.sort_values("Date").reset_index(drop=True)
    train = ordered[ordered["Date"] < split_date].copy()
    test = ordered[ordered["Date"] >= split_date].copy()
    train = train[train["y_target"].notna()].copy()
    test = test[test["y_target"].notna()].copy()
    train["y_target"] = train["y_target"].astype(int)
    test["y_target"] = test["y_target"].astype(int)
    return train, test
