from __future__ import annotations

import pandas as pd

from config import load_config
from data.split import split_train_test
from features.pipeline import build_dataset


def test_split_is_strictly_chronological(sample_matches, config):
    result = build_dataset(sample_matches, config)
    train, test = split_train_test(result.dataset, "2025-04-01")

    assert not train.empty
    assert not test.empty
    assert train["Date"].max() < test["Date"].min()
    assert (train["Date"] < "2025-04-01").all()
    assert (test["Date"] >= "2025-04-01").all()


def test_split_drops_rows_without_target(sample_matches, config):
    result = build_dataset(sample_matches, config)
    train, test = split_train_test(result.dataset, "2025-04-01")
    assert train["y_target"].notna().all()
    assert test["y_target"].notna().all()


def test_train_test_no_duplicate_event_rows(sample_matches, config):
    result = build_dataset(sample_matches, config)
    train, test = split_train_test(result.dataset, "2025-04-01")
    train_events = set(zip(train["event_id"], train["Team"]))
    test_events = set(zip(test["event_id"], test["Team"]))
    assert train_events.isdisjoint(test_events)
