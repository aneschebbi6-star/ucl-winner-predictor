from __future__ import annotations

from config import load_config
from console import configure_console, print_footer, print_header, print_kv, print_status, print_table
from data.loaders import load_json, load_matches, save_dataset
from data.split import split_train_test
from features.pipeline import build_dataset


def main() -> None:
    configure_console()
    config = load_config()
    print_header("Dataset professionnel anti-leakage")

    raw = load_matches(config["data"]["raw_matches_path"])
    injuries = load_json(config["data"]["injuries_path"], default={})
    result = build_dataset(raw, config, injuries)
    train_df, test_df = split_train_test(result.dataset, config["data"]["split_date"])

    save_dataset(result.dataset, config["data"]["processed_dataset_path"])
    save_dataset(train_df, config["data"]["train_path"])
    save_dataset(test_df, config["data"]["test_path"])

    print_table(
        ["Dataset", "Rows", "Start", "End"],
        [
            ["Train", len(train_df), train_df["Date"].min().date(), train_df["Date"].max().date()],
            ["Test", len(test_df), test_df["Date"].min().date(), test_df["Date"].max().date()],
            ["All", len(result.dataset), result.dataset["Date"].min().date(), result.dataset["Date"].max().date()],
        ],
    )
    print_kv("Feature columns", str(len(result.feature_columns)))
    print_kv("Categorical columns", ", ".join(result.categorical_columns) or "none")
    print_status("Chronological split", train_df["Date"].max() < test_df["Date"].min())
    print_status("Target available", result.dataset["y_target"].notna().all())
    print_footer("Datasets sauvegardes")


if __name__ == "__main__":
    main()
