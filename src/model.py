from __future__ import annotations

from config import load_config
from console import configure_console, print_footer, print_header, print_kv, print_section, print_status, print_table
from data.loaders import load_json, load_matches, save_dataset
from data.split import split_train_test
from features.pipeline import build_dataset
from models.train import train_and_save


def main() -> None:
    configure_console()
    config = load_config()
    print_header("Entrainement ML calibre")

    raw = load_matches(config["data"]["raw_matches_path"])
    injuries = load_json(config["data"]["injuries_path"], default={})
    result = build_dataset(raw, config, injuries)
    train_df, test_df = split_train_test(result.dataset, config["data"]["split_date"])
    save_dataset(result.dataset, config["data"]["processed_dataset_path"])
    save_dataset(train_df, config["data"]["train_path"])
    save_dataset(test_df, config["data"]["test_path"])

    artifact = train_and_save(
        result.dataset,
        result.feature_columns,
        result.categorical_columns,
        result.numeric_columns,
        config,
    )

    print_table(
        ["Dataset", "Rows", "Start", "End"],
        [
            ["Train", len(train_df), train_df["Date"].min().date(), train_df["Date"].max().date()],
            ["Test", len(test_df), test_df["Date"].min().date(), test_df["Date"].max().date()],
        ],
    )
    print_kv("Features", str(len(result.feature_columns)))
    print_status("No train/test overlap", train_df["Date"].max() < test_df["Date"].min())
    print_section("Metriques")
    metric_rows = []
    for model_name, metrics in artifact["metrics"].items():
        if "accuracy" in metrics:
            metric_rows.append(
                [
                    model_name,
                    f"{metrics['accuracy']:.4f}",
                    f"{metrics['log_loss']:.4f}",
                    f"{metrics['brier_score']:.4f}",
                ]
            )
        else:
            metric_rows.append(
                [
                    model_name,
                    "-",
                    f"{metrics['mean_log_loss']:.4f} +/- {metrics['std_log_loss']:.4f}",
                    "-",
                ]
            )
    print_table(["Model", "Accuracy", "LogLoss", "Brier"], metric_rows)
    print_footer("Modele sauvegarde -> artifacts/model.joblib")


if __name__ == "__main__":
    main()
