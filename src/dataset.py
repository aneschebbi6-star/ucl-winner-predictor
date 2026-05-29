import os

import numpy as np
import pandas as pd

from console import (
    configure_console,
    count_hidden_cl_matches,
    filter_cl_from_quarters,
    print_cl_knockout_table,
    print_footer,
    print_header,
    print_kv,
    print_section,
)

configure_console()


def prepare_dataset(matches_df: pd.DataFrame, split_date: str = "2025-08-01") -> tuple:
    """
    Etape 4 : choix de la cible et split temporel.

    Arguments:
    - matches_df : DataFrame contenant les matchs avec Date, Home_Goals, Away_Goals
    - split_date : date de separation chronologique

    Retourne:
    - train_df, test_df
    """
    print_header("Etape 4 : Cible & split temporel")
    print()

    df = matches_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    # Cible 1N2 :
    # 0 = equipe Home gagne, 1 = match nul, 2 = equipe Away gagne
    df["y_target"] = np.select(
        [
            df["Home_Goals"] > df["Away_Goals"],
            df["Home_Goals"] == df["Away_Goals"],
            df["Home_Goals"] < df["Away_Goals"],
        ],
        [0, 1, 2],
    )
    print_kv("Cible y_target", "0 = domicile gagne, 1 = nul, 2 = exterieur gagne")

    df = df.sort_values(by="Date").reset_index(drop=True)

    train_df = df[df["Date"] < split_date].copy()
    test_df = df[df["Date"] >= split_date].copy()

    print_kv("Date de coupure", split_date)
    print_kv("Train", f"{len(train_df)} matchs")
    print_kv("Test", f"{len(test_df)} matchs")
    hidden = count_hidden_cl_matches(test_df)
    if hidden:
        print_kv("LDC masques (test)", f"{hidden} matchs avant quarts de finale")
    cl_test = filter_cl_from_quarters(test_df)
    if not cl_test.empty:
        print_section("LDC dans le jeu de test (quarts -> finale)")
        print_cl_knockout_table(cl_test, team_perspective=True)

    return train_df, test_df


if __name__ == "__main__":
    data_path = os.path.join("data", "raw", "psg_arsenal_combined.csv")
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        train, test = prepare_dataset(df, split_date="2026-01-01")

        os.makedirs("data/processed", exist_ok=True)
        train.to_csv("data/processed/train_dataset.csv", index=False)
        test.to_csv("data/processed/test_dataset.csv", index=False)
        print_footer("Datasets sauvegardes -> data/processed/")
    else:
        print("  Fichier de donnees non trouve. Lancez src/scrapper.py d'abord.")
