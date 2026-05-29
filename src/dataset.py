import pandas as pd
import numpy as np
import os

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

def prepare_dataset(matches_df: pd.DataFrame, split_date: str = '2025-08-01') -> tuple:
    """
    Étape 4 : Choix de la Cible (Target Definition) & Split Temporel
    
    Arguments:
    - matches_df : DataFrame contenant les matchs (doit avoir les colonnes 'Date', 'Home_Goals', 'Away_Goals')
    - split_date : Date pour la séparation chronologique (Time-Based Split)
    
    Retourne:
    - train_df, test_df
    """
    print_header("Étape 4 : Cible & split temporel")
    print()
    
    df = matches_df.copy()
    
    # S'assurer que la colonne Date est bien au format datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 1. Choix de la Cible : Option A (Binaire)
    # 0 = Équipe A (Home) gagne (soulève le trophée)
    # 1 = Équipe B (Away) gagne (soulève le trophée)
    # Note : S'il y a des matchs nuls dans les données historiques, on peut les exclure
    # ou les attribuer à l'équipe qui passe aux tirs au but (ici on les exclut pour l'exemple strict "soulève le trophée")
    df = df[df['Home_Goals'] != df['Away_Goals']].copy()
    
    # y = 0 si Home gagne, 1 si Away gagne
    df['y_target'] = np.where(df['Home_Goals'] > df['Away_Goals'], 0, 1)
    print_kv("Cible y_target", "0 = domicile gagne, 1 = extérieur gagne")
    
    # 2. Time-Based Split : Séparer chronologiquement
    df = df.sort_values(by='Date').reset_index(drop=True)
    
    train_df = df[df['Date'] < split_date].copy()
    test_df = df[df['Date'] >= split_date].copy()
    
    print_kv("Date de coupure", split_date)
    print_kv("Train", f"{len(train_df)} matchs")
    print_kv("Test", f"{len(test_df)} matchs")
    hidden = count_hidden_cl_matches(test_df)
    if hidden:
        print_kv("LDC masqués (test)", f"{hidden} matchs avant quarts de finale")
    cl_test = filter_cl_from_quarters(test_df)
    if not cl_test.empty:
        print_section("LDC dans le jeu de test (quarts → finale)")
        print_cl_knockout_table(cl_test, team_perspective=True)
    
    return train_df, test_df

if __name__ == "__main__":
    # Test simple avec les données combinées si disponibles
    data_path = os.path.join("data", "raw", "psg_arsenal_combined.csv")
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        # On va utiliser une date factice pour le split vu que nous n'avons que 2025-2026 dans ce fichier
        # On coupe au milieu de la saison pour tester
        train, test = prepare_dataset(df, split_date='2026-01-01')
        
        # Sauvegarde
        os.makedirs("data/processed", exist_ok=True)
        train.to_csv("data/processed/train_dataset.csv", index=False)
        test.to_csv("data/processed/test_dataset.csv", index=False)
        print_footer("Datasets sauvegardés → data/processed/")
    else:
        print("  ❌ Fichier de données non trouvé. Lancez src/scrapper.py d'abord.")
