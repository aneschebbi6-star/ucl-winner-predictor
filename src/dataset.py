import pandas as pd
import numpy as np
import os

def prepare_dataset(matches_df: pd.DataFrame, split_date: str = '2025-08-01') -> tuple:
    """
    Étape 4 : Choix de la Cible (Target Definition) & Split Temporel
    
    Arguments:
    - matches_df : DataFrame contenant les matchs (doit avoir les colonnes 'Date', 'Home_Goals', 'Away_Goals')
    - split_date : Date pour la séparation chronologique (Time-Based Split)
    
    Retourne:
    - train_df, test_df
    """
    print("\n" + "="*60)
    print("  Étape 4 : Choix de la Cible & Split Temporel")
    print("="*60)
    
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
    print(f"[+] Variable cible (y_target) créée en binaire : 0 (Home gagne), 1 (Away gagne)")
    
    # 2. Time-Based Split : Séparer chronologiquement
    df = df.sort_values(by='Date').reset_index(drop=True)
    
    train_df = df[df['Date'] < split_date].copy()
    test_df = df[df['Date'] >= split_date].copy()
    
    print(f"[+] Time-Based Split appliqué avec la date charnière : {split_date}")
    print(f"    -> Train set (Saisons précédentes) : {len(train_df)} matchs")
    print(f"    -> Test set (Saison récente)       : {len(test_df)} matchs")
    
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
        print("\n[+] Datasets sauvegardés dans data/processed/ (train_dataset.csv, test_dataset.csv)")
    else:
        print("Fichier de données non trouvé pour le test.")
