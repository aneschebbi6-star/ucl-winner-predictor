import pandas as pd
import os
import sys

from console import configure_console, print_footer, print_header, print_kv, print_section

configure_console()

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
try:
    from xgboost import XGBClassifier
except ImportError:
    print("⚠️ XGBoost non installé. Installez-le avec `pip install xgboost`.")
    sys.exit(1)

def load_all_data():
    train_path = os.path.join("data", "processed", "train_dataset.csv")
    test_path = os.path.join("data", "processed", "test_dataset.csv")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("❌ Fichiers de données introuvables. Lancez src/dataset.py d'abord.")
        sys.exit(1)
        
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    # Pour la prédiction finale, on entraîne sur TOUTES les données disponibles
    return pd.concat([train_df, test_df], ignore_index=True)

def train_final_model(df):
    drop_cols = ['Date', 'Home', 'Away', 'Home_Goals', 'Away_Goals', 
                 'Buts_Pour', 'Buts_Contre', 'Resultat', 'Comp_Code', 'Group']
    
    X = df.drop(columns=drop_cols + ['y_target'], errors='ignore')
    y = df['y_target']
    
    categorical_cols = ['Competition', 'Team', 'Opponent', 'Venue', 'Stage']
    numeric_cols = ['Matchday']
    
    categorical_cols = [c for c in categorical_cols if c in X.columns]
    numeric_cols = [c for c in numeric_cols if c in X.columns]
    
    X[categorical_cols] = X[categorical_cols].fillna("Missing").astype(str)
    X[numeric_cols] = X[numeric_cols].fillna(0)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ])
    
    # On utilise les meilleurs hyperparamètres trouvés à l'étape 5
    xgb_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(
            use_label_encoder=False, 
            eval_metric='logloss', 
            random_state=42,
            learning_rate=0.1,
            max_depth=3,
            n_estimators=50
        ))
    ])
    
    xgb_pipeline.fit(X, y)
    return xgb_pipeline

def main():
    print_header("🏆 Étape 6 : Simulation et inférence (finale)")
    print()

    df = load_all_data()
    model = train_final_model(df)
    
    # 1. Construire le Feature Vector de la Finale
    # PSG est désigné "Home" sur le papier
    final_match = pd.DataFrame([{
        'Competition': 'Champions League',
        'Team': 'Paris Saint-Germain FC',
        'Opponent': 'Arsenal FC',
        'Venue': 'Neutral', # Ou 'Home' pour le statut administratif
        'Stage': 'FINAL',
        'Matchday': 1
    }])
    
    print_section("Feature vector — finale")
    for col in final_match.columns:
        print_kv(col, str(final_match[col].iloc[0]))
    
    # 2. Prédiction des probabilités et du résultat final
    # Notre cible y_target : 0 = Victoire Team (PSG), 1 = Victoire Opponent (Arsenal)
    probabilities = model.predict_proba(final_match)[0]
    psg_prob = probabilities[0]
    arsenal_prob = probabilities[1]
    
    predicted_class = model.predict(final_match)[0]
    predicted_winner = "Paris Saint-Germain FC" if predicted_class == 0 else "Arsenal FC"
    
    print_section("Prédiction XGBoost")
    print_kv("Vainqueur prédit", predicted_winner)
    print_kv("PSG", f"{psg_prob * 100:.2f}%")
    print_kv("Arsenal", f"{arsenal_prob * 100:.2f}%")
    
    # 3. Comparaison avec les bookmakers pour chercher la "Value"
    # Cotes hypothétiques pour la démonstration (Pinnacle/Bet365)
    # Note : Normalement, on scrapperait les cotes en temps réel.
    cotes_bookmakers = {
        'PSG': 2.20,     # Probabilité implicite = 1 / 2.20 = 45.45%
        'Arsenal': 2.90  # Probabilité implicite = 1 / 2.90 = 34.48%
    }
    
    psg_implied = (1 / cotes_bookmakers['PSG']) * 100
    arsenal_implied = (1 / cotes_bookmakers['Arsenal']) * 100
    
    print_section("Bookmakers vs modèle")
    print_kv("Cote PSG", f"{cotes_bookmakers['PSG']} (implicite {psg_implied:.2f}%)")
    print_kv("Cote Arsenal", f"{cotes_bookmakers['Arsenal']} (implicite {arsenal_implied:.2f}%)")
    
    print_section("Analyse value bet")
    if psg_prob * 100 > psg_implied:
        diff = (psg_prob * 100) - psg_implied
        print(f"  ✅ Value sur le PSG (+{diff:.2f}% vs bookmakers)")
        print("  → Recommandation : victoire PSG")
    elif arsenal_prob * 100 > arsenal_implied:
        diff = (arsenal_prob * 100) - arsenal_implied
        print(f"  ✅ Value sur Arsenal (+{diff:.2f}% vs bookmakers)")
        print("  → Recommandation : victoire Arsenal")
    else:
        print("  ❌ Pas de value bet claire — cotes alignées avec le modèle")
        
    print_footer("Simulation terminée")

if __name__ == "__main__":
    main()
