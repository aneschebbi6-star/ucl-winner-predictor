import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import GridSearchCV

from console import configure_console, print_header, print_kv, print_section

configure_console()
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠️ XGBoost non installé. Installez-le avec `pip install xgboost` pour l'utiliser.")

def load_data():
    train_path = os.path.join("data", "processed", "train_dataset.csv")
    test_path = os.path.join("data", "processed", "test_dataset.csv")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("❌ Fichiers de données introuvables. Lancez src/dataset.py d'abord.")
        sys.exit(1)
        
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def prepare_features(train_df, test_df):
    # Colonnes qui fuient la cible ou inutiles
    drop_cols = ['Date', 'Home', 'Away', 'Home_Goals', 'Away_Goals', 
                 'Buts_Pour', 'Buts_Contre', 'Resultat', 'Comp_Code', 'Group']
    
    X_train = train_df.drop(columns=drop_cols + ['y_target'], errors='ignore')
    y_train = train_df['y_target']
    
    X_test = test_df.drop(columns=drop_cols + ['y_target'], errors='ignore')
    y_test = test_df['y_target']
    
    # Identifier les variables catégorielles et numériques
    categorical_cols = ['Competition', 'Team', 'Opponent', 'Venue', 'Stage']
    numeric_cols = ['Matchday']
    
    # Assurez-vous que les colonnes existent
    categorical_cols = [c for c in categorical_cols if c in X_train.columns]
    numeric_cols = [c for c in numeric_cols if c in X_train.columns]
    
    X_train[categorical_cols] = X_train[categorical_cols].fillna("Missing").astype(str)
    X_test[categorical_cols] = X_test[categorical_cols].fillna("Missing").astype(str)
    X_train[numeric_cols] = X_train[numeric_cols].fillna(0)
    X_test[numeric_cols] = X_test[numeric_cols].fillna(0)

    # Pipeline de prétraitement
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ])
    
    return X_train, y_train, X_test, y_test, preprocessor

def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_proba, labels=model.classes_)
    cm = confusion_matrix(y_test, y_pred)
    
    print_section(f"📈 {name}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  LogLoss  : {ll:.4f}")
    print(f"  Matrice de confusion :\n{cm}")
    print("\n  Rapport de classification :")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    return acc, ll

def main():
    print_header("Étape 5 : Entraînement des modèles")
    print("\n  Chargement des données...")
    train_df, test_df = load_data()
    print_kv("Train", f"{len(train_df)} matchs")
    print_kv("Test", f"{len(test_df)} matchs")
    X_train, y_train, X_test, y_test, preprocessor = prepare_features(train_df, test_df)
    
    # 1. Baseline : Régression Logistique
    print("\nEntraînement de la Baseline (Régression Logistique)...")
    lr_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', LogisticRegression(max_iter=1000))])
    lr_pipeline.fit(X_train, y_train)
    evaluate_model("Régression Logistique (Baseline)", lr_pipeline, X_test, y_test)
    
    # 2. Modèle d'Ensemble : Random Forest avec optimisation (GridSearchCV)
    print("\nEntraînement du Random Forest avec GridSearchCV...")
    rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', RandomForestClassifier(random_state=42))])
    
    param_grid = {
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [None, 5, 10],
        'classifier__min_samples_split': [2, 5]
    }
    
    grid_search = GridSearchCV(rf_pipeline, param_grid, cv=3, scoring='neg_log_loss', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"Meilleurs paramètres RF : {grid_search.best_params_}")
    best_rf = grid_search.best_estimator_
    evaluate_model("Random Forest (Optimisé)", best_rf, X_test, y_test)
    
    # 3. XGBoost
    if HAS_XGB:
        print("\nEntraînement de XGBoost...")
        xgb_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                      ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42))])
        
        xgb_param_grid = {
            'classifier__n_estimators': [50, 100],
            'classifier__learning_rate': [0.01, 0.1],
            'classifier__max_depth': [3, 5]
        }
        
        xgb_grid = GridSearchCV(xgb_pipeline, xgb_param_grid, cv=3, scoring='neg_log_loss', n_jobs=-1)
        xgb_grid.fit(X_train, y_train)
        
        print(f"Meilleurs paramètres XGB : {xgb_grid.best_params_}")
        best_xgb = xgb_grid.best_estimator_
        evaluate_model("XGBoost (Optimisé)", best_xgb, X_test, y_test)

if __name__ == "__main__":
    main()
