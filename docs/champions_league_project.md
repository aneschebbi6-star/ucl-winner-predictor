# 🏆 Projet Machine Learning : Prédiction de la Finale de la Champions League

Ce document sert de feuille de route (Roadmap) pour concevoir, entraîner et déployer un modèle de Machine Learning capable de prédire l'issue de la finale de la Champions League.

---

## 📌 Architecture Générale du Projet

```
champions-predection/
├── data/
│   ├── raw/                           # CSV bruts (scraper)
│   │   ├── psg_matches.csv
│   │   ├── arsenal_matches.csv
│   │   ├── psg_arsenal_combined.csv
│   │   ├── cl_all_matches.csv
│   │   ├── fl1_standings.csv
│   │   └── pl_standings.csv
│   └── processed/                     # Datasets et features JSON
│       ├── injuries_impact.json
│       ├── team_features.json
│       ├── train_dataset.csv
│       └── test_dataset.csv
├── docs/
│   └── champions_league_project.md    # Feuille de route détaillée
├── src/
│   ├── scrapper.py                    # Étape 1 : collecte API + blessures
│   ├── features.py                    # Étape 2 : feature engineering
│   ├── dataset.py                     # Étape 3 : cible + split temporel
│   ├── model.py                       # Étape 4 : entraînement & évaluation
│   ├── predict.py                     # Étape 5 : prédiction finale
│   └── console.py                     # Affichage terminal homogène
├── requirements.txt                   # Dépendances Python
├── .env                               # Variables d'environnement
└── README.md                          # Vous êtes ici
```


---

## ⏳ Étapes de Réalisation du Projet

### Étape 1 : Collecte et Sourcing des Données (`data/raw`)
L'objectif est de centraliser l'historique des matchs des deux finalistes ainsi que l'historique des phases finales des saisons précédentes.
- [ ] Configurer un scraper avec `BeautifulSoup` / `Scrapy` pour **FBref** ou utiliser l'API **Football-Data.org** / **API-Football**.
- [ ] Récupérer les données de la saison en cours (Ligue des Champions + Championnats domestiques).
- [ ] Récupérer l'historique des confrontations directes (*Head-to-Head*).
- [ ] Collecter les données de blessures et suspensions de dernière minute.

### Étape 2 : Nettoyage et Préparation (Data Wrangling)
- [ ] Gérer les valeurs manquantes (matchs reportés, stats non enregistrées).
- [ ] Harmoniser le nom des clubs entre les différentes sources de données.
- [ ] Standardiser les formats de date et de score.

### Étape 3 : Feature Engineering (La clé de la performance)
Transformer les données brutes en indicateurs mathématiques digestes pour l'algorithme.
- [ ] **Moyennes Mobiles (Rolling Statistics) :** Calculer la moyenne des Expected Goals ($xG$) marqués et encaissés sur les 5 et 10 derniers matchs.
- [ ] **Indicateur de Forme :** Assigner des poids aux résultats récents (ex: Victoire = 3, Nul = 1, Défaite = 0 avec un coefficient plus fort pour les matchs récents).
- [ ] **Expérience & Fatigue :** Calculer le nombre moyen de minutes jouées par l'effectif clé et le nombre de jours de repos avant la finale.
- [ ] **Feature Terrain Neutre :** Adapter l'avantage domicile/extérieur pour refléter un contexte de finale sur terrain neutre.

### Étape 4 : Choix de la Cible (Target Definition) & Split Temporel
- [ ] Définir la variable cible $y$ : 
  - Option A (Binaire) : `0` = Équipe A soulève le trophée, `1` = Équipe B soulève le trophée.
  - Option B (Multi-classe, 90 mins) : `0` = Victoire A, `1` = Match Nul, `2` = Victoire B.
- [ ] **Time-Based Split :** Séparer le dataset d'entraînement et de test chronologiquement (ex: Saisons 2021-2025 en *Train*, Saison 2025-2026 en *Validation/Test*) pour éviter le *Data Leakage*.

### Étape 5 : Entraînement et Sélection du Modèle
- [ ] Développer un modèle de référence simple (Baseline) : **Régression Logistique**.
- [ ] Entraîner des modèles d'ensemble plus robustes : **Random Forest** et **XGBoost** / **LightGBM**.
- [ ] Optimiser les hyperparamètres (via `GridSearchCV` ou `Optuna`).
- [ ] Évaluer les performances avec des métriques adaptées : *Accuracy*, *LogLoss* (très importante pour les probabilités de paris) et la matrice de confusion.

### Étape 6 : Simulation et Inférence (Le Jour J)
- [ ] Construire la ligne de données (Feature Vector) spécifique pour la finale de samedi en combinant les stats récentes des deux équipes.
- [ ] Faire tourner le modèle avec la fonction `predict_proba()` pour obtenir un pourcentage de chance de victoire pour chaque équipe (ex: Arsenal 54% - 46% PSG).
- [ ] Comparer les probabilités du modèle avec les cotes des bookmakers pour identifier une potentielle "Value".

---

## 🛠️ Stack Technique Recommandée
* **Langage :** Python 3.10+
* **Data Science :** `pandas`, `numpy`, `scikit-learn`
* **Modélisation Avancée :** `xgboost` ou `lightgbm`
* **Scraping :** `requests`, `beautifulsoup4`