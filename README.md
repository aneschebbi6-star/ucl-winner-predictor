# UEFA Champions League Final Predictor 🏆

Projet Python de prédiction football pour la finale UEFA Champions League **PSG vs Arsenal**.  
Pipeline complet avec interface web interactive Streamlit : collecte de données, feature engineering, entraînement ML, prédiction et analyse.

> Projet éducatif uniquement. Les sorties ne sont pas un conseil de pari.

## 🎯 Objectif

Prédire le résultat 1N2 (Victoire domicile / Nul / Victoire extérieur) pour la finale PSG vs Arsenal sur terrain neutre.

| Classe | Signification |
|---|---|
| `0` | PSG gagne |
| `1` | Match nul |
| `2` | Arsenal gagne |

## 🚀 Lancement rapide

### Option 1: Interface Web (Recommandé)
```powershell
cd "E:\frrelance\champions predection"
streamlit run app.py
```
Accédez à : `http://localhost:8501` 🌐

### Option 2: Pipeline complet (Terminal)
```powershell
.\scripts\run_pipeline.ps1
python src/predict.py
```

## 📊 Architecture

## 📊 Architecture

```
champions-predection/
├── app.py                          # Interface Streamlit principale
├── .streamlit/
│   └── config.toml                 # Configuration Streamlit (thème, port)
├── scripts/
│   ├── run_app.ps1                 # Script pour lancer l'app
│   └── run_pipeline.ps1            # Script pour lancer le pipeline
├── data/
│   ├── raw/                        # Données brutes (API Football-Data.org)
│   │   ├── psg_matches.csv
│   │   ├── arsenal_matches.csv
│   │   ├── cl_all_matches.csv
│   │   └── ...
│   └── processed/                  # Datasets traités
│       ├── train_dataset.csv
│       ├── test_dataset.csv
│       └── team_features.json
├── artifacts/
│   └── model.joblib                # Modèle ML entraîné
├── src/
│   ├── app.py → streamlit
│   ├── scrapper.py                 # Collecte données + blessures
│   ├── dataset.py                  # Préparation dataset
│   ├── features.py                 # Feature engineering
│   ├── model.py                    # Entraînement modèles
│   ├── predict.py                  # Prédiction finale
│   ├── console.py                  # Affichage terminal
│   ├── config.py                   # Config centralisée
│   ├── ui/
│   │   ├── helpers.py              # Fonctions UI
│   │   └── styles.py               # Thème CSS (clair)
│   ├── data/
│   ├── features/
│   ├── models/
│   └── evaluation/
├── requirements.txt
└── README.md
```

## 🌐 Interface Streamlit

Tableau de bord interactif avec **5 onglets** :

| Onglet | Fonction |
|---|---|
| **🏠 Accueil** | Prédictions en grand format + graphiques probabilités + comparaison modèle vs marché |
| **📊 Analyse** | Analysis détaillée des probabilités et identification des "value bets" |
| **🔍 Features** | Affichage de tous les indicateurs utilisés par le modèle |
| **✅ Qualité** | Contrôles d'intégrité du modèle et métriques de performance |
| **ℹ️ À propos** | Documentation complète du projet et stack technique |

### Caractéristiques
- 🎨 **Thème clair** professionnel
- 📱 **Responsive** (mobile + desktop)
- 📊 **Graphiques interactifs** (Plotly)
- ⚡ **Cache des données** pour rapidité
- 🔄 **Mise à jour auto** en cas de modification

### Lancer l'app
```powershell
# Méthode 1 : Script automatique
.\scripts\run_app.ps1

# Méthode 2 : Commande directe
streamlit run app.py
```

**L'app s'ouvre automatiquement** à `http://localhost:8501` 🚀

### Déployer en ligne (Optionnel)
```bash
git add .
git commit -m "Add Streamlit interface"
git push
```
Puis allez sur [share.streamlit.io](https://share.streamlit.io) pour déployer gratuitement.

## 📋 Installation

### 1. Cloner le repository
```powershell
git clone https://github.com/aneschebbi6-star/ucl-winner-predictor.git
cd "champions predection"
```

### 2. Créer environnement virtuel
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Installer dépendances
```powershell
pip install -r requirements.txt
```

## 🔄 Pipeline complet

Ordre recommandé :

```powershell
# 1. Collecte données
python src/scrapper.py

# 2. Préparation dataset
python src/dataset.py

# 3. Entraînement modèles
python src/model.py

# 4. Prédiction finale
python src/predict.py
```

**OU utiliser le script :**
```powershell
.\scripts\run_pipeline.ps1
```

## 🛠️ Rôle des scripts

| Script | Rôle | Output |
|---|---|---|
| `src/scrapper.py` | Collecte données brutes via football-data.org API + Transfermarkt scraping | `data/raw/*.csv` |
| `src/dataset.py` | Prépare dataset ML avec features et split chronologique | `data/processed/*.csv` |
| `src/features.py` | Feature engineering (forme, xG, blessures, H2H, etc.) | Variables intégrées |
| `src/model.py` | Entraîne modèles (Logistic Regression, XGBoost) et calibre probabilités | `artifacts/model.joblib` |
| `src/predict.py` | Charge modèle et prédit le match PSG vs Arsenal | Terminal + affichage |
| `app.py` | Interface Streamlit interactive | `http://localhost:8501` |

## ⚙️ Features utilisés

Tous calculés **strictement avant match** avec `shift()` pour éviter le data leakage :

```
Différentiels d'équipes :
├── elo_diff           (Rating Elo)
├── form_diff          (Forme derniers 5 matchs, pondérée)
├── attack_diff        (Buts marqués par match)
├── defense_diff       (Buts encaissés par match)
├── xg_diff            (Expected Goals)
├── possession_diff    (% Possession moyen)

Contexte spécifique :
├── injuries_diff      (Impact des blessures vs adversaire)
├── ucl_experience_diff (Matchs joués en LDC)
├── h2h_win_rate_diff  (Historique face-à-face)
├── h2h_draw_rate      (% Nuls en confrontation directe)
├── bookmaker_prob_diff (Cotes vs proba modèle)
└── rest_days_diff     (Jours de repos avant match)
```

**Total : 40+ features engineered**

## 🤖 Modèles

Le training compare 3 approches :

| Modèle | Type | Avantages |
|---|---|---|
| **Logistic Regression (calibrée)** | Linéaire | Simple, probabilités prudentes, pas d'overfitting |
| **XGBoost** | Gradient Boosting | Capture interactions non-linéaires, performance robuste |
| **Elo Baseline** | Référence | Football-natif, interprétable |

**Sélection** : Le meilleur modèle (selon LogLoss CV) est automatiquement sauvegardé dans `artifacts/model.joblib`

## 📊 Métriques

Affichées dans l'interface et terminal :

```
├── Accuracy              (Taux de bonnes prédictions)
├── LogLoss               (Calibration des probabilités - CLEF)
├── Brier Score           (Erreur quadratique moyenne)
├── TimeSeriesSplit CV    (Validation sur données futures)
└── Calibration curves    (Fiabilité réelle vs prédite)
```

### Calibration
La calibration est essentielle pour l'analyse de "value bets". Un modèle bien calibré :
- Quand il prédit 60%, il gagne ~60% des fois
- Les probabilités peuvent être comparées aux cotes de bookmakers

## ✅ Tests

Vérifications d'intégrité automatiques :

```powershell
pytest -v
```

Tests inclus :

| Test | Vérifie |
|---|---|
| `test_target_mapping` | Les classes 0/1/2 correspondent au bon résultat |
| `test_no_leakage` | Aucune information du match future n'est utilisée |
| `test_shift_correctness` | Les features utilisent uniquement les matchs passés |
| `test_feature_alignment` | Cohérence features train/test/predict |
| `test_final_vector` | Vecteur PSG vs Arsenal complet et valide |

## 🔧 Configuration

Fichier principal : `config.yaml`

```yaml
project:
  random_state: 42
  season: 2025

data:
  split_date: "2026-01-01"      # Date de split train/test
  raw_matches_path: data/raw/...
  processed_dataset_path: data/processed/...

features:
  use_identity_features: false  # Évite la mémorisation

model:
  selected_model: xgboost       # ou logistic_calibrated
  random_state: 42
```

## 📊 Sources de données

| Source | Données | API/Scraping |
|---|---|---|
| [football-data.org](https://www.football-data.org/) | Matchs, standings, scores | REST API v4 (Plan gratuit) |
| [Transfermarkt](https://www.transfermarkt.com/) | Blessures, suspensions | Web Scraping (BeautifulSoup) |
| Bookmakers | Cotes 1N2 | Configuration manuelle dans `config.yaml` |

## ⚠️ Limitations et améliorations

### Limitations actuelles
- Dataset limité (historique 2024-2026)
- xG et possession = 0 si aucune source disponible
- Finale est événement haute-variance (rarement 100% prédictible)
- Modèle calibré sur matchs réguliers, contexte final peut différer

### Améliorations futures
- [ ] Intégration API temps réel pour Transfermarkt
- [ ] Prédictions probabilistes supplémentaires (score exact, over/under)
- [ ] Comparaison avec modèles d'autres plateformes
- [ ] Dashboard avec historique des prédictions antérieures
- [ ] Export PDF des analyses
- [ ] Intégration webscraping temps réel des cotes

## 📱 Portfolio et engagement

### Pour LinkedIn
- ✅ Partager le lien Streamlit directement
- ✅ Publier les prédictions et résultats en temps réel
- ✅ Montrer les captures d'écran du dashboard
- ✅ Mettre en avant : Data, ML, Feature Engineering, Calibration

### Pour GitHub
- ⭐ Repo: [aneschebbi6-star/ucl-winner-predictor](https://github.com/aneschebbi6-star/ucl-winner-predictor)
- ✅ Open source pour portfolio Data Science
- ✅ Tests inclus et bien documenté

## 🔗 Stack technique

**Backend ML**
- Python 3.10+
- pandas, numpy (Data processing)
- scikit-learn (Logistic Regression, metrics)
- XGBoost (Gradient boosting)
- joblib (Model persistence)

**Frontend**
- Streamlit (Web app)
- Plotly (Interactive charts)
- Streamlit Option Menu (Navigation)

**DevOps**
- Git / GitHub
- Pytest (Unit tests)
- PowerShell scripts (Automation)

## 📝 Limites légales / Légales

Le projet est **éducatif et à titre d'exemple seulement**.

## 📋 Exemple de sortie

### Terminal (`python src/predict.py`)
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
     Prediction finale PSG vs Arsenal
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[Match]
Team: PSG | Opponent: Arsenal | Venue: Neutral

[Prediction calibrée]
PSG gagne       50.27%  ✓ Favori
Match nul       21.26%
Arsenal gagne   28.46%

[Modèle vs Marché]
Outcome        Model    Market   Edge
PSG gagne      50.27%   41.57%   +8.70 pp  (Opportunity)
Match nul      21.26%   26.90%   -5.64 pp
Arsenal gagne  28.46%   31.53%   -3.07 pp

[Contrôle qualité]
✅ Model artifact exists
✅ Final venue is neutral
✅ Feature alignment
✅ Probabilities sum to 1
⚠️ xG data available (neutral fallback used)
⚠️ Possession data available (neutral fallback used)
```

### Streamlit (`streamlit run app.py`)
- Dashboard interactif avec graphiques
- Onglets pour analyse détaillée
- Export de données
- Navigation fluide

## 🏆 Résultats attendus

Le modèle calibré devrait :
- ✅ Donner des probabilités réalistes (bien calibrées)
- ✅ Identifier les "value bets" vs les cotes
- ✅ Performer mieux qu'un baseline Elo seul
- ✅ Gérer gracieusement les données manquantes

## 📞 Support & Amélioration

Pour des questions ou améliorations :
1. Ouvrir une **Issue** sur GitHub
2. Forker et proposer une **Pull Request**
3. Contacter via LinkedIn : [@aneschebbi6-star](https://linkedin.com/in/aneschebbi6-star)

## 📄 Licence

Projet éducatif **open source** - MIT License

Utilisation libre pour fins pédagogiques et portfolio. Respecte les conditions de football-data.org, Transfermarkt et sources externes.

---

**Dernière mise à jour** : Mai 2026  
**Auteur** : @aneschebbi6-star  
**Repository** : [aneschebbi6-star/ucl-winner-predictor](https://github.com/aneschebbi6-star/ucl-winner-predictor)
