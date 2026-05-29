# UEFA Champions League Final Predictor

Projet Python de prediction football pour la finale UEFA Champions League **PSG vs Arsenal**.  
Le pipeline est organise pour un usage portfolio Data Science/ML : features anti-leakage, split chronologique, calibration des probabilites, artefacts sauvegardes et tests automatises.

> Projet educatif uniquement. Les sorties ne sont pas un conseil de pari.

## Objectif

Predire le resultat 1N2 depuis la perspective de l'equipe analysee :

| Classe | Signification si `Team = PSG` |
|---|---|
| `0` | PSG gagne |
| `1` | Match nul |
| `2` | PSG perd / Arsenal gagne |

La prediction finale charge un modele deja entraine. Elle ne re-entraine pas sur le jeu de test.

## Architecture

```text
champions predection/
|-- config.yaml
|-- data/
|   |-- raw/
|   |   |-- psg_arsenal_combined.csv
|   |   |-- psg_matches.csv
|   |   |-- arsenal_matches.csv
|   |   |-- cl_all_matches.csv
|   |-- processed/
|       |-- model_dataset.csv
|       |-- train_dataset.csv
|       |-- test_dataset.csv
|       |-- team_features.json
|       |-- injuries_impact.json
|-- artifacts/
|   |-- model.joblib
|   |-- metrics.joblib
|   |-- calibration_table.csv
|-- src/
|   |-- dataset.py
|   |-- model.py
|   |-- predict.py
|   |-- scrapper.py
|   |-- console.py
|   |-- config.py
|   |-- data/
|   |-- features/
|   |-- models/
|   |-- evaluation/
|-- tests/
|   |-- test_feature_pipeline.py
|-- requirements.txt
```

## Installation

Depuis PowerShell :

```powershell
cd "E:\frrelance\champions predection"
python -m pip install -r requirements.txt
```

Si Windows ne reconnait pas `python`, utilise :

```powershell
& C:\Users\ANES\AppData\Local\Microsoft\WindowsApps\python3.11.exe -m pip install -r requirements.txt
```

## Execution

Ordre recommande :

```powershell
python src/dataset.py
python src/model.py
python src/predict.py
```

Avec ton Python WindowsApps :

```powershell
& C:\Users\ANES\AppData\Local\Microsoft\WindowsApps\python3.11.exe src/dataset.py
& C:\Users\ANES\AppData\Local\Microsoft\WindowsApps\python3.11.exe src/model.py
& C:\Users\ANES\AppData\Local\Microsoft\WindowsApps\python3.11.exe src/predict.py
```

## Role des scripts

| Script | Role |
|---|---|
| `src/scrapper.py` | Collecte les donnees brutes via football-data.org et sources externes. |
| `src/dataset.py` | Construit le dataset ML avec features pre-match et split chronologique. |
| `src/model.py` | Entraine les modeles, calibre les probabilites et sauvegarde les artefacts. |
| `src/predict.py` | Charge `artifacts/model.joblib` et predit PSG vs Arsenal. |

## Features

Les features sont calculees strictement avant match avec `shift()` pour reduire le data leakage.

Features principales :

```text
elo_diff
form_diff
xg_diff
attack_diff
defense_diff
injuries_diff
possession_diff
ucl_experience_diff
bookmaker_prob_diff
rest_days_diff
h2h_win_rate_diff
h2h_draw_rate
```

Par defaut, les colonnes `Team`, `Opponent` et `Stage` ne sont pas utilisees comme features pour limiter la memorisation.

## Modeles

Le training compare :

| Modele | Pourquoi |
|---|---|
| Logistic Regression calibree | Simple, stable sur petit dataset, probabilites plus prudentes. |
| XGBoost shallow regularise | Modele non lineaire limite pour eviter l'overfitting. |
| Elo baseline | Baseline football interpretable. |

Les artefacts generes :

```text
artifacts/model.joblib
artifacts/metrics.joblib
artifacts/calibration_table.csv
```

## Metriques

Le projet affiche :

```text
Accuracy
LogLoss
Brier Score
TimeSeriesSplit CV LogLoss
```

La calibration table est exportee dans :

```text
artifacts/calibration_table.csv
```

## Tests

Lancer les tests :

```powershell
python -m pytest -q
```

Tests inclus :

| Test | Ce qu'il verifie |
|---|---|
| Target mapping | `0/1/2` correspond bien a `Team gagne / nul / Team perd`. |
| No leakage | Les colonnes interdites ne sont pas dans les features. |
| Shift correctness | Les moyennes utilisent seulement les matchs precedents. |
| Feature alignment | Les features d'entrainement et de prediction sont coherentes. |
| Final vector consistency | Le vecteur PSG vs Arsenal est complet et aligne. |

## Configuration

Les parametres principaux sont dans `config.yaml` :

```yaml
project:
  random_state: 42

data:
  split_date: "2026-01-01"

features:
  use_identity_features: false

model:
  selected_model: logistic_calibrated
```

Les cotes bookmaker de demonstration sont aussi configurees dans `config.yaml`, puis normalisees sans marge avant comparaison.

## Limites

Le projet est maintenant structure proprement, mais la qualite finale depend encore des donnees disponibles :

- historique actuel encore limite ;
- xG et possession mis a zero si aucune source n'est fournie ;
- blessures encore agregees, pas parfaitement datees ;
- cotes bookmaker dans `config.yaml`, pas scrappees en temps reel ;
- backtesting anciennes finales possible seulement avec historique multi-saisons.

## Sortie attendue

`python src/predict.py` affiche :

```text
Match
Key feature checks
Prediction calibree
Bookmakers sans marge
Modele vs marche
Controle qualite
Controle
```

Exemple de prediction recente :

```text
PSG gagne       50.27%
Match nul       21.26%
Arsenal gagne   28.46%
```

La section `Modele vs marche` compare la probabilite du modele avec la probabilite bookmaker normalisee sans marge :

```text
Outcome        Model    Market   Edge
PSG gagne      50.27%   41.57%   +8.70 pp
Match nul      21.26%   26.90%   -5.64 pp
Arsenal gagne  28.46%   31.53%   -3.07 pp
```

La section `Controle qualite` signale les limites de donnees :

```text
Model artifact              [OK]
Final venue neutral         [OK]
Feature alignment           [OK]
Probabilities sum           [OK]
xG data available           [WARN] - neutral fallback used
Possession data available   [WARN] - neutral fallback used
H2H history available       [WARN] - neutral fallback used
```

Ces avertissements ne bloquent pas la prediction. Ils indiquent simplement que le modele utilise une valeur neutre parce que la source de donnees correspondante n'est pas encore disponible.

## Licence

Usage educatif / personnel uniquement. Respecte les conditions d'utilisation de football-data.org, Transfermarkt et toute source externe ajoutee.
