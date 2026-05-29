# Project Upgrade — PSG vs Arsenal (UCL Finale)

Document d'analyse et de feuille de route pour améliorer **la qualité prédictive** et **la maturité du projet**. Basé sur l'état actuel du dépôt (`src/`, `data/`, pipeline en 6 étapes).

---

## 1. État des lieux

### Ce qui fonctionne déjà

| Composant | Points forts |
|-----------|----------------|
| `scrapper.py` | API football-data.org, CSV structurés, blessures Transfermarkt, affichage terminal (`console.py`) |
| `dataset.py` | Cible 1N2 cohérente (perspective `Team`), features **pré-match** avec `shift()` (forme, buts, taux UCL) |
| `features.py` | Snapshot PSG/Arsenal + blessures dans `team_features.json` |
| `model.py` | Comparaison LR / RF / XGBoost, métriques LogLoss |
| `predict.py` | Inférence finale, lissage température + prior bookmaker, régression logistique régularisée |

### Architecture actuelle

```
scrapper → features (JSON) ─┐
         → dataset (CSV)  ──┼→ model (évaluation) → predict (finale)
```

### Limites structurelles (impact prédictif)

1. **Peu de données** : ~50 matchs / équipe / saison, ~100 lignes dans `train_dataset.csv` — insuffisant pour des modèles complexes.
2. **Deux pipelines de features** : rolling dans `dataset.py` vs agrégats dans `features.py` — pas toujours alignés à l'entraînement (`injury_impact_score` absent des CSV).
3. **`predict.py` entraîne sur train + test** : fuite d'information pour la « finale » ; les métriques de `model.py` ne reflètent pas ce choix.
4. **Variables catégorielles à risque** (`model.py`) : `Team`, `Opponent`, `Stage` peuvent mémoriser des patterns plutôt que généraliser.
5. **Finale = une seule ligne synthétique** : features adversaire (Arsenal) peu ou pas intégrées en différentiel dans `build_final_match`.
6. **Cotes bookmakers en dur** : pas de scraping, pas de marge retirée de façon systématique (sauf normalisation partielle dans `predict.py`).
7. **Pas de modèle persisté** : réentraînement à chaque `predict.py`, pas de versioning ni reproductibilité figée.

---

## 2. Améliorer la prédiction

### 2.1 Données (priorité haute)

| Action | Pourquoi | Comment |
|--------|----------|--------|
| **Historique multi-saisons** | Plus de finales / phases KO pour apprendre | Étendre `SEASON` ou boucler sur 2018–2025 via l'API |
| **Matchs LDC tiers** | Contexte « niveau finale » | Ajouter tous les matchs KO (quarts→finale) des saisons passées, pas seulement PSG/Arsenal |
| **Head-to-head structuré** | Confrontations directes | Features : nb victoires H2H, buts moyens H2H sur 5 dernières rencontres (`psg_arsenal_h2h.csv`) |
| **Cotes réelles** | Calibration marché | API odds (The Odds API, etc.) ou scraping ; stocker timestamp + closing line |
| **xG / tirs** | Signal plus fort que le score seul | FBref, Understat, StatsBomb open data — moyennes xG pour/contre sur 5/10 matchs |
| **Effectif jour J** | Variance finale | Aligner `injuries_impact.json` sur chaque date de match (pas seulement snapshot actuel) |

### 2.2 Features (priorité haute)

**Principe** : une ligne = **un match**, features calculées **strictement avant** le coup d'envoi (`shift` déjà bien amorcé dans `dataset.py`).

Features recommandées pour la finale :

```
# Différentiel (Team − Opponent)
diff_form_5, diff_xg_for_5, diff_xg_against_5
diff_rest_days, diff_ucl_experience

# Contexte match
is_neutral, is_final, stage_knockout_round
days_since_last_match, matches_last_14_days

# H2H
h2h_win_rate_team, h2h_goals_avg_team

# Marché (si disponible)
implied_prob_psg, implied_prob_draw, implied_prob_arsenal

# Blessures (par date)
team_injury_impact, opponent_injury_impact, diff_injury
```

**À faire dans le code** :

- Fusionner `features.py` dans `dataset.py` (une seule source de vérité).
- Pour `predict.py` : construire **deux vecteurs** (PSG et Arsenal) ou un vecteur **différentiel** symétrique, pas seulement les stats PSG.
- Exclure ou encoder proprement `Opponent` (target encoding avec validation temporelle, ou remplacer par features numériques adversaire).

### 2.3 Modélisation (priorité haute)

| Étape | Recommandation |
|-------|----------------|
| **Granularité** | Modèle **match-level** (1 ligne par match Home vs Away) plutôt que 2 lignes par match (perspective Team) — évite la corrélation et simplifie la cible 1N2. |
| **Validation** | `TimeSeriesSplit` ou split par **saison** ; jamais mélanger matchs futurs dans le train. |
| **Modèle production** | Régression logistique / XGBoost **peu profond** (`max_depth` 3–4) + régularisation forte sur petit N. |
| **Calibration** | `CalibratedClassifierCV` (isotonic ou sigmoid) sur fold de validation **postérieur** au train. |
| **Métriques** | LogLoss, Brier score, courbes de calibration ; pour le 1N2 : log-loss multiclasse + matrice de confusion. |
| **Baselines** | Probas implicites bookmakers ; modèle « constant » (fréquences historiques UCL). |

**Modèles alternatifs (moyen terme)** :

- **Poisson bivarié / Dixon-Coles** : prédire scorelines puis dériver 1N2 (meilleur pour les nuls).
- **Elo dynamique** par équipe (mis à jour après chaque match UCL + domestique).
- **Ensemble** : moyenne pondérée LR calibré + marché (stacking prudent).

### 2.4 Inférence finale (`predict.py`)

| Problème actuel | Amélioration |
|-----------------|--------------|
| Entraînement sur tout le dataset | Entraîner uniquement sur `train` ; réserver `test` pour évaluation finale |
| Lissage ad hoc (`temperature=1.7`) | Remplacer par calibration apprise (Platt/isotonic) + documenter |
| Prior bookmaker mélangé à postériori | Option A : modèle seul calibré ; Option B : ensemble explicite `α·modèle + (1−α)·marché` avec α validé |
| Pas d'intervalle d'incertitude | Bootstrap ou quantiles conformal prediction pour fourchettes PSG/Nul/Arsenal |

### 2.5 Évaluation spécifique « finale »

- Backtest : pour chaque finale UCL passée dans les données, entraîner **sans** cette finale, prédire, comparer au résultat.
- Mesurer **calibration** : si le modèle dit 45 % PSG, environ 45 % des cas similaires doivent gagner (sur bucket historiques).
- Comparer systématiquement **LogLoss modèle vs LogLoss cotes** (après retrait de marge).

---

## 3. Améliorer le projet (ingénierie & produit)

### 3.1 Structure du code

```
src/
├── config.py              # SEASON, paths, hyperparamètres
├── data/
│   ├── scraper.py         # (renommer scrapper → scraper optionnel)
│   └── loaders.py         # read CSV, validate schéma
├── features/
│   ├── rolling.py         # add_prematch_features
│   └── final.py           # build_final_match_vector
├── models/
│   ├── train.py           # logique actuelle model.py
│   ├── calibrate.py
│   └── predict.py
├── evaluation/
│   └── metrics.py
└── console.py
```

**Bénéfices** : imports clairs, tests unitaires par module, un seul point pour les chemins `data/`.

### 3.2 Reproductibilité & MLOps léger

| Élément | Action |
|---------|--------|
| Artefacts | Sauver `model.joblib`, `preprocessor.joblib`, `feature_columns.json` dans `models/` |
| Config | `config.yaml` ou `.env` + `config.py` (hyperparams, split_date) |
| Seeds | `random_state=42` partout, loggué |
| Pipeline | `Makefile` ou `scripts/run_all.ps1` : scrape → dataset → train → predict |
| Dépendances | Épingler versions dans `requirements.txt` (`pandas==2.x`) |

### 3.3 Qualité & tests

```python
# tests/test_features.py
def test_prematch_features_no_leakage():
    # but du match N ne doit pas influencer features du match N

# tests/test_dataset.py
def test_y_target_mapping():
    assert map_result("W") == 0  # perspective Team
```

- **pytest** sur : mapping cible, shift rolling, filtre LDC quarts+, chargement CSV.
- **Validation schéma** : `pandera` ou checks manuels (colonnes obligatoires, dates triées).

### 3.4 Documentation

| Fichier | Contenu |
|---------|---------|
| `README.md` | Déjà présent — ajouter section « Calibration » et « Limites connues » |
| `project_upgrade.md` | Ce document |
| `docs/model_card.md` | Données, biais, métriques, usage prévu (éducatif, pas paris) |
| `docs/data_dictionary.md` | Description colonne par colonne des CSV |

### 3.5 Données & Git

- Ajouter `data/raw/.gitkeep` + documenter ce qui est versionné vs régénéré.
- Ne pas committer `.env` (déjà dans `.gitignore`).
- Option : **DVC** ou scripts pour régénérer `data/` depuis l'API.

### 3.6 UX terminal (déjà bien avancé)

- Réintégrer le bloc **interprétation professionnelle** (surconfiance, limites) si supprimé lors du passage au lissage.
- Afficher **probas brutes / calibrées / marché** sur 3 lignes dans un même tableau.
- Commande unique : `python -m src.cli predict --final`.

### 3.7 Évolutions produit (optionnel)

- Export JSON des prédictions (`outputs/prediction_YYYYMMDD.json`).
- Notebook `notebooks/01_calibration_analysis.ipynb` pour graphiques.
- API FastAPI minimale `POST /predict` pour démo portfolio.

---

## 4. Risques de data leakage (à corriger en priorité)

| Risque | Où | Correction |
|--------|-----|------------|
| Train + test fusionnés pour la finale | `predict.py` `load_all_data()` | Séparer strictement ; option « production » avec modèle pré-entraîné |
| `Team` / `Opponent` one-hot | `model.py` | Retirer ou target encoding avec CV temporelle |
| `Stage=FINAL` rare | encodage | Feature binaire `is_knockout` / `is_final` |
| Defaults globaux sur premières lignes | `add_prematch_features` | Defaults par équipe ou médiane saison N-1 uniquement |
| Features.json snapshot actuel | `build_final_match` | Recalculer features à la date de la finale depuis l'historique CSV |

---

## 5. Roadmap priorisée

### Phase 1 — Quick wins (1–3 jours)

- [ ] Entraîner `predict` **uniquement sur train** ; évaluer sur test.
- [ ] Sauvegarder le pipeline sklearn (`joblib`) après `model.py`.
- [ ] Ajouter features **différentielles** Arsenal dans `build_final_match`.
- [ ] Intégrer `injury_impact_score` dans `dataset.py` (jointure par équipe).
- [ ] `CalibratedClassifierCV` sur la régression logistique.
- [ ] Unifier affichage : brutes / calibrées / marché + limites du modèle.

### Phase 2 — Données & validation (1–2 semaines)

- [ ] Historique 3–5 saisons LDC (KO + league phase pour Elo).
- [ ] Features H2H + repos entre matchs.
- [ ] Backtest finales passées.
- [ ] Scraping ou API cotes avec marge retirée.
- [ ] Tests pytest sur features et split.

### Phase 3 — Maturité pro (2–4 semaines)

- [ ] Refactor modules (`src/data`, `src/features`, `src/models`).
- [ ] Modèle Poisson ou Elo en baseline comparée.
- [ ] Model card + notebook calibration.
- [ ] CI GitHub Actions : lint + tests sur PR.

---

## 6. Synthèse

| Objectif | Action la plus rentable |
|----------|------------------------|
| **Probas réalistes** | Calibration sklearn + ne plus entraîner sur le test + différentiel PSG−Arsenal |
| **Meilleure précision** | Plus de données historiques + xG + H2H + modèle match-level |
| **Projet crédible** | Modèle sauvegardé, tests, config unique, doc `model_card`, séparation train/test stricte |
| **Valeur métier** | Comparer LogLoss au marché ; value bet seulement sur proba **calibrée** |

Le projet a déjà une **bonne base** (API, pipeline, features pré-match avec `shift`, console soignée). Le saut de qualité viendra surtout de **moins de fuite de données**, **plus d'historique**, **calibration mesurée** et **un seul pipeline de features** jusqu'à la finale.

---

*Généré pour le dépôt `champions predection` — à mettre à jour après chaque refonte majeure du pipeline.*
