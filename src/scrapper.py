"""
══════════════════════════════════════════════════════════════════════
  🏆 SCRAPER — DONNÉES RÉELLES — PSG vs ARSENAL
  Source : football-data.org API v4 (gratuite)
  
  Utilisation :
    1. Obtenir une clé API gratuite : https://www.football-data.org/client/register
    2. Mettre la clé dans .env : FOOTBALL_API_KEY=votre_cle
    3. Exécuter : python scrapper.py
══════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
import os
import time
import json
import requests
from dotenv import load_dotenv
from datetime import datetime

# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════
load_dotenv()

SEASON = int(os.getenv("SEASON", 2025))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "data/raw")
API_KEY = os.getenv("FOOTBALL_API_KEY", "")

# IDs football-data.org
PSG_TEAM_ID = int(os.getenv("PSG_TEAM_ID", 524))
ARSENAL_TEAM_ID = int(os.getenv("ARSENAL_TEAM_ID", 57))

# Compétitions
COMPETITIONS = {
    "CL": "Champions League",
    "FL1": "Ligue 1",
    "PL": "Premier League",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# CLASSE PRINCIPALE : FootballDataScraper
# ══════════════════════════════════════════════════════════════
class FootballDataScraper:
    """
    Collecte de données réelles via football-data.org API v4.
    Plan gratuit : 10 requêtes/minute, CL + top leagues incluses.
    """

    BASE_URL = "https://api.football-data.org/v4"

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError(
                "❌ Clé API manquante !\n"
                "   1. Inscription gratuite : https://www.football-data.org/client/register\n"
                "   2. Ajoutez dans .env : FOOTBALL_API_KEY=votre_cle"
            )
        self.headers = {"X-Auth-Token": api_key}
        self._request_count = 0
        self._last_request_time = 0

    def _rate_limit(self):
        """Respecte la limite de 10 requêtes/minute du plan gratuit."""
        self._request_count += 1
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < 6:  # ~10 req/min = 1 req/6s
            wait = 6 - elapsed
            print(f"   ⏳ Rate limit — pause {wait:.1f}s...")
            time.sleep(wait)
        self._last_request_time = time.time()

    def _get(self, endpoint: str, params: dict = None) -> dict:
        """Effectue une requête GET avec gestion d'erreurs."""
        self._rate_limit()
        url = f"{self.BASE_URL}{endpoint}"

        print(f"   🌐 GET {endpoint}")

        try:
            resp = requests.get(url, headers=self.headers, params=params, timeout=30)

            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                print("   ⚠️  Rate limit atteint — pause 60s...")
                time.sleep(60)
                return self._get(endpoint, params)
            elif resp.status_code == 403:
                print(f"   ❌ Accès refusé (403) — vérifiez votre clé API")
                print(f"      Réponse : {resp.text[:200]}")
                return {}
            else:
                print(f"   ❌ Erreur HTTP {resp.status_code} : {resp.text[:200]}")
                return {}

        except requests.exceptions.RequestException as e:
            print(f"   ❌ Erreur réseau : {e}")
            return {}

    # ─────────────────────────────────────────────────────
    # Matchs d'une équipe
    # ─────────────────────────────────────────────────────
    def get_team_matches(
        self, team_id: int, season: int, competitions: str = None
    ) -> pd.DataFrame:
        """
        Récupère tous les matchs terminés d'une équipe pour une saison.
        
        Args:
            team_id: ID de l'équipe (524=PSG, 57=Arsenal)
            season: Année de début de saison (2025 pour 2025/26)
            competitions: Codes séparés par virgules (ex: "CL,FL1")
        """
        params = {"season": season, "status": "FINISHED"}
        if competitions:
            params["competitions"] = competitions

        data = self._get(f"/teams/{team_id}/matches", params)
        if not data or "matches" not in data:
            print(f"   ⚠️  Aucun match trouvé pour team_id={team_id}")
            return pd.DataFrame()

        matches = []
        for m in data["matches"]:
            home_team = m["homeTeam"]["name"]
            away_team = m["awayTeam"]["name"]
            home_goals = m["score"]["fullTime"]["home"]
            away_goals = m["score"]["fullTime"]["away"]

            # Déterminer si c'est notre équipe à domicile ou extérieur
            is_home = m["homeTeam"]["id"] == team_id
            team_name = home_team if is_home else away_team
            opponent = away_team if is_home else home_team

            goals_for = home_goals if is_home else away_goals
            goals_against = away_goals if is_home else home_goals

            if goals_for > goals_against:
                result = "W"
            elif goals_for == goals_against:
                result = "D"
            else:
                result = "L"

            # Compétition
            comp_code = m["competition"]["code"]
            comp_name = COMPETITIONS.get(comp_code, m["competition"]["name"])

            matches.append(
                {
                    "Date": m["utcDate"][:10],
                    "Competition": comp_name,
                    "Comp_Code": comp_code,
                    "Matchday": m.get("matchday", None),
                    "Home": home_team,
                    "Away": away_team,
                    "Home_Goals": home_goals,
                    "Away_Goals": away_goals,
                    "Team": team_name,
                    "Opponent": opponent,
                    "Venue": "Home" if is_home else "Away",
                    "Buts_Pour": goals_for,
                    "Buts_Contre": goals_against,
                    "Resultat": result,
                    "Stage": m.get("stage", ""),
                    "Group": m.get("group", ""),
                }
            )

        df = pd.DataFrame(matches)
        if not df.empty:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date").reset_index(drop=True)
        return df

    # ─────────────────────────────────────────────────────
    # Matchs d'une compétition
    # ─────────────────────────────────────────────────────
    def get_competition_matches(
        self, comp_code: str, season: int
    ) -> pd.DataFrame:
        """Récupère tous les matchs d'une compétition."""
        params = {"season": season}
        data = self._get(f"/competitions/{comp_code}/matches", params)

        if not data or "matches" not in data:
            return pd.DataFrame()

        matches = []
        for m in data["matches"]:
            home_goals = (
                m["score"]["fullTime"]["home"]
                if m["score"]["fullTime"]["home"] is not None
                else None
            )
            away_goals = (
                m["score"]["fullTime"]["away"]
                if m["score"]["fullTime"]["away"] is not None
                else None
            )

            matches.append(
                {
                    "Date": m["utcDate"][:10],
                    "Competition": COMPETITIONS.get(comp_code, comp_code),
                    "Matchday": m.get("matchday"),
                    "Stage": m.get("stage", ""),
                    "Home": m["homeTeam"]["name"],
                    "Away": m["awayTeam"]["name"],
                    "Home_Goals": home_goals,
                    "Away_Goals": away_goals,
                    "Status": m["status"],
                }
            )

        df = pd.DataFrame(matches)
        if not df.empty:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date").reset_index(drop=True)
        return df

    # ─────────────────────────────────────────────────────
    # Classement
    # ─────────────────────────────────────────────────────
    def get_standings(self, comp_code: str, season: int) -> pd.DataFrame:
        """Récupère le classement d'une compétition."""
        params = {"season": season}
        data = self._get(f"/competitions/{comp_code}/standings", params)

        if not data or "standings" not in data:
            return pd.DataFrame()

        rows = []
        for standing_type in data["standings"]:
            for entry in standing_type.get("table", []):
                rows.append(
                    {
                        "Position": entry["position"],
                        "Team": entry["team"]["name"],
                        "Team_ID": entry["team"]["id"],
                        "Played": entry["playedGames"],
                        "Won": entry["won"],
                        "Draw": entry["draw"],
                        "Lost": entry["lost"],
                        "GF": entry["goalsFor"],
                        "GA": entry["goalsAgainst"],
                        "GD": entry["goalDifference"],
                        "Points": entry["points"],
                        "Type": standing_type.get("type", ""),
                        "Group": standing_type.get("group", ""),
                    }
                )

        return pd.DataFrame(rows)

    # ─────────────────────────────────────────────────────
    # Head-to-Head
    # ─────────────────────────────────────────────────────
    def get_head_to_head(
        self, team1_id: int, team2_id: int, limit: int = 20
    ) -> pd.DataFrame:
        """
        Récupère les confrontations directes entre deux équipes.
        Utilise l'endpoint /teams/{id}/matches et filtre.
        """
        # Récupérer les matchs récents de l'équipe 1
        params = {"limit": 200, "status": "FINISHED"}
        data = self._get(f"/teams/{team1_id}/matches", params)

        if not data or "matches" not in data:
            return pd.DataFrame()

        h2h_matches = []
        for m in data["matches"]:
            home_id = m["homeTeam"]["id"]
            away_id = m["awayTeam"]["id"]

            # Garder seulement les matchs entre les deux équipes
            if not ({home_id, away_id} == {team1_id, team2_id}):
                continue

            home_goals = m["score"]["fullTime"]["home"]
            away_goals = m["score"]["fullTime"]["away"]

            # Perspective de team1
            is_home = home_id == team1_id
            gf = home_goals if is_home else away_goals
            ga = away_goals if is_home else home_goals

            if gf > ga:
                result = "W"
            elif gf == ga:
                result = "D"
            else:
                result = "L"

            h2h_matches.append(
                {
                    "Date": m["utcDate"][:10],
                    "Competition": m["competition"]["name"],
                    "Home": m["homeTeam"]["name"],
                    "Away": m["awayTeam"]["name"],
                    "Home_Goals": home_goals,
                    "Away_Goals": away_goals,
                    "GF_Team1": gf,
                    "GA_Team1": ga,
                    "Result_Team1": result,
                }
            )

        df = pd.DataFrame(h2h_matches)
        if not df.empty:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date", ascending=False).head(limit).reset_index(drop=True)
        return df

    # ─────────────────────────────────────────────────────
    # Blessures et Suspensions (Scraping de Transfermarkt)
    # ─────────────────────────────────────────────────────
    def get_injuries(self) -> dict:
        """
        Récupère la liste des blessés/suspendus depuis Transfermarkt.
        Calcule un Injury Impact Score approximatif.
        """
        import requests
        from bs4 import BeautifulSoup
        
        print("   🚑 Scraping des données de blessures et suspensions (Transfermarkt)...")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        teams_urls = {
            "PSG": "https://www.transfermarkt.com/paris-saint-germain/sperrenundverletzungen/verein/583",
            "Arsenal": "https://www.transfermarkt.com/fc-arsenal/sperrenundverletzungen/verein/11"
        }
        
        injuries_data = {}
        
        for team, url in teams_urls.items():
            print(f"      📡 Scraping {team}...")
            injuries_data[team] = {"players": []}
            try:
                r = requests.get(url, headers=headers, timeout=15)
                if r.status_code == 200:
                    soup = BeautifulSoup(r.text, 'html.parser')
                    table = soup.find('table', {'class': 'items'})
                    if table:
                        rows = table.find('tbody').find_all('tr', recursive=False)
                        for row in rows:
                            cols = row.find_all('td')
                            if len(cols) == 9: # Ligne d'un joueur
                                name = cols[2].text.strip()
                                injury = cols[5].text.strip()
                                
                                # Assignation basique de l'importance (3 par défaut)
                                importance = 3
                                if "cruciate" in injury.lower() or "achilles" in injury.lower():
                                    importance = 4
                                
                                injuries_data[team]["players"].append({
                                    "name": name,
                                    "type": injury,
                                    "importance": importance
                                })
            except Exception as e:
                print(f"      ❌ Erreur scraping {team} : {e}")
                
        # Calcul de l'Injury Impact Score
        for team, data in injuries_data.items():
            impact_score = sum(p["importance"] for p in data["players"])
            data["impact_score"] = impact_score
            
        return injuries_data


# ══════════════════════════════════════════════════════════════
# AFFICHAGE
# ══════════════════════════════════════════════════════════════
def print_team_summary(df: pd.DataFrame, team_name: str):
    """Affiche un résumé des performances réelles."""
    if df.empty:
        print(f"  ⚠️  Aucun match pour {team_name}")
        return

    wins = (df["Resultat"] == "W").sum()
    draws = (df["Resultat"] == "D").sum()
    losses = (df["Resultat"] == "L").sum()

    print(f"\n  📊 {team_name}")
    print(f"     • Matchs : {len(df)}")
    print(f"     • Victoires : {wins} | Nuls : {draws} | Défaites : {losses}")
    print(f"     • Buts marqués : {df['Buts_Pour'].sum()}")
    print(f"     • Buts encaissés : {df['Buts_Contre'].sum()}")
    print(f"     • Différence : {df['Buts_Pour'].sum() - df['Buts_Contre'].sum():+d}")

    # Par compétition
    print(f"\n     Par compétition :")
    for comp, group in df.groupby("Competition"):
        comp_wins = (group["Resultat"] == "W").sum()
        print(f"       • {comp}: {len(group)} matchs ({comp_wins}V / {len(group) - comp_wins})")

    # Par venue (domicile/extérieur)
    if "Venue" in df.columns:
        print(f"\n     Par lieu :")
        for venue, group in df.groupby("Venue"):
            v_wins = (group["Resultat"] == "W").sum()
            label = "🏠 Domicile" if venue == "Home" else "✈️  Extérieur"
            print(f"       • {label}: {len(group)} matchs ({v_wins}V)")

    # Derniers matchs
    print(f"\n     5 derniers matchs :")
    for _, row in df.tail(5).iterrows():
        date_str = row["Date"].strftime("%Y-%m-%d") if hasattr(row["Date"], "strftime") else str(row["Date"])[:10]
        opponent = row.get("Opponent", "?")
        venue_icon = "🏠" if row.get("Venue") == "Home" else "✈️"
        score = f"{row['Buts_Pour']}-{row['Buts_Contre']}"
        result_icon = {"W": "✅", "D": "🟡", "L": "❌"}.get(row["Resultat"], "?")
        print(
            f"       {result_icon} {date_str} | {row['Competition']:20s} | "
            f"vs {opponent:25s} | {score} {venue_icon}"
        )


def print_h2h_summary(df: pd.DataFrame, team1: str, team2: str):
    """Affiche le résumé des confrontations directes."""
    if df.empty:
        print(f"  ⚠️  Aucune confrontation directe trouvée entre {team1} et {team2}")
        return

    wins = (df["Result_Team1"] == "W").sum()
    draws = (df["Result_Team1"] == "D").sum()
    losses = (df["Result_Team1"] == "L").sum()

    print(f"\n  ⚔️  HEAD-TO-HEAD : {team1} vs {team2}")
    print(f"     • Total : {len(df)} matchs")
    print(f"     • {team1} : {wins}V | {draws}N | {losses}D")
    print(f"     • Buts {team1} : {df['GF_Team1'].sum()} | Buts {team2} : {df['GA_Team1'].sum()}")

    print(f"\n     Détail :")
    for _, row in df.iterrows():
        date_str = row["Date"].strftime("%Y-%m-%d") if hasattr(row["Date"], "strftime") else str(row["Date"])[:10]
        result_icon = {"W": "✅", "D": "🟡", "L": "❌"}.get(row["Result_Team1"], "?")
        print(
            f"       {result_icon} {date_str} | {row['Home']} {row['Home_Goals']}-{row['Away_Goals']} {row['Away']} "
            f"| {row['Competition']}"
        )


# ══════════════════════════════════════════════════════════════
# SAUVEGARDE
# ══════════════════════════════════════════════════════════════
def save_data(df: pd.DataFrame, filename: str) -> bool:
    """Sauvegarde en CSV."""
    if df.empty:
        print(f"   ⚠️  {filename} — aucune donnée à sauvegarder")
        return False
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index=False)
    print(f"   💾 {filename} ({len(df)} lignes)")
    return True


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("  🏆 COLLECTE DONNÉES RÉELLES — PSG vs ARSENAL — SAISON " + str(SEASON))
    print("=" * 70)
    print(f"  📡 Source : football-data.org API v4")
    print(f"  📅 Saison : {SEASON}/{SEASON + 1}")
    print("=" * 70 + "\n")

    try:
        scraper = FootballDataScraper(API_KEY)

        # ── PSG ──────────────────────────────────────────
        print("🔵 PSG\n")
        psg = scraper.get_team_matches(
            PSG_TEAM_ID, SEASON, competitions="CL,FL1"
        )
        save_data(psg, "psg_matches.csv")
        print_team_summary(psg, "PSG")

        print("\n" + "-" * 70)

        # ── Arsenal ──────────────────────────────────────
        print("\n🔴 ARSENAL\n")
        arsenal = scraper.get_team_matches(
            ARSENAL_TEAM_ID, SEASON, competitions="CL,PL"
        )
        save_data(arsenal, "arsenal_matches.csv")
        print_team_summary(arsenal, "Arsenal")

        print("\n" + "-" * 70)

        # ── Combiné ──────────────────────────────────────
        if not psg.empty and not arsenal.empty:
            combined = pd.concat([psg, arsenal], ignore_index=True)
            save_data(combined, "psg_arsenal_combined.csv")
            print(f"\n📊 Données combinées : {len(combined)} matchs")

        # ── Head-to-Head ─────────────────────────────────
        print("\n⚔️  Confrontations directes\n")
        h2h = scraper.get_head_to_head(PSG_TEAM_ID, ARSENAL_TEAM_ID, limit=15)
        save_data(h2h, "psg_arsenal_h2h.csv")
        print_h2h_summary(h2h, "PSG", "Arsenal")

        # ── Classements ──────────────────────────────────
        print("\n\n📊 Classements\n")
        for comp_code, comp_name in [("FL1", "Ligue 1"), ("PL", "Premier League")]:
            standings = scraper.get_standings(comp_code, SEASON)
            if not standings.empty:
                save_data(standings, f"{comp_code.lower()}_standings.csv")
                # Afficher le top 5
                total = standings[standings["Type"] == "TOTAL"]
                if not total.empty:
                    print(f"\n  🏆 {comp_name} — Top 5 :")
                    for _, row in total.head(5).iterrows():
                        print(
                            f"     {row['Position']:2d}. {row['Team']:25s} | "
                            f"{row['Points']}pts | {row['Won']}V {row['Draw']}N {row['Lost']}D | "
                            f"{row['GF']}-{row['GA']} ({row['GD']:+d})"
                        )

        # ── Champions League matches ─────────────────────
        print("\n\n🏆 Champions League — Tous les matchs\n")
        cl_matches = scraper.get_competition_matches("CL", SEASON)
        if not cl_matches.empty:
            save_data(cl_matches, "cl_all_matches.csv")

            # Filtrer les matchs terminés
            finished = cl_matches[cl_matches["Status"] == "FINISHED"]
            print(f"   Total : {len(cl_matches)} matchs ({len(finished)} terminés)")

            # Matchs à élimination directe
            knockout = cl_matches[cl_matches["Stage"] != "GROUP_STAGE"]
            if not knockout.empty:
                print(f"\n  🔥 Phase finale CL ({len(knockout)} matchs) :")
                for _, row in knockout.iterrows():
                    date_str = row["Date"].strftime("%Y-%m-%d") if hasattr(row["Date"], "strftime") else str(row["Date"])[:10]
                    if pd.notna(row["Home_Goals"]):
                        score = f"{int(float(row['Home_Goals']))}-{int(float(row['Away_Goals']))}"
                    else:
                        score = "à venir"
                    stage_short = row["Stage"].replace("_", " ").title()
                    print(
                        f"     {date_str} | {stage_short:20s} | "
                        f"{row['Home']:25s} {score:>5s} {row['Away']}"
                    )

        # ── Blessures et Suspensions ─────────────────────
        print("\n\n🚑 Blessures et Suspensions\n")
        injuries = scraper.get_injuries()
        
        os.makedirs("data/processed", exist_ok=True)
        injuries_path = "data/processed/injuries_impact.json"
        with open(injuries_path, "w", encoding="utf-8") as f:
            json.dump(injuries, f, indent=2, ensure_ascii=False)
        print(f"   💾 Sauvegardé : {injuries_path}")
        
        for team, data in injuries.items():
            print(f"\n  🤕 {team} — Impact Score : {data['impact_score']}")
            for p in data["players"]:
                print(f"     • {p['name']} ({p['type']}) - Importance: {p['importance']}/5")

        # ── Résumé final ─────────────────────────────────
        print("\n" + "=" * 70)
        print("  ✅ DONNÉES RÉELLES COLLECTÉES")
        print(f"  📁 Dossier : {OUTPUT_DIR}")
        print("=" * 70)

        # Lister les fichiers générés
        csv_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".csv")]
        print(f"\n  📄 Fichiers générés ({len(csv_files)}) :")
        for f in sorted(csv_files):
            size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
            print(f"     • {f} ({size:,} bytes)")

        print("\n  ➡️  Prochaine étape : python features.py")
        print("=" * 70 + "\n")

    except ValueError as e:
        print(f"\n{e}")
        print("\n" + "=" * 70)
        print("  💡 INSTRUCTIONS :")
        print("     1. Allez sur https://www.football-data.org/client/register")
        print("     2. Créez un compte gratuit")
        print("     3. Copiez votre clé API")
        print("     4. Ajoutez dans .env : FOOTBALL_API_KEY=votre_cle")
        print("     5. Relancez : python scrapper.py")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n❌ Erreur inattendue : {e}")
        import traceback
        traceback.print_exc()
