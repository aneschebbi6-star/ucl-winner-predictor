"""Affichage terminal homogène pour tout le pipeline."""

import sys
from typing import Optional

import pandas as pd

WIDTH = 70

# Phases LDC affichées (à partir des quarts de finale)
CL_FROM_QUARTERS_STAGES = frozenset({
    "QUARTER_FINALS",
    "SEMI_FINALS",
    "FINAL",
})

STAGE_LABELS = {
    "QUARTER_FINALS": "Quarts de finale",
    "SEMI_FINALS": "Demi-finales",
    "FINAL": "Finale",
    "LAST_16": "8es de finale",
    "PLAYOFFS": "Barrages",
    "LEAGUE_STAGE": "Phase de ligue",
    "GROUP_STAGE": "Phase de groupes",
}


def configure_console() -> None:
    """UTF-8 sur Windows pour emojis et accents."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass


def print_header(title: str, subtitle: Optional[str] = None) -> None:
    print("\n" + "=" * WIDTH)
    print(f"  {title}")
    if subtitle:
        print(f"  {subtitle}")
    print("=" * WIDTH)


def print_footer(message: str) -> None:
    print("\n" + "=" * WIDTH)
    print(f"  {message}")
    print("=" * WIDTH + "\n")


def print_section(title: str) -> None:
    print(f"\n{title}")
    print("-" * WIDTH)


def print_kv(label: str, value: str, indent: int = 2) -> None:
    pad = " " * indent
    print(f"{pad}• {label:<22} {value}")


def format_date(value) -> str:
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    return str(value)[:10]


def format_stage(stage: str) -> str:
    if not stage or (isinstance(stage, float) and pd.isna(stage)):
        return "—"
    return STAGE_LABELS.get(str(stage), str(stage).replace("_", " ").title())


def short_name(name: str, width: int = 22) -> str:
    text = str(name) if name else "?"
    return text if len(text) <= width else text[: width - 1] + "…"


def cl_from_quarters_mask(df: pd.DataFrame, stage_col: str = "Stage") -> pd.Series:
    if stage_col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[stage_col].isin(CL_FROM_QUARTERS_STAGES)


def filter_cl_from_quarters(df: pd.DataFrame, stage_col: str = "Stage") -> pd.DataFrame:
    if df.empty or stage_col not in df.columns:
        return df.iloc[0:0]
    return df[cl_from_quarters_mask(df, stage_col)].copy()


def count_hidden_cl_matches(df: pd.DataFrame) -> int:
    """Matchs LDC hors quarts → finale (non affichés)."""
    if df.empty or "Stage" not in df.columns:
        return 0
    cl_mask = df["Competition"].astype(str).str.contains("Champions", case=False, na=False)
    visible = cl_from_quarters_mask(df)
    return int((cl_mask & ~visible).sum())


def _format_score(row: pd.Series) -> str:
    if "Home" in row.index and "Away" in row.index:
        if pd.notna(row.get("Home_Goals")) and pd.notna(row.get("Away_Goals")):
            return f"{int(float(row['Home_Goals']))}-{int(float(row['Away_Goals']))}"
        return "à venir"
    if "Buts_Pour" in row.index:
        return f"{int(row['Buts_Pour'])}-{int(row['Buts_Contre'])}"
    return "—"


def print_cl_knockout_table(
    df: pd.DataFrame,
    *,
    team_perspective: bool = False,
    indent: int = 5,
) -> None:
    """Liste alignée des matchs LDC (quarts → finale)."""
    filtered = filter_cl_from_quarters(df)
    pad = " " * indent
    if filtered.empty:
        print(f"{pad}(aucun match depuis les quarts de finale)")
        return

    print(f"{pad}{'Date':<12} {'Phase':<18} {'Match':<36} {'Score':>7}")
    print(f"{pad}{'-' * 12} {'-' * 18} {'-' * 36} {'-' * 7}")

    sort_df = filtered.sort_values("Date")
    for _, row in sort_df.iterrows():
        date_str = format_date(row["Date"])
        phase = format_stage(row.get("Stage", ""))
        score = _format_score(row)

        if team_perspective and "Opponent" in row.index:
            venue = "🏠" if row.get("Venue") == "Home" else "✈️"
            icon = {"W": "✅", "D": "🟡", "L": "❌"}.get(row.get("Resultat"), " ")
            match = f"{icon} vs {short_name(row['Opponent'], 28)} {venue}"
        else:
            home = short_name(row.get("Home", "?"), 14)
            away = short_name(row.get("Away", "?"), 14)
            match = f"{home} — {away}"

        status = ""
        if row.get("Status") == "TIMED":
            status = " 📅"
        print(f"{pad}{date_str:<12} {phase:<18} {match:<36} {score:>7}{status}")
