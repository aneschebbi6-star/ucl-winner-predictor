"""Shared terminal rendering helpers for the ML pipeline."""

from __future__ import annotations

import sys
from typing import Iterable, Optional

import pandas as pd

WIDTH = 88

CL_FROM_QUARTERS_STAGES = frozenset(
    {
        "QUARTER_FINALS",
        "SEMI_FINALS",
        "FINAL",
    }
)

STAGE_LABELS = {
    "QUARTER_FINALS": "Quarter-finals",
    "SEMI_FINALS": "Semi-finals",
    "FINAL": "Final",
    "LAST_16": "Round of 16",
    "PLAYOFFS": "Playoffs",
    "LEAGUE_STAGE": "League phase",
    "GROUP_STAGE": "Group stage",
}


def configure_console() -> None:
    """Use UTF-8 when the terminal supports stream reconfiguration."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass


def print_header(title: str, subtitle: Optional[str] = None) -> None:
    print()
    print("=" * WIDTH)
    print(f" {title}")
    if subtitle:
        print(f" {subtitle}")
    print("=" * WIDTH)


def print_footer(message: str) -> None:
    print()
    print("=" * WIDTH)
    print(f" {message}")
    print("=" * WIDTH)
    print()


def print_section(title: str) -> None:
    print()
    print(title)
    print("-" * WIDTH)


def print_kv(label: str, value: str, indent: int = 2, label_width: int = 30) -> None:
    pad = " " * indent
    print(f"{pad}- {label:<{label_width}} {value}")


def print_status(label: str, ok: bool, detail: str = "") -> None:
    status = "OK" if ok else "WARN"
    suffix = f" - {detail}" if detail else ""
    print_kv(label, f"[{status}]{suffix}")


def print_table(headers: list[str], rows: Iterable[Iterable[object]], indent: int = 2) -> None:
    materialized = [[str(value) for value in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in materialized:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]

    pad = " " * indent
    header_line = "  ".join(header.ljust(width) for header, width in zip(headers, widths))
    separator = "  ".join("-" * width for width in widths)
    print(f"{pad}{header_line}")
    print(f"{pad}{separator}")
    for row in materialized:
        print(f"{pad}" + "  ".join(value.ljust(width) for value, width in zip(row, widths)))


def probability_bar(value: float, width: int = 28) -> str:
    value = max(0.0, min(1.0, float(value)))
    filled = int(round(value * width))
    return "[" + "#" * filled + "." * (width - filled) + f"] {value * 100:6.2f}%"


def print_probability_table(rows: Iterable[tuple[str, float]], indent: int = 2) -> None:
    formatted = [(label, probability_bar(probability)) for label, probability in rows]
    print_table(["Outcome", "Probability"], formatted, indent=indent)


def format_date(value) -> str:
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    return str(value)[:10]


def format_stage(stage: str) -> str:
    if not stage or (isinstance(stage, float) and pd.isna(stage)):
        return "-"
    return STAGE_LABELS.get(str(stage), str(stage).replace("_", " ").title())


def short_name(name: str, width: int = 22) -> str:
    text = str(name) if name else "?"
    return text if len(text) <= width else text[: width - 1] + "."


def cl_from_quarters_mask(df: pd.DataFrame, stage_col: str = "Stage") -> pd.Series:
    if stage_col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[stage_col].isin(CL_FROM_QUARTERS_STAGES)


def filter_cl_from_quarters(df: pd.DataFrame, stage_col: str = "Stage") -> pd.DataFrame:
    if df.empty or stage_col not in df.columns:
        return df.iloc[0:0]
    return df[cl_from_quarters_mask(df, stage_col)].copy()


def count_hidden_cl_matches(df: pd.DataFrame) -> int:
    if df.empty or "Stage" not in df.columns:
        return 0
    cl_mask = df["Competition"].astype(str).str.contains("Champions", case=False, na=False)
    visible = cl_from_quarters_mask(df)
    return int((cl_mask & ~visible).sum())


def _format_score(row: pd.Series) -> str:
    if "Home" in row.index and "Away" in row.index:
        if pd.notna(row.get("Home_Goals")) and pd.notna(row.get("Away_Goals")):
            return f"{int(float(row['Home_Goals']))}-{int(float(row['Away_Goals']))}"
        return "TBD"
    if "Buts_Pour" in row.index:
        return f"{int(row['Buts_Pour'])}-{int(row['Buts_Contre'])}"
    return "-"


def print_cl_knockout_table(
    df: pd.DataFrame,
    *,
    team_perspective: bool = False,
    indent: int = 2,
) -> None:
    filtered = filter_cl_from_quarters(df)
    if filtered.empty:
        print(" " * indent + "(no Champions League matches from quarter-finals onward)")
        return

    rows = []
    for _, row in filtered.sort_values("Date").iterrows():
        if team_perspective and "Opponent" in row.index:
            venue = row.get("Venue", "-")
            marker = {"W": "WIN", "D": "DRAW", "L": "LOSS"}.get(row.get("Resultat"), "-")
            match = f"{marker} vs {short_name(row['Opponent'], 28)} ({venue})"
        else:
            home = short_name(row.get("Home", "?"), 18)
            away = short_name(row.get("Away", "?"), 18)
            match = f"{home} vs {away}"
        rows.append([format_date(row["Date"]), format_stage(row.get("Stage", "")), match, _format_score(row)])

    print_table(["Date", "Stage", "Match", "Score"], rows, indent=indent)
