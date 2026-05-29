"""Thème visuel UCL — Streamlit (Mode Clair)."""

from __future__ import annotations

import streamlit as st

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,600;0,9..40,700;1,9..40,400&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', system-ui, sans-serif;
}

.block-container {
    padding-top: 1.5rem;
    max-width: 1200px;
}

.hero {
    background: linear-gradient(135deg, #e8f1fc 0%, #d4e5f7 45%, #f0f4fa 100%);
    border: 2px solid #1f77b4;
    border-radius: 16px;
    padding: 1.75rem 2rem;
    margin-bottom: 1.5rem;
    text-align: center;
}
.hero-title {
    font-size: 1.75rem;
    font-weight: 700;
    color: #0f172a;
    margin: 0 0 0.25rem 0;
    letter-spacing: -0.02em;
}
.hero-sub {
    color: #475569;
    font-size: 0.95rem;
    margin: 0;
}
.hero-vs {
    display: inline-block;
    margin: 1rem 0;
    padding: 0.35rem 1rem;
    background: #e3f2fd;
    color: #1565c0;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.85rem;
}
.disclaimer {
    background: #fef3c7;
    border-left: 4px solid #f59e0b;
    padding: 0.75rem 1rem;
    border-radius: 0 8px 8px 0;
    color: #78350f;
    font-size: 0.88rem;
    margin: 1rem 0;
}
.prob-card {
    background: linear-gradient(180deg, #f8fafc 0%, #f0f2f6 100%);
    border: 2px solid #cbd5e1;
    border-radius: 14px;
    padding: 1.25rem 1rem;
    text-align: center;
    min-height: 140px;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.prob-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(31, 119, 180, 0.15);
}
.prob-card.winner {
    border-color: #1f77b4;
    box-shadow: 0 0 0 1px rgba(31, 119, 180, 0.4), 0 8px 32px rgba(31, 119, 180, 0.12);
    background: linear-gradient(180deg, #e3f2fd 0%, #f0f4fa 100%);
}
.prob-label {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #64748b;
    margin-bottom: 0.35rem;
}
.prob-value {
    font-size: 2.25rem;
    font-weight: 700;
    color: #0f172a;
    line-height: 1.1;
}
.prob-badge {
    display: inline-block;
    margin-top: 0.5rem;
    font-size: 0.75rem;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    background: #dbeafe;
    color: #1565c0;
}
.confidence-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.9rem;
    margin: 0.5rem 0 1rem 0;
}
.conf-high { background: #fee2e2; color: #dc2626; }
.conf-med { background: #fef3c7; color: #d97706; }
.conf-low { background: #dcfce7; color: #16a34a; }
.section-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #0f172a;
    margin: 1.5rem 0 0.75rem 0;
    padding-bottom: 0.35rem;
    border-bottom: 2px solid #cbd5e1;
}
.check-ok { color: #16a34a; }
.check-warn { color: #d97706; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
}
[data-testid="stSidebar"] .block-container {
    padding-top: 1rem;
}
</style>
"""


def inject_theme() -> None:
    st.markdown(THEME_CSS, unsafe_allow_html=True)

