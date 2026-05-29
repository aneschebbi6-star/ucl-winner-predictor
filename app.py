"""
Interface Streamlit — Finale Ligue des Champions (PSG vs Arsenal).
Lancer : streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_option_menu import option_menu

sys.path.insert(0, str(Path(__file__).parent / "src"))

from ui.helpers import (  # noqa: E402
    OUTCOME_LABELS,
    PredictionBundle,
    confidence_info,
    feature_rows,
    load_prediction_bundle,
)
from ui.styles import inject_theme  # noqa: E402

st.set_page_config(
    page_title="UCL Final Predictor",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = [
    "Accueil",
    "Analyse",
    "Features",
    "Qualité",
    "À propos",
]


@st.cache_resource(show_spinner="Chargement du modèle…")
def cached_bundle() -> PredictionBundle | None:
    return load_prediction_bundle()


def render_hero(bundle: PredictionBundle) -> None:
    fm = bundle.config["final_match"]
    st.markdown(
        f"""
        <div class="hero">
            <p class="hero-sub">Ligue des Champions · Saison 2025/2026 · Finale</p>
            <h1 class="hero-title">Paris Saint-Germain vs Arsenal</h1>
            <span class="hero-vs">Terrain neutre · {fm.get("date", "—")}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prob_card(class_id: int, prob: float, is_winner: bool) -> str:
    short, long_label, _color = OUTCOME_LABELS[class_id]
    winner_class = " winner" if is_winner else ""
    badge = "Prédiction" if is_winner else ""
    badge_html = f'<span class="prob-badge">{badge}</span>' if badge else ""
    return f"""
    <div class="prob-card{winner_class}">
        <div class="prob-label">{short}</div>
        <div class="prob-value">{prob * 100:.1f}%</div>
        <div style="color:#94a3b8;font-size:0.85rem;">{long_label}</div>
        {badge_html}
    </div>
    """


def page_accueil(bundle: PredictionBundle) -> None:
    ordered = bundle.ordered
    pred = bundle.predicted_class
    conf_label, conf_class, conf_tip = confidence_info(ordered)

    cols = st.columns(3)
    for col, class_id in zip(cols, [0, 1, 2]):
        with col:
            st.markdown(render_prob_card(class_id, ordered[class_id], class_id == pred), unsafe_allow_html=True)

    st.markdown(
        f'<span class="confidence-pill {conf_class}">● {conf_label}</span>',
        unsafe_allow_html=True,
    )
    st.caption(conf_tip)

    st.markdown('<p class="section-title">Répartition des probabilités (modèle calibré)</p>', unsafe_allow_html=True)
    labels = [OUTCOME_LABELS[i][0] for i in range(3)]
    colors = [OUTCOME_LABELS[i][2] for i in range(3)]

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=ordered * 100,
                marker_color=colors,
                text=[f"{p:.1f}%" for p in ordered * 100],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        template="plotly",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=320,
        margin=dict(l=20, r=20, t=30, b=40),
        yaxis_title="Probabilité (%)",
        yaxis=dict(range=[0, max(ordered.max() * 100 + 15, 55)]),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="section-title">Modèle vs marché (cotes sans marge)</p>', unsafe_allow_html=True)
    market = np.array([bundle.bookmaker["team"], bundle.bookmaker["draw"], bundle.bookmaker["opponent"]])
    compare = pd.DataFrame(
        {
            "Issue": labels,
            "Modèle": (ordered * 100).round(1),
            "Marché": (market * 100).round(1),
            "Écart (pp)": ((ordered - market) * 100).round(1),
        }
    )
    st.dataframe(compare, use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="disclaimer">
        <strong>Avertissement</strong> — Outil à visée analytique et portfolio ML.
        Les probabilités sont calibrées sur l'historique disponible ; une finale reste un événement à forte variance.
        Ce n'est pas un conseil de pari.
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_analyse(bundle: PredictionBundle) -> None:
    ordered = bundle.ordered
    market = bundle.bookmaker
    labels = [OUTCOME_LABELS[i][1] for i in range(3)]
    edges = {
        "PSG": ordered[0] - market["team"],
        "Nul": ordered[1] - market["draw"],
        "Arsenal": ordered[2] - market["opponent"],
    }

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Issue favorite (modèle)", labels[int(np.argmax(ordered))])
        st.metric("Probabilité max.", f"{ordered.max() * 100:.1f}%")
    with c2:
        best = max(edges, key=edges.get)
        st.metric("Plus grand écart vs marché", f"{best} ({edges[best] * 100:+.1f} pp)")

    fig = go.Figure()
    x = [OUTCOME_LABELS[i][0] for i in range(3)]
    fig.add_trace(
        go.Bar(name="Modèle", x=x, y=ordered * 100, marker_color="#3b82f6", opacity=0.9)
    )
    fig.add_trace(
        go.Bar(
            name="Marché",
            x=x,
            y=[market["team"], market["draw"], market["opponent"]],
            marker_color="#94a3b8",
            opacity=0.75,
        )
    )
    fig.update_layout(
        barmode="group",
        template="plotly",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        yaxis_title="%",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Analyse value (indicative)")
    for name, edge, class_id in [
        ("PSG", edges["PSG"], 0),
        ("Nul", edges["Nul"], 1),
        ("Arsenal", edges["Arsenal"], 2),
    ]:
        if edge > 0.03:
            st.success(f"**{name}** : le modèle est au-dessus du marché de {edge * 100:+.1f} pp")
        elif edge < -0.03:
            st.info(f"**{name}** : le marché est au-dessus du modèle de {abs(edge) * 100:.1f} pp")
        else:
            st.caption(f"**{name}** : alignement modèle / marché ({edge * 100:+.1f} pp)")


def page_features(bundle: PredictionBundle) -> None:
    rows = feature_rows(bundle.final_vector)

    if not rows:
        st.warning("Aucune feature disponible.")
        return

    df = pd.DataFrame(rows)
    st.caption(
        "Valeurs positives du différentiel → avantage PSG · négatives → avantage Arsenal"
    )

    fig = go.Figure(
        go.Bar(
            x=df["value"],
            y=df["feature"],
            orientation="h",
            marker_color=df["value"].apply(lambda v: "#004170" if v >= 0 else "#EF0107"),
            text=df["value"].map(lambda v: f"{v:+.3f}"),
            textposition="outside",
        )
    )
    fig.update_layout(
        template="plotly",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=max(320, len(rows) * 36),
        margin=dict(l=10, r=40, t=20, b=20),
        xaxis_title="Valeur (différentiel)",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(
        df.rename(columns={"feature": "Indicateur", "value": "Valeur"})[["Indicateur", "Valeur"]],
        use_container_width=True,
        hide_index=True,
    )


def page_qualite(bundle: PredictionBundle) -> None:
    full = bundle.final_vector
    X = full[bundle.feature_columns]
    ordered = bundle.ordered

    def fv(col: str) -> float:
        return float(full[col].iloc[0]) if col in full.columns else 0.0

    checks = [
        ("Artefact modèle chargé", bundle.artifact_path.exists()),
        ("Terrain neutre", str(X["Venue"].iloc[0]) == "Neutral"),
        ("Features alignées", list(bundle.final_vector.columns) >= bundle.feature_columns),
        ("Probas = 100 %", abs(float(ordered.sum()) - 1.0) < 1e-5),
        ("Données xG", abs(fv("xg_diff")) > 0),
        ("Historique H2H", abs(fv("h2h_win_rate_diff")) > 0 or abs(fv("h2h_draw_rate")) > 0),
    ]

    c1, c2 = st.columns(2)
    for idx, (label, ok) in enumerate(checks):
        col = c1 if idx % 2 == 0 else c2
        with col:
            icon = "✅" if ok else "⚠️"
            st.markdown(f"{icon} **{label}**")

    st.markdown("---")
    st.subheader("Métriques hold-out")
    selected = bundle.artifact["config"]["model"].get("selected_model", "—")
    metrics = bundle.artifact.get("metrics", {}).get(selected, {})
    if metrics:
        m1, m2, m3 = st.columns(3)
        m1.metric("Modèle", selected.replace("_", " ").title())
        if "accuracy" in metrics:
            m2.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        if "log_loss" in metrics:
            m3.metric("LogLoss", f"{metrics['log_loss']:.3f}")
        if "brier_score" in metrics:
            st.metric("Brier score", f"{metrics['brier_score']:.3f}")
    else:
        st.info("Lance `python src/model.py` pour générer les métriques.")

    cal_path = bundle.artifact_path.parent / "calibration_table.csv"
    if cal_path.exists():
        st.subheader("Table de calibration")
        st.dataframe(pd.read_csv(cal_path), use_container_width=True, hide_index=True)


def page_about() -> None:
    st.markdown(
        """
        ### Pipeline ML — Finale UCL

        Projet de **data science** : collecte API, features anti-leakage, entraînement calibré,
        comparaison avec le marché.

        | Étape | Script |
        |-------|--------|
        | Données | `python src/scrapper.py` |
        | Dataset | `python src/dataset.py` |
        | Entraînement | `python src/model.py` |
        | Prédiction CLI | `python src/predict.py` |

        **Stack** : Python · pandas · scikit-learn · XGBoost · Streamlit · Plotly

        [Documentation complète](https://github.com/aneschebbi6-star/ucl-winner-predictor) sur GitHub.
        """
    )


def render_sidebar(bundle: PredictionBundle | None) -> str:
    with st.sidebar:
        st.markdown("### 🏆 UCL Predictor")
        st.caption("PSG vs Arsenal · Finale")

        if bundle is None:
            st.error("Modèle absent")
            st.code("python src/model.py", language="bash")
        else:
            st.success("Modèle chargé")
            sel = bundle.artifact["config"]["model"].get("selected_model", "—")
            st.caption(f"`{sel}`")

        st.markdown("---")
        page = option_menu(
            menu_title=None,
            options=PAGES,
            icons=["house", "graph-up", "sliders", "shield-check", "info-circle"],
            default_index=0,
            styles={
                "container": {"padding": "0", "background-color": "transparent"},
                "icon": {"font-size": "16px"},
                "nav-link": {
                    "font-size": "14px",
                    "padding": "10px 12px",
                    "border-radius": "8px",
                },
                "nav-link-selected": {"background-color": "#334155"},
            },
        )
        st.markdown("---")
        st.caption("v1 · Projet portfolio ML")
    return page


def main() -> None:
    inject_theme()
    bundle = cached_bundle()
    page = render_sidebar(bundle)

    if bundle is None:
        st.title("Finale Ligue des Champions")
        st.error("Aucun modèle trouvé dans `artifacts/model.joblib`.")
        st.markdown(
            """
            **Pour démarrer :**
            1. `pip install -r requirements.txt`
            2. `python src/scrapper.py` (données)
            3. `python src/model.py` (entraînement + artefact)
            4. Relance cette page
            """
        )
        if st.button("Recharger après entraînement"):
            st.cache_resource.clear()
            st.rerun()
        return

    render_hero(bundle)

    if page == "Accueil":
        page_accueil(bundle)
    elif page == "Analyse":
        page_analyse(bundle)
    elif page == "Features":
        page_features(bundle)
    elif page == "Qualité":
        page_qualite(bundle)
    else:
        page_about()


if __name__ == "__main__":
    main()
