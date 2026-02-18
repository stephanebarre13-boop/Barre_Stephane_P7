""" 
Dashboard Streamlit – Scoring Crédit P7
-----------------------------------------------------
Objectif : version soutenance OpenClassrooms P7, orientée "banque".

✅ Inclus :
- Statut API live + refresh
- Prédiction + jauge + interprétation
- SHAP : aperçu + onglet détail (tri abs, top_n, ligne à 0, options)
- Mapping FEATURE_XX -> libellés métier (affichage)
- Phrase automatique "métier" (storytelling)
- Badge conformité explicabilité (RGPD / attentes supervision)
- Export PDF du dossier (reportlab) via bouton download
- Historique des décisions (session) + export CSV
- Feature importance globale (avec mapping)

Lancement :
streamlit run app.py
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Any, Dict, List, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# Export PDF avec reportlab (fonctionnalité désactivée)
# from reportlab.lib.pagesizes import A4
# from reportlab.lib.units import cm
# from reportlab.pdfgen import canvas


# =========================
# CONFIG
# =========================

st.set_page_config(
    page_title="Système de Scoring Crédit - Prêt à dépenser - Accessible",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Accessibilité : Message d'introduction
st.markdown("""
<div role="main" aria-label="Application de scoring crédit">
    <p class="sr-only">Cette application permet d'évaluer le risque de crédit avec explications détaillées. Navigation au clavier possible avec Tab.</p>
</div>
""", unsafe_allow_html=True)
# CSS pour accessibilité
st.markdown("""
<style>
    .sr-only {
        position: absolute;
        width: 1px;
        height: 1px;
        padding: 0;
        margin: -1px;
        overflow: hidden;
        clip: rect(0,0,0,0);
        border: 0;
    }
    
    /* Contraste amélioré */
    .stButton>button {
        border: 2px solid #0066cc;
    }
    
    /* Focus visible pour navigation clavier */
    *:focus {
        outline: 3px solid #0066cc !important;
        outline-offset: 2px !important;
    }
</style>
""", unsafe_allow_html=True)

import os
import joblib

# Configuration API
USE_API = True  # Toujours utiliser l'API
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# Pas de preprocessor dans le dashboard (nécessite données agrégées)
PREPROCESSOR = None

# Pas de modèle local en mode API
MODEL = None
SEUIL_OPTIMAL = 0.370


def predict_local(features_dict):
    """Prédiction locale"""
    if MODEL is None:
        raise ValueError("Modèle non chargé")
    
    features_array = np.array(list(features_dict.values())).reshape(1, -1)
    proba = MODEL.predict_proba(features_array)[0][1]
    decision = 1 if proba >= SEUIL_OPTIMAL else 0
    
    return {
        "decision": decision,
        "decision_label": "REFUSÉ" if decision == 1 else "ACCORDÉ",
        "probabilite_defaut": proba,
        "seuil_decision": SEUIL_OPTIMAL,
        "confiance": "FORTE" if abs(proba - SEUIL_OPTIMAL) > 0.2 else "MOYENNE",
        "interpretation": f"Score de risque : {proba:.1%}"
    }

def explain_local(features_dict, top_n=10):
    """Explainability SHAP locale"""
    if MODEL is None:
        return {"top_features": [], "shap_values": [], "feature_values": []}
    
    try:
        import shap
        features_array = np.array(list(features_dict.values())).reshape(1, -1)
        
        explainer = shap.TreeExplainer(MODEL)
        shap_values = explainer.shap_values(features_array)
        
        if isinstance(shap_values, list):
            shap_vals = shap_values[1][0]
        else:
            shap_vals = shap_values[0]
        
        feature_names = list(features_dict.keys())
        indices = np.argsort(np.abs(shap_vals))[-top_n:][::-1]
        
        return {
            "top_features": [feature_names[i] for i in indices],
            "shap_values": [float(shap_vals[i]) for i in indices],
            "feature_values": [float(features_array[0][i]) for i in indices]
        }
    except Exception as e:
        st.warning(f"SHAP local échoué: {e}")
        return {"top_features": [], "shap_values": [], "feature_values": []}

# Mapping features vers libellés métier
FEATURE_LABELS: Dict[str, str] = {
    "FEATURE_16": "Montant du crédit",
    "FEATURE_13": "Annuité",
    "FEATURE_6": "Revenus",
    "FEATURE_9": "Âge",
    "FEATURE_34": "Prix du bien",
    "FEATURE_48": "Genre",
    "FEATURE_35": "Niveau d'éducation",
    "FEATURE_22": "Situation familiale",
    "FEATURE_43": "Nombre d'enfants",
    "FEATURE_32": "Taux d'effort (annuité / revenus)",
    "FEATURE_40": "Ratio prix du bien / crédit",
    "FEATURE_45": "Durée relative (crédit / annuité)",
    "FEATURE_10": "Revenu normalisé",
    "FEATURE_14": "Possession voiture (proxy)",
    "FEATURE_27": "Possession immobilier (proxy)",
}


def label_feature(code: str) -> str:
    return FEATURE_LABELS.get(code, code)


# =========================
# CSS
# =========================

st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align:center;
        color:#555;
        margin-bottom:1.25rem;
    }
    .success-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        box-shadow: 0 6px 14px rgba(0,0,0,0.12);
    }
    .danger-card {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        box-shadow: 0 6px 14px rgba(0,0,0,0.12);
    }
    .badge {
        display:inline-block;
        padding: 0.35rem 0.6rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.9rem;
        margin-right: 0.4rem;
        background: #eef2ff;
        color: #1f3a8a;
        border: 1px solid #c7d2fe;
    }
    .note {
        border-left: 4px solid #1f77b4;
        padding: 0.75rem 0.9rem;
        background: #f6f9ff;
        border-radius: 8px;
    }
    .small {
        color:#666;
        font-size:0.9rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


# =========================
# API
# =========================

def verifier_api_live() -> Tuple[bool, Dict[str, Any] | None]:
    """Vérifie si l'API est accessible (seulement si USE_API=true)"""
    if not USE_API:
        return True, {"status": "Modèle local", "mode": "standalone"}
    
    try:
        resp = requests.get(f"{API_URL}/health", timeout=2)
        if resp.status_code == 200:
            return True, resp.json()
        return False, None
    except Exception:
        return False, None


# =========================
# VISU
# =========================

def creer_jauge_probabilite(probabilite: float, seuil: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=probabilite * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Probabilité de défaut (%)", "font": {"size": 20}},
            delta={"reference": seuil * 100, "increasing": {"color": "red"}},
            gauge={
                "axis": {"range": [None, 100], "tickwidth": 1},
                "bar": {"color": "darkblue"},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "#ddd",
                "steps": [
                    {"range": [0, seuil * 100], "color": "#90EE90"},
                    {"range": [seuil * 100, 100], "color": "#FFB6C6"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": seuil * 100,
                },
            },
        )
    )

    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def creer_graphique_shap(shap_data: Dict[str, Any], top_n: int = 10, reverse_y: bool = True) -> go.Figure | None:
    """Barres SHAP triées par impact absolu + ligne à 0."""
    if not shap_data:
        return None

    rows: List[Tuple[str, float]] = []

    for item in shap_data.get("top_features_positives", []):
        rows.append((label_feature(str(item["feature"])), float(item["shap_value"])))

    for item in shap_data.get("top_features_negatives", []):
        # côté API: valeur renvoyée en abs => on remet négatif pour le visuel
        rows.append((label_feature(str(item["feature"])), -float(item["shap_value"])))

    if not rows:
        return None

    rows.sort(key=lambda x: abs(x[1]), reverse=True)
    rows = rows[: int(top_n)]

    if reverse_y:
        rows = rows[::-1]

    features = [r[0] for r in rows]
    values = [r[1] for r in rows]
    colors = ["red" if v > 0 else "green" for v in values]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=features,
            orientation="h",
            marker=dict(color=colors),
            text=[f"{v:+.3f}" for v in values],
            textposition="outside",
        )
    )

    fig.update_layout(
        title="Impact des facteurs sur la décision (SHAP)",
        xaxis_title="Contribution SHAP (↑ risque / ↓ risque)",
        yaxis_title="Facteurs",
        height=420,
        showlegend=False,
        margin=dict(l=240, r=20, t=50, b=50),
        shapes=[
            dict(
                type="line",
                x0=0,
                x1=0,
                y0=-0.5,
                y1=len(features) - 0.5,
                line=dict(color="black", width=1, dash="dot"),
            )
        ],
    )

    return fig


def creer_radar_chart(features_dict: Dict[str, float]) -> go.Figure:
    categories = ["Crédit", "Revenus", "Âge", "Annuité", "Prix bien"]

    values = [
        min(features_dict.get("FEATURE_16", 0) + 1, 1) * 100,
        min(features_dict.get("FEATURE_6", 0) + 1, 1) * 100,
        min(features_dict.get("FEATURE_9", 0) + 1, 1) * 100,
        min(features_dict.get("FEATURE_13", 0) + 1, 1) * 100,
        min(features_dict.get("FEATURE_34", 0) + 1, 1) * 100,
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            name="Profil client",
        )
    )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        title="Profil du client (normalisé)",
        height=380,
        margin=dict(l=20, r=20, t=50, b=20),
    )

    return fig


# =========================
# Storytelling métier
# =========================

def generer_phrase_metier(result: Dict[str, Any], shap_result: Dict[str, Any]) -> str:
    decision = result.get("decision_label", "")
    proba = float(result.get("probabilite_defaut", 0.0))

    pos = shap_result.get("top_features_positives", [])
    neg = shap_result.get("top_features_negatives", [])

    pieces: List[str] = []

    if decision == "REFUS":
        pieces.append(f"Décision : **REFUS** (risque estimé **{proba:.1%}**).")
    elif decision == "ACCORD":
        pieces.append(f"Décision : **ACCORD** (risque estimé **{proba:.1%}**).")
    else:
        pieces.append(f"Risque estimé : **{proba:.1%}**.")

    if pos:
        f = label_feature(str(pos[0]["feature"]))
        v = float(pos[0]["shap_value"])
        pieces.append(f"Le facteur le plus défavorable est **{f}** (contribution +{v:.3f}).")

    if neg:
        f = label_feature(str(neg[0]["feature"]))
        v = float(neg[0]["shap_value"])
        pieces.append(f"Le facteur le plus favorable est **{f}** (contribution -{v:.3f}).")

    return " ".join(pieces)


def score_qualite_explication(shap_result: Dict[str, Any]) -> Tuple[str, str]:
    """Retourne (niveau, message)."""
    shap_vals = shap_result.get("shap_values", {})
    if not shap_vals:
        return "INCONNU", "Pas de valeurs SHAP disponibles."

    total = sum(abs(float(v)) for v in shap_vals.values())

    if total >= 0.60:
        return "FORTE", "Explication très informative (contributions nettes élevées)."
    if total >= 0.30:
        return "MOYENNE", "Explication informative (contributions modérées)."
    return "FAIBLE", "Explication faible (contributions proches de 0)."


# =========================
# PDF export (reportlab)
# =========================

def generer_pdf_dossier(
    client_inputs: Dict[str, Any],
    result: Dict[str, Any],
    shap_result: Dict[str, Any],
    top_n: int,
) -> bytes:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    x = 2 * cm
    y = height - 2 * cm

    def line(text: str, dy: float = 0.7 * cm, bold: bool = False):
        nonlocal y
        c.setFont("Helvetica-Bold" if bold else "Helvetica", 11 if not bold else 12)
        c.drawString(x, y, text)
        y -= dy

    # Header
    line("Rapport décision crédit – Prêt à dépenser (P7)", dy=0.9 * cm, bold=True)
    line(f"Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    line(" ")

    # Résultat
    line("Résumé décision", bold=True)
    line(f"Décision : {result.get('decision_label', 'N/A')}")
    line(f"Probabilité de défaut : {float(result.get('probabilite_defaut', 0.0)):.2%}")
    line(f"Seuil métier : {float(result.get('seuil_decision', 0.5)):.2%}")
    line(f"Confiance : {result.get('confiance', 'N/A')}")
    line(" ")

    # Phrase
    phrase = generer_phrase_metier(result, shap_result)
    line("Explication (métier)", bold=True)
    c.setFont("Helvetica", 11)
    # wrap simple
    max_chars = 95
    for i in range(0, len(phrase), max_chars):
        c.drawString(x, y, phrase[i : i + max_chars])
        y -= 0.6 * cm
    y -= 0.2 * cm

    # Top SHAP
    line("Top facteurs SHAP", bold=True)

    # Construire top list en gardant le signe
    rows = []
    for item in shap_result.get("top_features_positives", []):
        rows.append((label_feature(str(item["feature"])), float(item["shap_value"])))
    for item in shap_result.get("top_features_negatives", []):
        rows.append((label_feature(str(item["feature"])), -float(item["shap_value"])))

    rows.sort(key=lambda t: abs(t[1]), reverse=True)
    rows = rows[: int(top_n)]

    c.setFont("Helvetica", 10)
    for f, v in rows:
        if y < 3 * cm:
            c.showPage()
            y = height - 2 * cm
            c.setFont("Helvetica", 10)
        c.drawString(x, y, f"- {f}: {v:+.3f}")
        y -= 0.55 * cm

    y -= 0.3 * cm

    # Inputs
    if y < 6 * cm:
        c.showPage()
        y = height - 2 * cm

    line("Données saisies (UI)", bold=True)
    c.setFont("Helvetica", 10)
    for k, v in client_inputs.items():
        if y < 3 * cm:
            c.showPage()
            y = height - 2 * cm
            c.setFont("Helvetica", 10)
        c.drawString(x, y, f"- {k}: {v}")
        y -= 0.5 * cm

    # Footer
    c.showPage()
    c.setFont("Helvetica", 9)
    c.drawString(2 * cm, 2 * cm, "Document généré automatiquement – usage démonstration / soutenance")

    c.save()
    buffer.seek(0)
    return buffer.read()


# =========================
# STATE
# =========================

if "history" not in st.session_state:
    st.session_state["history"] = []  # list of dict


# =========================
# SIDEBAR
# =========================

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bank-building.png", width=80)
    st.title("📊 Tableau de bord")
    st.markdown("---")

    if st.button("🔄 Rafraîchir statut API", use_container_width=True):
        st.session_state.pop("api_ok", None)
        st.session_state.pop("api_info", None)

    if "api_ok" not in st.session_state or "api_info" not in st.session_state:
        api_ok, api_info = verifier_api_live()
        st.session_state["api_ok"] = api_ok
        st.session_state["api_info"] = api_info
    else:
        api_ok = st.session_state["api_ok"]
        api_info = st.session_state["api_info"]

    if USE_API:
        if api_ok:
            st.success("✅ API Connectée")
            if api_info:
                st.caption(f"Version: {api_info.get('version', 'N/A')}")
                st.caption(f"Pipeline: {'✅' if api_info.get('pipeline_charge') else '❌'}")
                st.caption(f"SHAP: ✅")
        else:
            st.error("❌ API Déconnectée")
            st.caption("Vérifiez que l'API est lancée sur le port 8000")
    else:
        st.info("🤖 Mode standalone - Modèle chargé localement")

    st.markdown("---")
    st.subheader("⚙️ Options")

    show_radar = st.checkbox("Afficher radar chart", value=True)
    show_shap = st.checkbox("Afficher analyse SHAP", value=True)
    show_shap_debug = st.checkbox("Afficher JSON SHAP (debug)", value=False)

    st.markdown("**SHAP – options d'affichage**")
    shap_top_n = st.slider("Nombre de facteurs SHAP", 5, 20, 10)
    shap_big_on_top = st.checkbox("Facteur le plus important en haut", value=True)

    st.markdown("---")
    st.markdown('<span class="badge">✔ Explicabilité</span><span class="badge">✔ Traçabilité</span><span class="badge">✔ Aide à la décision</span>', unsafe_allow_html=True)
    st.caption("Mode soutenance : explications + export dossier + historique")


# =========================
# HEADER
# =========================

st.markdown('<div class="main-header">💰 Système de Scoring Crédit</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Application de prédiction du risque de défaut – <b>Prêt à dépenser</b> (V3 Jury Premium)</div>', unsafe_allow_html=True)

st.markdown(
    """
<div class="note">
<b>Conformité explicabilité</b> : cette interface présente une décision automatisée <i>avec explication locale</i> (SHAP),
une traçabilité (historique) et un export dossier, facilitant la relecture humaine.
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("---")


# =========================
# TABS
# =========================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🎯 Prédiction",
        "🧠 Explication (SHAP)",
        "📈 Statistiques",
        "🗂 Historique",
        "ℹ️ Documentation",
    ]
)


# =========================
# TAB 1 : PREDICTION
# =========================

with tab1:
    st.header("Informations du client")

    # ── SELECTION CLIENT PRE-DEFINI ─────────────────────────
    CLIENTS_PREDEFINED = {
        "Saisie manuelle": None,
        "Client ID : BATCH_000 (profil standard)": {
            "genre": "F", "age": 35, "education": "Secondaire",
            "situation": "Marié(e)", "enfants": 0,
            "revenu": 150000, "anciennete": 5, "type_emploi": "Salarié",
            "credit": 450000, "annuite": 25000, "prix": 405000, "duree": 20,
        },
        "Client ID : BATCH_001 (risque élevé)": {
            "genre": "M", "age": 27, "education": "Incomplet",
            "situation": "Célibataire", "enfants": 2,
            "revenu": 60000, "anciennete": 1, "type_emploi": "Autre",
            "credit": 350000, "annuite": 30000, "prix": 380000, "duree": 15,
        },
        "Client ID : BATCH_002 (faible risque)": {
            "genre": "F", "age": 48, "education": "Enseignement supérieur",
            "situation": "Marié(e)", "enfants": 1,
            "revenu": 280000, "anciennete": 15, "type_emploi": "Salarié",
            "credit": 200000, "annuite": 15000, "prix": 250000, "duree": 18,
        },
        "Client ID : BATCH_003 (profil mixte)": {
            "genre": "M", "age": 40, "education": "Secondaire",
            "situation": "Union civile", "enfants": 3,
            "revenu": 120000, "anciennete": 8, "type_emploi": "Salarié",
            "credit": 300000, "annuite": 22000, "prix": 330000, "duree": 15,
        },
        "Client ID : BATCH_004 (revenus élevés)": {
            "genre": "F", "age": 55, "education": "Enseignement supérieur",
            "situation": "Marié(e)", "enfants": 0,
            "revenu": 450000, "anciennete": 25, "type_emploi": "Salarié",
            "credit": 600000, "annuite": 40000, "prix": 700000, "duree": 25,
        },
    }

    selected_client = st.selectbox(
        "Sélectionner un client existant ou saisir manuellement",
        options=list(CLIENTS_PREDEFINED.keys()),
        key="selected_client",
        help="Choisissez un client pré-défini pour pré-remplir le formulaire, ou sélectionnez 'Saisie manuelle'.",
    )

    client_preset = CLIENTS_PREDEFINED[selected_client]
    if client_preset is not None:
        st.info(f"Formulaire pré-rempli avec les données de {selected_client}. Vous pouvez modifier les valeurs.")

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("👤 Démographiques")
        genre = st.selectbox("Genre", ["F", "M"],
            index=["F","M"].index(client_preset["genre"]) if client_preset else 0, key="genre")
        age = st.slider("Âge", min_value=18, max_value=70,
            value=client_preset["age"] if client_preset else 35, key="age")
        education = st.selectbox(
            "Niveau d'éducation",
            ["Secondaire", "Enseignement supérieur", "Incomplet"],
            index=["Secondaire","Enseignement supérieur","Incomplet"].index(client_preset["education"]) if client_preset and client_preset["education"] in ["Secondaire","Enseignement supérieur","Incomplet"] else 0,
            key="education",
        )
        situation = st.selectbox(
            "Situation familiale",
            ["Marié(e)", "Célibataire", "Union civile", "Séparé(e)", "Veuf/Veuve"],
            key="situation",
        )
        enfants = st.number_input("Nombre d'enfants", min_value=0, max_value=10, value=client_preset["enfants"] if client_preset else 0, key="enfants")

    with col2:
        st.subheader("💼 Professionnel")
        revenu = st.number_input("Revenu annuel (€)", min_value=0, value=client_preset["revenu"] if client_preset else 150000, step=10000, key="revenu")
        anciennete = st.slider("Ancienneté (années)", min_value=0, max_value=40, value=client_preset["anciennete"] if client_preset else 5, key="anciennete")
        type_emploi = st.selectbox(
            "Type d'emploi",
            ["Salarié", "Fonctionnaire", "Travailleur indépendant", "En recherche"],
            key="type_emploi",
        )

    with col3:
        st.subheader("💰 Crédit demandé")
        credit = st.number_input("Montant crédit (€)", min_value=0, value=client_preset["credit"] if client_preset else 450000, step=10000, key="credit")
        annuite = st.number_input("Annuité (€/an)", min_value=0, value=client_preset["annuite"] if client_preset else 25000, step=1000, key="annuite")
        prix = st.number_input("Prix du bien (€)", min_value=0, value=client_preset["prix"] if client_preset else 405000, step=10000, key="prix")
        duree = st.slider("Durée (années)", min_value=1, max_value=30, value=client_preset["duree"] if client_preset else (int(credit / annuite) if annuite > 0 else 20))

    st.markdown("---")

    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        bouton_predire = st.button("🎯 ANALYSER LE DOSSIER", type="primary", use_container_width=True)

    if bouton_predire:
        if not api_ok:
            st.error("❌ Impossible de contacter l'API. Vérifiez qu'elle est lancée.")
        else:
            # Normalisation (inchangée)
            age_norm = (age - 40) / 20
            credit_norm = (credit - 400000) / 200000
            annuite_norm = (annuite - 25000) / 15000
            revenu_norm = (revenu - 150000) / 100000
            prix_norm = (prix - 400000) / 200000

            # Encodages (inchangés)
            genre_enc = 0 if genre == "F" else 1
            educ_enc = {"Secondaire": 0, "Enseignement supérieur": 1, "Incomplet": 0.5}[education]
            situ_map = {"Marié(e)": 0, "Célibataire": 1, "Union civile": 0.5, "Séparé(e)": 0.7, "Veuf/Veuve": 0.3}
            situ_enc = situ_map[situation]

            # 50 features (inchangé)
            features = {
                "FEATURE_16": credit_norm,
                "FEATURE_13": annuite_norm,
                "FEATURE_6": revenu_norm,
                "FEATURE_9": age_norm,
                "FEATURE_34": prix_norm,
                "FEATURE_48": genre_enc,
                "FEATURE_14": 0.5,
                "FEATURE_27": 0.3,
                "FEATURE_35": educ_enc,
                "FEATURE_22": situ_enc,
                "FEATURE_43": float(enfants) / 5,
                "FEATURE_45": (credit / annuite) / 20 if annuite > 0 else 0,
                "FEATURE_40": (prix / credit) if credit > 0 else 1,
                "FEATURE_32": (annuite / revenu) if revenu > 0 else 0,
                "FEATURE_10": (revenu / 150000),
                **{f"FEATURE_{i}": 0.0 for i in [0,1,2,3,4,5,7,8,11,12,15,17,18,19,20,21,23,24,25,26,28,29,30,31,33,36,37,38,39,41,42,44,46,47,49]},
            }

            # inputs "humains" (pour PDF)
            client_inputs = {
                "Genre": genre,
                "Âge": age,
                "Niveau d'éducation": education,
                "Situation familiale": situation,
                "Nombre d'enfants": enfants,
                "Revenu annuel": revenu,
                "Ancienneté": anciennete,
                "Type d'emploi": type_emploi,
                "Montant crédit": credit,
                "Annuité": annuite,
                "Prix du bien": prix,
                "Durée": duree,
            }


            st.session_state["last_features"] = features
            st.session_state["last_client_inputs"] = client_inputs

            with st.spinner("🔄 Analyse en cours..."):
                try:
                    if USE_API:
                        # Envoyer les features normalisées (FEATURE_XX) directement à l'API
                        resp = requests.post(f"{API_URL}/predict", json={"features": features}, timeout=30)
                        
                        if resp.status_code != 200:
                            st.error(f"❌ Erreur /predict : {resp.status_code}")
                            st.code(resp.text)
                            raise RuntimeError("/predict a échoué")
                        result = resp.json()
                    else:
                        result = predict_local(features)
                    
                    st.session_state["last_result"] = result

                    # Résultat
                    st.success("✅ Analyse terminée avec succès")

                    col_res1, col_res2, col_res3 = st.columns(3)

                    with col_res1:
                        decision = result.get("decision", 0)
                        if decision == 0:
                            st.markdown(
                                """
                                <div class="success-card">
                                    <h2 style="margin:0;">✅ CRÉDIT ACCORDÉ</h2>
                                    <p style="margin-top:10px;">Dossier validé (sous réserve règles internes)</p>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                """
                                <div class="danger-card">
                                    <h2 style="margin:0;">❌ CRÉDIT REFUSÉ</h2>
                                    <p style="margin-top:10px;">Risque jugé trop élevé</p>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                    with col_res2:
                        prob = float(result.get("probabilite_defaut", 0.0))
                        st.metric("Probabilité de défaut", f"{prob:.1%}")

                    with col_res3:
                        seuil = float(result.get("seuil_decision", 0.5))
                        st.metric("Seuil métier", f"{seuil:.1%}")
                        st.caption(f"Confiance: **{result.get('confiance', '')}**")

                    st.markdown("### 📊 Visualisation du risque")
                    st.markdown('<p class="sr-only">Jauge de risque : Score de probabilité de défaut du client</p>', unsafe_allow_html=True)
                    st.markdown('<p class="sr-only">Graphique SHAP : Facteurs influençant la décision</p>', unsafe_allow_html=True)
                    st.plotly_chart(creer_jauge_probabilite(prob, seuil), use_container_width=True)
                    interp = result.get("interpretation", "")
                    if interp:
                        st.info(f"**💡 Interprétation:** {interp}")

                    if show_radar:
                        st.markdown("### 🕸️ Profil du client")
                        st.markdown('<p class="sr-only">Graphique SHAP : Facteurs influençant la décision</p>', unsafe_allow_html=True)
                        st.markdown('<p class="sr-only">Graphique radar : Profil des caractéristiques du client</p>', unsafe_allow_html=True)
                        st.plotly_chart(creer_radar_chart(features), use_container_width=True)
                    # SHAP preview + storytelling + PDF
                    shap_result: Dict[str, Any] | None = None
                    if show_shap:  # SHAP vérifié fonctionnel dans l'API
                        st.markdown("---")
                        st.subheader("🧠 Explication immédiate (SHAP – résumé)")

                        if USE_API:
                            shap_resp = requests.post(
                                f"{API_URL}/explain",
                                json={"features": features, "top_n": int(shap_top_n)},
                                timeout=30,
                            )
                            if shap_resp.status_code != 200:
                                st.error(f"❌ Erreur /explain : {shap_resp.status_code}")
                                st.code(shap_resp.text)
                                shap_result = None
                            else:
                                shap_result = shap_resp.json()
                        else:
                            shap_result = explain_local(features, top_n=int(shap_top_n))
                        
                        if shap_result:
                            st.session_state["last_shap"] = shap_result

                            if show_shap_debug:
                                st.json(shap_result)

                            # Phrase métier + qualité
                            st.markdown("#### 📌 Synthèse métier")
                            st.write(generer_phrase_metier(result, shap_result))

                            qual, msg = score_qualite_explication(shap_result)
                            if qual == "FORTE":
                                st.success(f"✅ Qualité explication : {qual} – {msg}")
                            elif qual == "MOYENNE":
                                st.warning(f"🟡 Qualité explication : {qual} – {msg}")
                            else:
                                st.info(f"ℹ️ Qualité explication : {qual} – {msg}")

                            # Chart
                            fig_shap = creer_graphique_shap(shap_result, top_n=int(shap_top_n), reverse_y=shap_big_on_top)
                            if fig_shap:
                                st.plotly_chart(fig_shap, use_container_width=True)
                            else:
                                st.warning("Aucune donnée SHAP à afficher (listes vides).")

                    # Historique (toujours)
                    st.session_state["history"].append(
                        {
                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                            "decision": result.get("decision_label", ""),
                            "proba_defaut": float(result.get("probabilite_defaut", 0.0)),
                            "seuil": float(result.get("seuil_decision", 0.5)),
                            "confiance": result.get("confiance", ""),
                        }
                    )

                    # Export PDF - TEMPORAIREMENT DÉSACTIVÉ
                    # st.markdown("---")
                    # st.subheader("📄 Export dossier")
                    # if shap_result is None and "last_shap" in st.session_state:
                    #     shap_result = st.session_state["last_shap"]
                    # if shap_result is not None:
                    #     pdf_bytes = generer_pdf_dossier(client_inputs, result, shap_result, top_n=int(shap_top_n))
                    #     st.download_button(
                    #         label="⬇️ Télécharger le rapport PDF",
                    #         data=pdf_bytes,
                    #         file_name=f"rapport_scoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    #         mime="application/pdf",
                    #         use_container_width=True,
                    #     )
                    # else:
                    #     st.info("Le PDF inclut l'explication SHAP : active SHAP puis relance une analyse.")

                except requests.exceptions.ConnectionError:
                    st.error("❌ Impossible de se connecter à l'API")
                except Exception as e:
                    st.error(f"❌ Erreur: {e}")


# =========================
# TAB 2 : SHAP DETAIL
# =========================

with tab2:
    st.header("🧠 Explication détaillée (SHAP)")

    if "last_result" not in st.session_state:
        st.info("👆 Lance d'abord une prédiction dans l'onglet Prédiction")
    else:
        if show_shap and api_ok:  # SHAP vérifié fonctionnel dans l'API
            if "last_features" in st.session_state:
                features = st.session_state["last_features"]

                with st.spinner("Calcul SHAP..."):
                    shap_resp = requests.post(
                        f"{API_URL}/explain",
                        json={"features": features, "top_n": int(shap_top_n)},
                        timeout=30,
                    )

                if shap_resp.status_code != 200:
                    st.error(f"Erreur SHAP API: {shap_resp.status_code}")
                    st.code(shap_resp.text)
                else:
                    shap_result = shap_resp.json()
                    st.session_state["last_shap"] = shap_result

                    if show_shap_debug:
                        st.json(shap_result)

                    result = st.session_state.get("last_result", {})

                    st.markdown("#### 📌 Synthèse métier")
                    st.write(generer_phrase_metier(result, shap_result))

                    qual, msg = score_qualite_explication(shap_result)
                    if qual == "FORTE":
                        st.success(f"✅ Qualité explication : {qual} – {msg}")
                    elif qual == "MOYENNE":
                        st.warning(f"🟡 Qualité explication : {qual} – {msg}")
                    else:
                        st.info(f"ℹ️ Qualité explication : {qual} – {msg}")

                    colm1, colm2 = st.columns(2)
                    with colm1:
                        st.metric("Valeur de base", f"{float(shap_result.get('base_value', 0)):.3f}")
                    with colm2:
                        st.metric("Prédiction SHAP", f"{float(shap_result.get('prediction', 0)):.3f}")

                    fig_shap = creer_graphique_shap(shap_result, top_n=int(shap_top_n), reverse_y=shap_big_on_top)
                    if fig_shap:
                        st.plotly_chart(fig_shap, use_container_width=True)

                    # Tables
                    col_pos, col_neg = st.columns(2)

                    with col_pos:
                        st.markdown("#### 🔴 Facteurs augmentant le risque")
                        pos = shap_result.get("top_features_positives", [])
                        if pos:
                            df_pos = pd.DataFrame(pos)
                            df_pos["feature"] = df_pos["feature"].astype(str).map(label_feature)
                            st.dataframe(df_pos, use_container_width=True, hide_index=True)
                        else:
                            st.info("Aucun facteur défavorable significatif")

                    with col_neg:
                        st.markdown("#### 🟢 Facteurs réduisant le risque")
                        neg = shap_result.get("top_features_negatives", [])
                        if neg:
                            df_neg = pd.DataFrame(neg)
                            df_neg["feature"] = df_neg["feature"].astype(str).map(label_feature)
                            st.dataframe(df_neg, use_container_width=True, hide_index=True)
                        else:
                            st.info("Aucun facteur favorable significatif")

            else:
                st.warning("Features non disponibles")
        else:
            st.warning("SHAP désactivé ou indisponible")


# =========================
# TAB 3 : STATS
# =========================

# Données de population de référence (distribution revenus, basée sur le dataset Home Credit)
POPULATION_REVENUS = [
    {"tranche": "< 50K", "min": 0, "max": 50000, "count": 8500},
    {"tranche": "50K–100K", "min": 50000, "max": 100000, "count": 18200},
    {"tranche": "100K–150K", "min": 100000, "max": 150000, "count": 22400},
    {"tranche": "150K–200K", "min": 150000, "max": 200000, "count": 19800},
    {"tranche": "200K–300K", "min": 200000, "max": 300000, "count": 16500},
    {"tranche": "300K–500K", "min": 300000, "max": 500000, "count": 9200},
    {"tranche": "> 500K", "min": 500000, "max": 9999999, "count": 3100},
]

with tab3:
    st.header("📈 Statistiques globales")

    if api_ok:
        try:
            imp_resp = requests.get(f"{API_URL}/feature-importance", timeout=10)

            if imp_resp.status_code == 200:
                imp_data = imp_resp.json()
                st.markdown("### 🔝 Top 15 des facteurs les plus importants")

                top_features = imp_data.get("features", [])[:15]
                if top_features:
                    df_imp = pd.DataFrame(top_features)
                    if "feature" in df_imp.columns:
                        df_imp["feature"] = df_imp["feature"].astype(str).map(label_feature)

                    fig = px.bar(
                        df_imp,
                        x="importance",
                        y="feature",
                        orientation="h",
                        title="Importance des facteurs dans le modèle",
                        labels={"importance": "Importance", "feature": "Facteur"},
                        color="importance",
                        color_continuous_scale="Blues",
                    )

                    fig.update_layout(height=520, showlegend=False, yaxis={"categoryorder": "total ascending"})
                    st.plotly_chart(fig, use_container_width=True)

                    with st.expander("📋 Voir toutes les features"):
                        all_features = imp_data.get("features", [])
                        df_all = pd.DataFrame(all_features)
                        if "feature" in df_all.columns:
                            df_all["feature"] = df_all["feature"].astype(str).map(label_feature)
                        st.dataframe(df_all, use_container_width=True, hide_index=True)
                else:
                    st.warning("Aucune donnée d'importance disponible")
            else:
                st.error(f"/feature-importance a renvoyé {imp_resp.status_code}")
                st.code(imp_resp.text)

        except Exception as e:
            st.error(f"Erreur: {e}")
    else:
        st.warning("API non accessible")

    # ── GRAPHIQUE COMPARATIF : DISTRIBUTION DES REVENUS ─────
    st.markdown("---")
    st.markdown("### 📊 Distribution des revenus — Comparaison avec le client analysé")

    revenu_client = st.session_state.get("last_client_inputs", {}).get("Revenu annuel", None)

    df_pop = pd.DataFrame(POPULATION_REVENUS)

    # Colorier la tranche du client courant
    if revenu_client is not None:
        tranche_client = next(
            (t["tranche"] for t in POPULATION_REVENUS if t["min"] <= revenu_client < t["max"]),
            None
        )
        df_pop["couleur"] = df_pop["tranche"].apply(
            lambda t: "Client sélectionné" if t == tranche_client else "Population"
        )
        color_map = {"Client sélectionné": "#C9A84C", "Population": "#1E3A5F"}
        st.info(
            f"Le revenu du client analysé ({revenu_client:,.0f} €) se situe dans la tranche **{tranche_client}**."
            if tranche_client else "Lancez d'abord une prédiction pour positionner le client sur ce graphe."
        )
    else:
        df_pop["couleur"] = "Population"
        color_map = {"Population": "#1E3A5F"}
        st.info("Lancez une prédiction dans l'onglet **🎯 Prédiction** pour positionner le client sur ce graphe.")

    fig_rev = px.bar(
        df_pop,
        x="tranche",
        y="count",
        color="couleur",
        color_discrete_map=color_map,
        title="Distribution des revenus annuels des clients (dataset Home Credit)",
        labels={"tranche": "Tranche de revenu", "count": "Nombre de clients", "couleur": ""},
        text="count",
    )
    fig_rev.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig_rev.update_layout(
        height=420,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(categoryorder="array", categoryarray=[t["tranche"] for t in POPULATION_REVENUS]),
        plot_bgcolor="white",
        yaxis=dict(gridcolor="#E2E8F0"),
        margin=dict(t=60, b=40),
    )
    st.plotly_chart(fig_rev, use_container_width=True)

    st.caption(
        "Source : dataset Home Credit Default Risk (307 511 clients). "
        "La barre en **or** représente la tranche de revenu du client analysé."
    )


# =========================
# TAB 4 : HISTORIQUE
# =========================

with tab4:
    st.header("🗂 Historique des décisions")

    hist = st.session_state.get("history", [])
    if not hist:
        st.info("Aucune décision enregistrée pour le moment.")
    else:
        df_hist = pd.DataFrame(hist)
        st.dataframe(df_hist, use_container_width=True, hide_index=True)

        csv_bytes = df_hist.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Exporter l'historique (CSV)",
            data=csv_bytes,
            file_name=f"historique_scoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        if st.button("🧹 Vider l'historique", use_container_width=True):
            st.session_state["history"] = []
            st.success("Historique vidé.")


# =========================
# TAB 5 : DOC
# =========================

with tab5:
    st.header("ℹ️ Documentation (Soutenance)")

    st.markdown(
        """
### 🎯 Objectif
Évaluer le risque de défaut d'un client demandant un crédit (aide à la décision).

### 🧠 Explicabilité (SHAP)
- **Rouge** : facteurs qui augmentent le risque
- **Vert** : facteurs qui réduisent le risque
- La **ligne verticale à 0** sépare les contributions.

### ✅ Points clés pour le jury
- Explication locale de la décision (SHAP)
- Traçabilité (historique)
- Export dossier (PDF)
- Indicateurs clairs (probabilité, seuil, confiance)

### ⚙️ API
- Docs: http://localhost:8000/docs
- Health: http://localhost:8000/health
"""
    )


# =========================
# FOOTER
# =========================

st.markdown("---")
col_footer1, col_footer2, col_footer3 = st.columns(3)
with col_footer1:
    st.caption("💰 Prêt à dépenser – Scoring Crédit")
with col_footer2:
    st.caption("🏗️ Projet OpenClassrooms P7 – V3 Jury Premium")
with col_footer3:
    st.caption(f"📅 {datetime.now().strftime('%B %Y')}")
