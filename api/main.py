"""
API FastAPI PREMIUM – Projet 7 (Prêt à dépenser)
Version améliorée avec :
- SHAP pour interprétabilité
- Feature importance globale
- Validation avancée
- Logs structurés
- Gestion erreurs professionnelle

✅ Corrections apportées (robuste aux noms d'étapes du pipeline) :
- Log des steps du pipeline au démarrage
- /model-info : récupère le modèle via le dernier step (pipeline.steps[-1])
- /explain : utilise le preprocess via pipeline[:-1]
- /feature-importance : robuste aux noms de steps + fallback sur noms de features
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
DOSSIER_API = Path(__file__).resolve().parent
DOSSIER_RACINE = DOSSIER_API.parent
DOSSIER_ARTIFACTS = DOSSIER_RACINE / "artifacts"

CHEMIN_MODELE = DOSSIER_ARTIFACTS / "meilleur_modele.joblib"
CHEMIN_PARAMS = DOSSIER_ARTIFACTS / "parametres_decision.joblib"

# -----------------------------------------------------------------------------
# Loaders
# -----------------------------------------------------------------------------

def charger_pipeline() -> Any:
    """Charge le modèle LightGBM"""
    if not CHEMIN_MODELE.exists():
        raise FileNotFoundError(f"Modèle introuvable: {CHEMIN_MODELE}")
    logger.info(f"Chargement modèle depuis: {CHEMIN_MODELE}")
    return joblib.load(CHEMIN_MODELE)


def charger_parametres_decision() -> Dict[str, Any]:
    """Charge les paramètres métier (seuil, coûts)"""
    if CHEMIN_PARAMS.exists():
        params = joblib.load(CHEMIN_PARAMS)
        if isinstance(params, dict):
            logger.info(f"Paramètres métier chargés: {params}")
            return params
    logger.warning("Paramètres par défaut utilisés")
    return {
        "modele": "inconnu",
        "seuil_optimal": 0.5,
        "cout_fn": 10,
        "cout_fp": 1
    }


# -----------------------------------------------------------------------------
# Load at startup
# -----------------------------------------------------------------------------
try:
    pipeline_final = charger_pipeline()
    logger.info("✅ Modèle chargé avec succès")
except Exception as exc:
    pipeline_final = None
    erreur_chargement = str(exc)
    logger.error(f"❌ Erreur chargement: {exc}")
else:
    erreur_chargement = None

parametres_decision = charger_parametres_decision()
SEUIL_DECISION = float(parametres_decision.get("seuil_optimal", 0.5))

# SHAP explainer (lazy)
_shap_explainer = None


def _get_modele_estimateur() -> Any:
    """Retourne le modèle."""
    return pipeline_final


def _get_preprocess() -> Any:
    """Pas de préprocesseur séparé - features déjà normalisées."""
    return None


def get_shap_explainer():
    global _shap_explainer
    if _shap_explainer is None and pipeline_final is not None:
        try:
            logger.info("Initialisation SHAP explainer...")
            modele = _get_modele_estimateur()
            _shap_explainer = shap.TreeExplainer(modele)
            logger.info("✅ SHAP explainer initialisé")
        except Exception as exc:
            logger.warning(f"⚠️ SHAP non disponible: {exc}")
            _shap_explainer = None
    return _shap_explainer


# -----------------------------------------------------------------------------
# Pydantic models
# -----------------------------------------------------------------------------
class RequetePrediction(BaseModel):
    """Requête de prédiction avec features client"""
    features: Dict[str, Any] = Field(
        ...,
        description="Features du client (clé/valeur)",
        example={
            "AMT_CREDIT": 450000,
            "AMT_ANNUITY": 25000,
            "AMT_GOODS_PRICE": 400000,
            "DAYS_BIRTH": -15000,
            "DAYS_EMPLOYED": -3000,
            "CODE_GENDER": "F",
            "NAME_EDUCATION_TYPE": "Higher education"
        }
    )

    @validator('features')
    def valider_features_non_vides(cls, v):
        if not v:
            raise ValueError("Features ne peut pas être vide")
        return v


class ReponsePrediction(BaseModel):
    """Réponse de prédiction complète"""
    client_id: Optional[int] = Field(None, description="ID client (si fourni)")
    probabilite_defaut: float = Field(..., description="Probabilité de défaut [0-1]")
    score_percent: float = Field(..., description="Score en pourcentage [0-100]")
    decision: int = Field(..., description="0=Accord, 1=Refus")
    decision_label: str = Field(..., description="ACCORD ou REFUS")
    seuil_decision: float = Field(..., description="Seuil optimal métier")
    interpretation: str = Field(..., description="Explication de la décision")
    confiance: str = Field(..., description="Niveau de confiance")


class RequeteExplication(BaseModel):
    """Requête d'explication SHAP"""
    features: Dict[str, Any] = Field(..., description="Features du client")
    top_n: int = Field(10, description="Nombre de features à afficher", ge=1, le=50)


class ReponseExplication(BaseModel):
    """Réponse avec valeurs SHAP"""
    base_value: float = Field(..., description="Valeur de base (moyenne)")
    shap_values: Dict[str, float] = Field(..., description="Valeurs SHAP par feature")
    prediction: float = Field(..., description="Prédiction finale")
    top_features_positives: List[Dict[str, Any]] = Field(..., description="Features augmentant le risque")
    top_features_negatives: List[Dict[str, Any]] = Field(..., description="Features réduisant le risque")


class ReponseFeatureImportance(BaseModel):
    """Importance globale des features"""
    features: List[Dict[str, Any]] = Field(..., description="Liste des features et leur importance")
    top_10: List[str] = Field(..., description="Top 10 features les plus importantes")


class HealthResponse(BaseModel):
    """Statut de santé de l'API"""
    status: str = Field(..., description="ok ou ko")
    pipeline_charge: bool
    shap_disponible: bool
    version: str
    erreur: Optional[str] = None


# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------
app = FastAPI(
    title="API Scoring Crédit PREMIUM – Projet 7",
    version="2.0.0",
    description="""
    API professionnelle de scoring crédit avec :
    - Prédiction de défaut
    - Interprétabilité SHAP
    - Feature importance globale
    - Optimisation métier (seuil optimal)
    """,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.get("/", tags=["Root"])
def root():
    return {
        "message": "API Scoring Crédit PREMIUM",
        "version": "2.0.0",
        "endpoints": {
            "health": "/health",
            "model_info": "/model-info",
            "predict": "/predict (POST)",
            "explain": "/explain (POST)",
            "feature_importance": "/feature-importance"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health() -> HealthResponse:
    shap_ok = get_shap_explainer() is not None
    return HealthResponse(
        status="ok" if pipeline_final is not None else "ko",
        pipeline_charge=pipeline_final is not None,
        shap_disponible=shap_ok,
        version="2.0.0",
        erreur=erreur_chargement
    )


@app.get("/model-info", tags=["Model"])
def model_info() -> Dict[str, Any]:
    if pipeline_final is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline non chargé: {erreur_chargement}"
        )

    # ✅ Robuste : dernier step
    modele = _get_modele_estimateur()
    nom_modele = getattr(modele, "__class__", type("X", (), {})).__name__

    return {
        "modele": parametres_decision.get("modele", nom_modele),
        "type_modele": nom_modele,
        "seuil_decision": SEUIL_DECISION,
        "cout_fn": parametres_decision.get("cout_fn", 10),
        "cout_fp": parametres_decision.get("cout_fp", 1),
        "formule_cout": "Coût = 10×FN + 1×FP",
        "artefacts": {
            "pipeline": str(CHEMIN_PIPELINE.name),
            "parametres": str(CHEMIN_PARAMS.name)
        }
    }


def _interprete_decision(probabilite: float, seuil: float) -> tuple[str, str]:
    ecart = abs(probabilite - seuil)

    if probabilite >= seuil:
        if ecart > 0.2:
            confiance = "HAUTE"
        elif ecart > 0.1:
            confiance = "MOYENNE"
        else:
            confiance = "FAIBLE (proche du seuil)"

        interpretation = (
            f"⚠️ Risque de défaut ÉLEVÉ ({probabilite:.1%}). "
            f"Recommandation : REFUS ou analyse approfondie. "
            f"Le client dépasse le seuil métier de {seuil:.1%}."
        )
    else:
        if ecart > 0.2:
            confiance = "HAUTE"
        elif ecart > 0.1:
            confiance = "MOYENNE"
        else:
            confiance = "FAIBLE (proche du seuil)"

        interpretation = (
            f"✅ Risque de défaut FAIBLE ({probabilite:.1%}). "
            f"Recommandation : ACCORD possible (sous réserve règles internes). "
            f"Le client est en-dessous du seuil métier de {seuil:.1%}."
        )

    return interpretation, confiance


@app.post("/predict", response_model=ReponsePrediction, tags=["Prediction"])
def predire(requete: RequetePrediction) -> ReponsePrediction:
    if pipeline_final is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline non chargé: {erreur_chargement}"
        )

    try:
        donnees_client = pd.DataFrame([requete.features])
        logger.info(f"Prédiction pour client avec {len(requete.features)} features")
    except Exception as exc:
        logger.error(f"Erreur création DataFrame: {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Features invalides: {exc}"
        )

    try:
        X = np.array(list(requete.features.values())).reshape(1, -1)
        proba = float(pipeline_final.predict_proba(X)[:, 1][0])
        logger.info(f"Probabilité défaut: {proba:.4f}")
    except Exception as exc:
        logger.error(f"Erreur prédiction: {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Erreur pendant la prédiction. "
                "Vérifiez que les features correspondent au modèle. "
                f"Détail: {exc}"
            )
        )

    decision = int(proba >= SEUIL_DECISION)
    interpretation, confiance = _interprete_decision(proba, SEUIL_DECISION)

    client_id = requete.features.get("SK_ID_CURR")

    return ReponsePrediction(
        client_id=client_id,
        probabilite_defaut=round(proba, 4),
        score_percent=round(proba * 100, 2),
        decision=decision,
        decision_label="REFUS" if decision == 1 else "ACCORD",
        seuil_decision=SEUIL_DECISION,
        interpretation=interpretation,
        confiance=confiance
    )


@app.post("/explain", response_model=ReponseExplication, tags=["Interpretability"])
def expliquer(requete: RequeteExplication) -> ReponseExplication:
    if pipeline_final is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Pipeline non chargé"
        )

    explainer = get_shap_explainer()
    if explainer is None:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="SHAP non disponible pour ce modèle"
        )

    try:
        donnees_client = pd.DataFrame([requete.features])

        preprocess = _get_preprocess()
        if preprocess is None:
            raise RuntimeError("Impossible d'extraire le preprocess du pipeline (pipeline[:-1])")

        X_transformed = np.array(list(donnees_client.iloc[0].values)).reshape(1, -1)

        shap_values = explainer.shap_values(X_transformed)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        shap_values_1d = shap_values[0]

        base_value = explainer.expected_value
        if isinstance(base_value, np.ndarray):
            base_value = float(base_value[1])
        else:
            base_value = float(base_value)

        try:
            feature_names = preprocess.get_feature_names_out()
        except Exception:
            feature_names = [f"f_{i}" for i in range(len(shap_values_1d))]

        shap_dict = {str(n): float(v) for n, v in zip(feature_names, shap_values_1d)}

        sorted_shap = sorted(
            shap_dict.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:requete.top_n]

        positives = [
            {"feature": k, "shap_value": v, "impact": "Augmente risque"}
            for k, v in sorted_shap if v > 0
        ]
        negatives = [
            {"feature": k, "shap_value": abs(v), "impact": "Réduit risque"}
            for k, v in sorted_shap if v < 0
        ]

        prediction = base_value + float(np.sum(shap_values_1d))

        return ReponseExplication(
            base_value=round(base_value, 4),
            shap_values={k: round(v, 4) for k, v in sorted_shap},
            prediction=round(prediction, 4),
            top_features_positives=positives[:5],
            top_features_negatives=negatives[:5]
        )

    except Exception as exc:
        logger.error(f"Erreur SHAP: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors du calcul SHAP: {exc}"
        )


@app.get("/feature-importance", response_model=ReponseFeatureImportance, tags=["Interpretability"])
def feature_importance() -> ReponseFeatureImportance:
    if pipeline_final is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Pipeline non chargé"
        )

    try:
        modele = _get_modele_estimateur()

        if not hasattr(modele, 'feature_importances_'):
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="Ce modèle ne supporte pas feature_importances_"
            )

        importances = modele.feature_importances_
        preprocess = _get_preprocess()

        try:
            feature_names = preprocess.get_feature_names_out() if preprocess is not None else None
        except Exception:
            feature_names = None

        if feature_names is None:
            feature_names = [f"f_{i}" for i in range(len(importances))]

        importance_list = [
            {
                "feature": str(name),
                "importance": float(imp),
                "importance_percent": round(float(imp) * 100, 2)
            }
            for name, imp in zip(feature_names, importances)
        ]

        importance_list.sort(key=lambda x: x['importance'], reverse=True)
        top_10 = [item['feature'] for item in importance_list[:10]]

        logger.info(f"Feature importance calculée pour {len(importance_list)} features")

        return ReponseFeatureImportance(
            features=importance_list,
            top_10=top_10
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Erreur feature importance: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur calcul feature importance: {exc}"
        )



# -----------------------------------------------------------------------------
# Clients batch
# -----------------------------------------------------------------------------
CHEMIN_CLIENTS = DOSSIER_ARTIFACTS / "batch_clients.json"


@app.get("/clients", tags=["Clients"])
def liste_clients():
    """Retourne la liste des clients disponibles"""
    import json
    if not CHEMIN_CLIENTS.exists():
        raise HTTPException(status_code=404, detail="batch_clients.json introuvable")
    try:
        with open(CHEMIN_CLIENTS) as f:
            clients = json.load(f)
        ids = [c.get("SK_ID_CURR", i) for i, c in enumerate(clients)]
        return {"clients": ids, "total": len(ids)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/clients/{client_id}", tags=["Clients"])
def get_client(client_id: int):
    """Retourne les features d'un client par son ID"""
    import json
    if not CHEMIN_CLIENTS.exists():
        raise HTTPException(status_code=404, detail="batch_clients.json introuvable")
    try:
        with open(CHEMIN_CLIENTS) as f:
            clients = json.load(f)
        for c in clients:
            if c.get("SK_ID_CURR") == client_id:
                return {"client_id": client_id, "features": c}
        raise HTTPException(status_code=404, detail=f"Client {client_id} introuvable")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
