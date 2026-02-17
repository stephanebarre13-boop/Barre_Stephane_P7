"""
API Scoring Crédit - Projet 7 OpenClassrooms
Version corrigée complète - Compatible dashboard Streamlit
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ===============================
# FIX JOBLIB (convert_to_string)
# ===============================
import __main__
from utils_serialization import convert_to_string
__main__.convert_to_string = convert_to_string

# ===============================
# LOGGING
# ===============================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("credit_api")

# ===============================
# PATHS
# ===============================
API_DIR = Path(__file__).resolve().parent
ROOT_DIR = API_DIR.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "meilleur_modele.joblib"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocesseur.joblib"
PARAMS_PATH = ARTIFACTS_DIR / "parametres_decision.joblib"
API_VERSION = "2.0.0"

# ===============================
# LOAD ARTIFACTS
# ===============================
try:
    logger.info(f"Chargement modèle : {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    model_load_error = None
except Exception as e:
    model = None
    model_load_error = str(e)
    logger.error(f"Erreur chargement modèle : {e}")

try:
    logger.info(f"Chargement préprocesseur : {PREPROCESSOR_PATH}")
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    preprocessor_load_error = None
except Exception as e:
    preprocessor = None
    preprocessor_load_error = str(e)
    logger.warning(f"Préprocesseur non chargé : {e}")

THRESHOLD = 0.5
try:
    if PARAMS_PATH.exists():
        params = joblib.load(PARAMS_PATH)
        if isinstance(params, dict):
            THRESHOLD = float(params.get("seuil_optimal", 0.5))
            logger.info(f"Seuil chargé : {THRESHOLD}")
except Exception:
    pass

# ===============================
# FEATURE NAMES
# ===============================
def get_model_feature_names(m):
    if m is None:
        return None
    if hasattr(m, "feature_name_"):
        names = list(m.feature_name_)
        if names:
            return names
    try:
        if hasattr(m, "booster_"):
            return list(m.booster_.feature_name())
    except Exception:
        pass
    return None

MODEL_FEATURE_NAMES = get_model_feature_names(model)
if MODEL_FEATURE_NAMES:
    logger.info(f"Modèle : {type(model).__name__} | {len(MODEL_FEATURE_NAMES)} features | ex: {MODEL_FEATURE_NAMES[:5]}")

# ===============================
# UTILS
# ===============================
def ensure_2d_array(x):
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return x


def build_transformed_row(features: Dict[str, Any]):
    if model is None:
        raise RuntimeError("Modèle non chargé")
    if MODEL_FEATURE_NAMES is None:
        raise RuntimeError("MODEL_FEATURE_NAMES non disponible")

    has_feature_format = any(k.startswith("FEATURE_") for k in features)

    # CAS 1 : FEATURE_XX (dashboard) → remappé par index
    if has_feature_format:
        remapped: Dict[str, Any] = {}
        for k, v in features.items():
            if k.startswith("FEATURE_"):
                try:
                    idx = int(k.replace("FEATURE_", ""))
                    if 0 <= idx < len(MODEL_FEATURE_NAMES):
                        remapped[MODEL_FEATURE_NAMES[idx]] = v
                except ValueError:
                    pass
        df = pd.DataFrame([remapped])
        df = df.reindex(columns=MODEL_FEATURE_NAMES, fill_value=0.0)
        X = ensure_2d_array(df.values)
        logger.info(f"CAS 1 (FEATURE_XX) : {X.shape[1]} features")
        return X, MODEL_FEATURE_NAMES

    # CAS 2 : Column_XX ou colonnes déjà alignées
    df = pd.DataFrame([features])
    df = df.reindex(columns=MODEL_FEATURE_NAMES, fill_value=0.0)
    X = ensure_2d_array(df.values)
    logger.info(f"CAS 2 (reindex) : {X.shape[1]} features")
    return X, MODEL_FEATURE_NAMES


def interpret_decision(prob: float, threshold: float):
    gap = abs(prob - threshold)
    if gap > 0.2:
        confidence = "HAUTE"
    elif gap > 0.1:
        confidence = "MOYENNE"
    else:
        confidence = "FAIBLE (proche du seuil)"
    if prob >= threshold:
        interpretation = (
            f"Risque de défaut ÉLEVÉ ({prob:.1%}). Recommandation : REFUS. "
            f"Le client dépasse le seuil métier de {threshold:.1%}."
        )
    else:
        interpretation = (
            f"Risque de défaut FAIBLE ({prob:.1%}). Recommandation : ACCORD possible. "
            f"Le client est en-dessous du seuil métier de {threshold:.1%}."
        )
    return interpretation, confidence

# ===============================
# SHAP
# ===============================
_shap_explainer = None

def get_shap_explainer():
    global _shap_explainer
    if _shap_explainer is None and model is not None:
        try:
            logger.info("Initialisation SHAP TreeExplainer...")
            _shap_explainer = shap.TreeExplainer(model)
            logger.info("SHAP prêt")
        except Exception as e:
            logger.warning(f"SHAP indisponible : {e}")
    return _shap_explainer

# ===============================
# SCHEMAS
# ===============================
class PredictionRequest(BaseModel):
    features: Dict[str, Any]

class PredictionResponse(BaseModel):
    probabilite_defaut: float
    score_percent: float
    decision: int
    decision_label: str
    seuil_decision: float
    interpretation: str
    confiance: str
    client_id: Optional[int] = None

class ExplainRequest(BaseModel):
    features: Dict[str, Any]
    top_n: int = 10

class ExplainResponse(BaseModel):
    base_value: float
    prediction: float
    shap_values: Dict[str, float]
    top_features_positives: List[Dict[str, Any]]
    top_features_negatives: List[Dict[str, Any]]

class FeatureImportanceResponse(BaseModel):
    features: List[Dict[str, Any]]
    top_10: List[str]

# ===============================
# FASTAPI
# ===============================
app = FastAPI(
    title="API Scoring Crédit – Projet 7",
    version=API_VERSION,
    description="API de scoring crédit avec prédiction, SHAP et feature importance.",
    docs_url="/docs",
    redoc_url="/redoc",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ===============================
# ROUTES
# ===============================
@app.get("/")
def root():
    return {"message": "API Scoring Crédit P7", "version": API_VERSION, "docs": "/docs"}


@app.get("/health")
def health():
    shap_ok = get_shap_explainer() is not None
    return {
        "status": "ok" if model is not None else "ko",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "pipeline_charge": model is not None,
        "shap_disponible": shap_ok,
        "shap_available": shap_ok,
        "version": API_VERSION,
        "error": model_load_error or preprocessor_load_error,
    }


@app.get("/model-info")
def model_info():
    if model is None:
        raise HTTPException(status_code=500, detail=f"Modèle non chargé : {model_load_error}")
    n = len(MODEL_FEATURE_NAMES) if MODEL_FEATURE_NAMES else getattr(model, "n_features_", None)
    return {
        "type_modele": type(model).__name__,
        "n_features": n,
        "seuil_decision": THRESHOLD,
        "cout_fn": 10,
        "cout_fp": 1,
        "formule_cout": "Coût = 10×FN + 1×FP",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail=f"Modèle non chargé : {model_load_error}")
    try:
        X, _ = build_transformed_row(req.features)
        proba = float(model.predict_proba(X)[:, 1][0])
    except Exception as e:
        logger.exception("Erreur prédiction")
        raise HTTPException(status_code=400, detail=f"Erreur prédiction : {e}")

    decision = int(proba >= THRESHOLD)
    interpretation, confiance = interpret_decision(proba, THRESHOLD)
    return PredictionResponse(
        probabilite_defaut=round(proba, 4),
        score_percent=round(proba * 100, 2),
        decision=decision,
        decision_label="REFUS" if decision == 1 else "ACCORD",
        seuil_decision=THRESHOLD,
        interpretation=interpretation,
        confiance=confiance,
    )


@app.post("/explain", response_model=ExplainResponse)
def explain(req: ExplainRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Modèle non chargé")
    explainer = get_shap_explainer()
    if explainer is None:
        raise HTTPException(status_code=501, detail="SHAP non disponible")
    try:
        X, feature_names = build_transformed_row(req.features)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        sv = np.asarray(shap_values)[0]
        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = float(np.asarray(base_value)[-1])
        else:
            base_value = float(base_value)
        shap_dict = {str(n): float(v) for n, v in zip(feature_names, sv)}
        top = sorted(shap_dict.items(), key=lambda kv: abs(kv[1]), reverse=True)[: req.top_n]
        positives = [{"feature": k, "shap_value": float(v), "impact": "Augmente risque"} for k, v in top if v > 0]
        negatives = [{"feature": k, "shap_value": float(abs(v)), "impact": "Réduit risque"} for k, v in top if v < 0]
        return ExplainResponse(
            base_value=round(base_value, 6),
            prediction=round(base_value + float(np.sum(sv)), 6),
            shap_values={k: round(v, 6) for k, v in top},
            top_features_positives=positives[:5],
            top_features_negatives=negatives[:5],
        )
    except Exception as e:
        logger.exception("Erreur SHAP")
        raise HTTPException(status_code=500, detail=f"Erreur SHAP : {e}")


@app.get("/feature-importance", response_model=FeatureImportanceResponse)
def feature_importance():
    if model is None:
        raise HTTPException(status_code=500, detail="Modèle non chargé")
    if not hasattr(model, "feature_importances_"):
        raise HTTPException(status_code=501, detail="feature_importances_ non disponible")
    try:
        importances = np.asarray(model.feature_importances_, dtype=float)
        names = MODEL_FEATURE_NAMES or [f"f_{i}" for i in range(len(importances))]
        total = float(np.sum(importances)) or 1.0
        rows = [
            {"feature": str(n), "importance": float(imp), "importance_percent": round(float(imp) / total * 100, 4)}
            for n, imp in zip(names, importances)
        ]
        rows.sort(key=lambda r: r["importance"], reverse=True)
        return FeatureImportanceResponse(features=rows, top_10=[r["feature"] for r in rows[:10]])
    except Exception as e:
        logger.exception("Erreur feature importance")
        raise HTTPException(status_code=500, detail=f"Erreur : {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
