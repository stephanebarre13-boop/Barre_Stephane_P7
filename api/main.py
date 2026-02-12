"""
API Scoring Crédit - Projet 7 OpenClassrooms

Description :
- API de prédiction du risque de défaut de paiement
- Modèle : LightGBM Classifier optimisé (804 features)
- Explainability via SHAP values

Endpoints :
- GET  /health
- GET  /model-info
- POST /predict
- POST /explain
- GET  /feature-importance
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("credit_api")

# -----------------------------------------------------------------------------
# Helpers needed for joblib unpickling (IMPORTANT)
# Some sklearn pipelines include FunctionTransformer with a custom function.
# It must exist at module level with the same name to unpickle.
# -----------------------------------------------------------------------------
def convert_to_string(x):
    if isinstance(x, (pd.Series, pd.DataFrame)):
        return x.astype(str)
    if isinstance(x, np.ndarray):
        return x.astype(str)
    return x


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
API_DIR = Path(__file__).resolve().parent
ROOT_DIR = API_DIR.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

MODEL_PATH = ARTIFACTS_DIR / "meilleur_modele.joblib"          # LGBMClassifier optimisé
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocesseur.joblib"     # Pipeline de prétraitement
PARAMS_PATH = ARTIFACTS_DIR / "parametres_decision.joblib"     # seuil/couts (dict)

API_VERSION = "2.0.0"


# -----------------------------------------------------------------------------
# Load artifacts
# -----------------------------------------------------------------------------
def load_model() -> Any:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modèle introuvable: {MODEL_PATH}")
    logger.info("Chargement modèle: %s", MODEL_PATH)
    return joblib.load(MODEL_PATH)


def load_preprocessor() -> Optional[Any]:
    if not PREPROCESSOR_PATH.exists():
        logger.warning("Préprocesseur introuvable: %s (on ne pourra pas transformer les features métier)", PREPROCESSOR_PATH)
        return None
    logger.info("Chargement préprocesseur: %s", PREPROCESSOR_PATH)
    return joblib.load(PREPROCESSOR_PATH)


def load_business_params() -> Dict[str, Any]:
    if PARAMS_PATH.exists():
        try:
            params = joblib.load(PARAMS_PATH)
            if isinstance(params, dict):
                logger.info("Paramètres métier chargés: %s", params)
                return params
        except Exception as exc:
            logger.warning("Impossible de charger %s: %s", PARAMS_PATH, exc)

    logger.warning("Paramètres par défaut utilisés")
    return {"modele": "inconnu", "seuil_optimal": 0.5, "cout_fn": 10, "cout_fp": 1}


# Globals loaded at import
try:
    model = load_model()
    model_load_error = None
except Exception as exc:
    model = None
    model_load_error = str(exc)
    logger.error("Erreur chargement modèle: %s", exc)

try:
    preprocessor = load_preprocessor() if model is not None else None
    preprocessor_load_error = None
except Exception as exc:
    preprocessor = None
    preprocessor_load_error = str(exc)
    logger.error("Erreur chargement préprocesseur: %s", exc)

business_params = load_business_params()
THRESHOLD = float(business_params.get("seuil_optimal", 0.5))

# SHAP explainer (lazy)
_shap_explainer: Optional[Any] = None


# -----------------------------------------------------------------------------
# Model utilities
# -----------------------------------------------------------------------------
def get_model_feature_names(m: Any) -> Optional[List[str]]:
    """
    For LightGBM sklearn wrapper, booster_.feature_name() returns the names used at training.
    In your artifacts: Column_0..Column_803
    """
    if m is None:
        return None
    try:
        if hasattr(m, "booster_") and m.booster_ is not None:
            names = list(m.booster_.feature_name())
            return names if names else None
    except Exception:
        pass
    # fallback sometimes stored here
    names = getattr(m, "feature_name_", None)
    if isinstance(names, (list, tuple)) and names:
        return list(names)
    return None


MODEL_FEATURE_NAMES = get_model_feature_names(model)
MODEL_N_FEATURES = getattr(model, "n_features_", None) if model is not None else None

if MODEL_FEATURE_NAMES is not None:
    logger.info("Modèle: %s | n_features=%s | first_5=%s",
                type(model).__name__, len(MODEL_FEATURE_NAMES), MODEL_FEATURE_NAMES[:5])
else:
    logger.info("Modèle: %s | n_features=%s", type(model).__name__ if model is not None else None, MODEL_N_FEATURES)


def ensure_2d_array(x: Any) -> np.ndarray:
    if hasattr(x, "toarray"):  # sparse
        x = x.toarray()
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return x


def expand_features_to_804(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expand 50 FEATURE_XX to 804 Column_0..Column_803 (FIXED FOR DASHBOARD)
    """
    # Si déjà des Column_*, on retourne tel quel
    if any(k.startswith("Column_") for k in features.keys()):
        return features
    
    # Sinon, on mappe FEATURE_XX vers Column_XX et on remplit le reste avec 0
    expanded = {}
    
    # Mapping FEATURE_XX -> Column_XX
    for feat_key, feat_val in features.items():
        if feat_key.startswith("FEATURE_"):
            col_num = feat_key.replace("FEATURE_", "")
            expanded[f"Column_{col_num}"] = feat_val
    
    # Remplir tous les Column_0..Column_803 manquants avec 0.0
    for i in range(804):
        col_name = f"Column_{i}"
        if col_name not in expanded:
            expanded[col_name] = 0.0
    
    logger.info(f"✅ Expanded {len(features)} FEATURE_XX to 804 Column_* features")
    return expanded


def build_transformed_row(features: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
    """
    Returns (X_transformed, feature_names) aligned with the trained model.
    
    NEW VERSION: 
    - Détecte si colonnes métier (50 colonnes) → utilise preprocessor
    - Sinon FEATURE_XX → expand to Column_0..803
    - Sinon Column_XX → utilise tel quel
    """
    if model is None:
        raise RuntimeError("Modèle non chargé")
    
    # Détection du type de features reçues
    feature_keys = list(features.keys())
    has_column_format = any(k.startswith("Column_") for k in feature_keys)
    has_feature_format = any(k.startswith("FEATURE_") for k in feature_keys)
    
    # CAS 1: Colonnes métier brutes (ex: AGE, INCOME, etc.) → utiliser preprocessor
    if not has_column_format and not has_feature_format and preprocessor is not None:
        logger.info("🔄 Colonnes métier détectées → transformation via preprocessor")
        try:
            df_input = pd.DataFrame([features])
            X_transformed = preprocessor.transform(df_input)
            X = ensure_2d_array(X_transformed)
            
            if MODEL_FEATURE_NAMES is None:
                raise RuntimeError("MODEL_FEATURE_NAMES non disponible")
            
            # Vérifier que la transformation a bien produit 804 features
            if X.shape[1] != len(MODEL_FEATURE_NAMES):
                logger.warning(
                    f"⚠️ Preprocessor a produit {X.shape[1]} features, "
                    f"attendu {len(MODEL_FEATURE_NAMES)}. Padding avec 0."
                )
                if X.shape[1] < len(MODEL_FEATURE_NAMES):
                    padding = np.zeros((1, len(MODEL_FEATURE_NAMES) - X.shape[1]))
                    X = np.hstack([X, padding])
                else:
                    X = X[:, :len(MODEL_FEATURE_NAMES)]
            
            logger.info(f"✅ Transformation réussie: {X.shape[1]} features")
            return X, MODEL_FEATURE_NAMES
            
        except Exception as exc:
            logger.error(f"❌ Erreur transformation preprocessor: {exc}")
            raise RuntimeError(f"Erreur preprocessing: {exc}")
    
    # CAS 2: Format FEATURE_XX → expand to Column_0..803
    if has_feature_format:
        logger.info("🔄 Format FEATURE_XX détecté → expansion vers Column_*")
        features = expand_features_to_804(features)
    
    # CAS 3: Format Column_XX ou après expansion
    if MODEL_FEATURE_NAMES is None:
        raise RuntimeError("MODEL_FEATURE_NAMES non disponible")
    
    df = pd.DataFrame([features])
    df = df.reindex(columns=MODEL_FEATURE_NAMES, fill_value=0.0)
    X = ensure_2d_array(df.values)
    
    logger.info(f"✅ Features alignées: {X.shape[1]} colonnes")
    return X, MODEL_FEATURE_NAMES


def get_shap_explainer() -> Optional[Any]:
    global _shap_explainer
    if _shap_explainer is not None:
        return _shap_explainer
    if model is None:
        return None
    try:
        logger.info("Initialisation SHAP TreeExplainer…")
        _shap_explainer = shap.TreeExplainer(model)
        logger.info("✅ SHAP explainer prêt")
    except Exception as exc:
        logger.warning("⚠️ SHAP indisponible: %s", exc)
        _shap_explainer = None
    return _shap_explainer


def interpret_decision(prob: float, threshold: float) -> Tuple[str, str]:
    gap = abs(prob - threshold)
    if gap > 0.2:
        confidence = "HAUTE"
    elif gap > 0.1:
        confidence = "MOYENNE"
    else:
        confidence = "FAIBLE (proche du seuil)"

    if prob >= threshold:
        interpretation = (
            f"⚠️ Risque de défaut ÉLEVÉ ({prob:.1%}). "
            f"Recommandation : REFUS ou analyse approfondie. "
            f"Le client dépasse le seuil métier de {threshold:.1%}."
        )
    else:
        interpretation = (
            f"✅ Risque de défaut FAIBLE ({prob:.1%}). "
            f"Recommandation : ACCORD possible (sous réserve règles internes). "
            f"Le client est en-dessous du seuil métier de {threshold:.1%}."
        )
    return interpretation, confidence


# -----------------------------------------------------------------------------
# Schemas (Pydantic v2)
# -----------------------------------------------------------------------------
class PredictionRequest(BaseModel):
    features: Dict[str, Any] = Field(..., description="Features client (métier) OU déjà préprocessées (Column_*)")


    @field_validator("features")
    @classmethod
    def features_not_empty(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        if not v:
            raise ValueError("features ne peut pas être vide")
        return v


class PredictionResponse(BaseModel):
    client_id: Optional[int] = Field(None, description="ID client (si fourni, ex: SK_ID_CURR)")
    probabilite_defaut: float = Field(..., description="Probabilité de défaut [0-1]")
    score_percent: float = Field(..., description="Score en pourcentage [0-100]")
    decision: int = Field(..., description="0=ACCORD, 1=REFUS")
    decision_label: str = Field(..., description="ACCORD ou REFUS")
    seuil_decision: float = Field(..., description="Seuil métier (optimal)")
    interpretation: str = Field(..., description="Texte d'interprétation")
    confiance: str = Field(..., description="Niveau de confiance")


class ExplainRequest(BaseModel):
    features: Dict[str, Any] = Field(..., description="Features client (métier) OU déjà préprocessées (Column_*)")
    top_n: int = Field(10, ge=1, le=50, description="Nombre de features à afficher")


class ExplainResponse(BaseModel):
    base_value: float
    prediction: float
    shap_values: Dict[str, float]
    top_features_positives: List[Dict[str, Any]]
    top_features_negatives: List[Dict[str, Any]]


class FeatureImportanceResponse(BaseModel):
    features: List[Dict[str, Any]]
    top_10: List[str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    preprocessor_loaded: bool
    pipeline_charge: bool  # Alias français pour compatibilité Dashboard
    shap_disponible: bool  # Alias français pour compatibilité Dashboard
    shap_available: bool   # Keep English version for compatibility
    version: str
    error: Optional[str] = None


# -----------------------------------------------------------------------------
# FastAPI
# -----------------------------------------------------------------------------
app = FastAPI(
    title="API Scoring Crédit PREMIUM – Projet 7",
    version=API_VERSION,
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
# Routes
# -----------------------------------------------------------------------------
@app.get("/", tags=["Root"])
def root():
    return {
        "message": "API Scoring Crédit PREMIUM",
        "version": API_VERSION,
        "endpoints": {
            "health": "/health",
            "model_info": "/model-info",
            "predict": "/predict (POST)",
            "explain": "/explain (POST)",
            "feature_importance": "/feature-importance",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health():
    shap_ok = get_shap_explainer() is not None
    err = model_load_error or preprocessor_load_error
    # Le pipeline est OK si le modèle est chargé (préprocesseur optionnel car Dashboard envoie features transformées)
    pipeline_ok = model is not None
    return HealthResponse(
        status="ok" if model is not None else "ko",
        model_loaded=model is not None,
        preprocessor_loaded=preprocessor is not None,
        pipeline_charge=pipeline_ok,  # OK si modèle chargé
        shap_disponible=shap_ok,
        shap_available=shap_ok,
        version=API_VERSION,
        error=err,
    )


@app.get("/model-info", tags=["Model"])
def model_info() -> Dict[str, Any]:
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Modèle non chargé: {model_load_error}",
        )

    model_name = type(model).__name__
    n_features = MODEL_N_FEATURES
    if MODEL_FEATURE_NAMES is not None:
        n_features = len(MODEL_FEATURE_NAMES)

    return {
        "modele": business_params.get("modele", model_name),
        "type_modele": model_name,
        "n_features": n_features,
        "seuil_decision": THRESHOLD,
        "cout_fn": business_params.get("cout_fn", 10),
        "cout_fp": business_params.get("cout_fp", 1),
        "formule_cout": "Coût = 10×FN + 1×FP",
        "artefacts": {
            "modele": MODEL_PATH.name,
            "preprocesseur": PREPROCESSOR_PATH.name if PREPROCESSOR_PATH.exists() else None,
            "parametres": PARAMS_PATH.name if PARAMS_PATH.exists() else None,
        },
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(req: PredictionRequest) -> PredictionResponse:
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Modèle non chargé: {model_load_error}",
        )

    try:
        X, _names = build_transformed_row(req.features)
        proba = float(model.predict_proba(X)[:, 1][0])
    except Exception as exc:
        logger.exception("Erreur prédiction")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Erreur pendant la prédiction. "
                "Vérifiez que les features correspondent au modèle (features métier) "
                "ou envoyez Column_0..Column_803. "
                f"Détail: {exc}"
            ),
        )

    decision = int(proba >= THRESHOLD)
    interpretation, confidence = interpret_decision(proba, THRESHOLD)

    client_id = req.features.get("SK_ID_CURR")
    if isinstance(client_id, (np.integer,)):
        client_id = int(client_id)

    return PredictionResponse(
        client_id=client_id if isinstance(client_id, int) else None,
        probabilite_defaut=round(proba, 4),
        score_percent=round(proba * 100.0, 2),
        decision=decision,
        decision_label="REFUS" if decision == 1 else "ACCORD",
        seuil_decision=THRESHOLD,
        interpretation=interpretation,
        confiance=confidence,
    )


@app.post("/explain", response_model=ExplainResponse, tags=["Interpretability"])
def explain(req: ExplainRequest) -> ExplainResponse:
    if model is None:
        raise HTTPException(status_code=500, detail="Modèle non chargé")

    explainer = get_shap_explainer()
    if explainer is None:
        raise HTTPException(status_code=501, detail="SHAP non disponible pour ce modèle")

    try:
        X, feature_names = build_transformed_row(req.features)

        # SHAP values (binary classifier -> sometimes returns list[class0, class1])
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        sv = np.asarray(shap_values)[0]  # (n_features,)

        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            # take class 1 base if provided
            base_value = float(np.asarray(base_value)[-1])
        else:
            base_value = float(base_value)

        # build dict & top_n by abs impact
        shap_dict = {str(n): float(v) for n, v in zip(feature_names, sv)}
        top = sorted(shap_dict.items(), key=lambda kv: abs(kv[1]), reverse=True)[: req.top_n]

        positives = [{"feature": k, "shap_value": float(v), "impact": "Augmente risque"} for k, v in top if v > 0]
        negatives = [{"feature": k, "shap_value": float(abs(v)), "impact": "Réduit risque"} for k, v in top if v < 0]

        # For log-odds models, base+sum is not a probability; we return it as "raw" explanation score.
        raw_pred = base_value + float(np.sum(sv))

        return ExplainResponse(
            base_value=round(base_value, 6),
            prediction=round(raw_pred, 6),
            shap_values={k: round(v, 6) for k, v in top},
            top_features_positives=positives[:5],
            top_features_negatives=negatives[:5],
        )
    except Exception as exc:
        logger.exception("Erreur SHAP")
        raise HTTPException(status_code=500, detail=f"Erreur lors du calcul SHAP: {exc}")


@app.get("/feature-importance", response_model=FeatureImportanceResponse, tags=["Interpretability"])
def feature_importance() -> FeatureImportanceResponse:
    if model is None:
        raise HTTPException(status_code=500, detail="Modèle non chargé")

    if not hasattr(model, "feature_importances_"):
        raise HTTPException(status_code=501, detail="Ce modèle ne supporte pas feature_importances_")

    try:
        importances = np.asarray(model.feature_importances_, dtype=float)
        names = MODEL_FEATURE_NAMES or [f"f_{i}" for i in range(len(importances))]

        rows = []
        total = float(np.sum(importances)) if float(np.sum(importances)) > 0 else 1.0
        for name, imp in zip(names, importances):
            rows.append(
                {
                    "feature": str(name),
                    "importance": float(imp),
                    "importance_percent": round(float(imp) / total * 100.0, 4),
                }
            )

        rows.sort(key=lambda r: r["importance"], reverse=True)
        top_10 = [r["feature"] for r in rows[:10]]
        return FeatureImportanceResponse(features=rows, top_10=top_10)

    except Exception as exc:
        logger.exception("Erreur feature importance")
        raise HTTPException(status_code=500, detail=f"Erreur calcul feature importance: {exc}")


# -----------------------------------------------------------------------------
# Local entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
