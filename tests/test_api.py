"""
Tests unitaires - API Scoring Credit
Projet 7 OpenClassrooms
"""

import pytest
import numpy as np
import sys
import os
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Mock du modele pour eviter de charger les fichiers joblib
mock_model = MagicMock()
mock_model.feature_name_ = [f"FEATURE_{i}" for i in range(804)]
mock_model.feature_importances_ = np.ones(804) / 804
mock_model.n_features_ = 804
mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])

mock_explainer = MagicMock()
mock_explainer.expected_value = -2.5
mock_explainer.shap_values.return_value = np.random.uniform(-0.1, 0.1, (1, 804))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))

with patch("joblib.load", return_value=mock_model), \
     patch("shap.TreeExplainer", return_value=mock_explainer):
    try:
        from main import app
        client = TestClient(app)
    except Exception as e:
        print(f"Erreur import: {e}")


def test_health():
    """Verifie que le endpoint health repond avec le bon format."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "version" in data
    assert "model_loaded" in data


def test_predict_features_valides():
    """Verifie que predict retourne une prediction correcte."""
    features = {f"FEATURE_{i}": 0.5 for i in range(50)}
    response = client.post("/predict", json={"features": features})
    assert response.status_code == 200
    data = response.json()
    assert "probabilite_defaut" in data
    assert "decision_label" in data
    assert "seuil_decision" in data
    assert 0.0 <= data["probabilite_defaut"] <= 1.0
    assert data["decision_label"] in ["ACCORD", "REFUS"]


def test_predict_features_vides():
    """Verifie que predict rejette une requete sans features."""
    response = client.post("/predict", json={"features": {}})
    assert response.status_code in [400, 422, 500]


def test_root():
    """Verifie que la racine de l API repond."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
