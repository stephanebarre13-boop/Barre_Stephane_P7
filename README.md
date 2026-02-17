# Projet 7 - Implementez un modèle de scoring

[![CI Tests](https://github.com/stephanebarre13-boop/Barre_Stephane_P7/actions/workflows/main.yml/badge.svg)](https://github.com/stephanebarre13-boop/Barre_Stephane_P7/actions/workflows/main.yml)

**Auteur :** Stéphane BARRE
**Formation :** OpenClassrooms - Parcours Data Scientist  
**Date :** Février 2026

---

## 📋 Description du projet

Développement d'un système de scoring crédit pour la société financière "Prêt à dépenser", permettant de prédire la probabilité de défaut de paiement d'un client et d'améliorer la transparence des décisions de crédit.

### Objectifs

- Construire un modèle de machine learning pour prédire le risque de défaut
- Optimiser le seuil de décision selon les coûts métier (ratio 10:1)
- Développer une API pour les prédictions en temps réel
- Créer un dashboard interactif avec explainability (SHAP)
- Implémenter un système de monitoring des dérives (data drift)

---

## 🎯 Livrables

### 1. Notebooks d'analyse (7 notebooks)

1. **NB01 - Agrégation des tables** : Transformation des 122 features initiales en 804 features agrégées
2. **NB02 - Pipeline de préparation** : Preprocessing et feature engineering
3. **NB03 - Comparaison des modèles** : Benchmark de différents algorithmes
4. **NB04 - Gestion du déséquilibre** : Application de SMOTE et techniques de rééchantillonnage
5. **NB05 - Optimisation du seuil** : Calibration selon les coûts métier (FN=10, FP=1)
6. **NB06 - Interprétabilité SHAP** : Explainability globale et locale des prédictions
7. **NB07 - Monitoring data drift** : Détection des dérives avec Evidently

### 2. API FastAPI

- Endpoint de prédiction
- Endpoint d'explainability (SHAP values)
- Health check et informations du modèle
- Dockerisé pour déploiement

### 3. Dashboard Streamlit

- Interface pour chargés de relation client
- Prédiction en temps réel avec jauge visuelle
- Graphiques SHAP interactifs
- Mapping des features vers libellés métier
- Historique des décisions

---

## 📁 Structure du projet
```
Barre_Stephane_P7/
├── README.md                                     # Ce fichier
├── .gitignore                                    # Fichiers exclus de Git
│
├── Barre_Stephane_P7_01_aggregation_tables.ipynb
├── Barre_Stephane_P7_02_preparation_pipeline.ipynb
├── Barre_Stephane_P7_03_comparaison_modeles.ipynb
├── Barre_Stephane_P7_04_desequilibre.ipynb
├── Barre_Stephane_P7_05_optimisation_seuil.ipynb
├── Barre_Stephane_P7_06_interpretabilite_shap.ipynb
├── Barre_Stephane_P7_07_data_drift.ipynb
│
├── api/                                          # API FastAPI
│   ├── main.py                                   # Code principal de l'API
│   ├── requirements.txt                          # Dépendances Python
│   └── Dockerfile                                # Configuration Docker
│
├── dashboard/                                    # Dashboard Streamlit
│   ├── app.py                                    # Application Streamlit
│   ├── requirements.txt                          # Dépendances Python
│   └── Dockerfile                                # Configuration Docker
│
├── docs/                                         # Documentation
│   ├── DATA_STRUCTURE.md                         # Description des données
│   └── database_schema.png                       # Schéma de la base
│
├── reports/                                      # Rapports générés
│   └── .gitkeep                                  # Maintient le dossier dans Git
│
├── scripts/                                      # Scripts utilitaires
│   └── .gitkeep                                  # Maintient le dossier dans Git
│
└── test_samples_backup/                          # Échantillons de test
    ├── batch_clients.json                        # Batch de clients
    ├── client_high_risk.json                     # Client à haut risque
    ├── client_low_risk.json                      # Client à faible risque
    ├── client_mixed.json                         # Client mixte
    ├── client_zeros.json                         # Client avec valeurs nulles
    └── README.md                                 # Documentation des échantillons
```

---

## 🚀 Installation et utilisation

### Prérequis

- Python 3.10 ou supérieur
- pip (gestionnaire de paquets Python)
- Git

### Installation

**1. Cloner le repository**
```bash
git clone https://github.com/stephanebarre13-boop/Barre_Stephane_P7.git
cd Barre_Stephane_P7
```

**2. Télécharger les données**

Les données sources proviennent du challenge Kaggle [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/data).

Télécharger et placer les fichiers CSV dans un dossier `/data/` :
- `application_train.csv`
- `application_test.csv`
- `bureau.csv`
- `bureau_balance.csv`
- `credit_card_balance.csv`
- `installments_payments.csv`
- `POS_CASH_balance.csv`
- `previous_application.csv`

**3. Générer les modèles**

Exécuter les notebooks dans l'ordre (01 à 05) pour générer les artifacts dans `/artifacts/`.

---

### Lancement de l'API
```bash
cd api
pip install -r requirements.txt
uvicorn main:app --reload
```

L'API sera accessible sur `http://localhost:8000`

Documentation interactive : `http://localhost:8000/docs`

---

### Lancement du dashboard
```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

Le dashboard sera accessible sur `http://localhost:8501`

---

## 🛠️ Technologies utilisées

### Machine Learning & Data Science
- **pandas** : Manipulation de données
- **numpy** : Calculs numériques
- **scikit-learn** : Preprocessing et métriques
- **LightGBM** : Algorithme de gradient boosting
- **imbalanced-learn** : Gestion du déséquilibre (SMOTE)
- **SHAP** : Explainability des prédictions

### Backend & API
- **FastAPI** : Framework API REST
- **Pydantic** : Validation des données
- **uvicorn** : Serveur ASGI
- **joblib** : Sérialisation des modèles

### Frontend
- **Streamlit** : Dashboard interactif
- **Plotly** : Visualisations interactives

### Monitoring & DevOps
- **Evidently** : Détection de data drift
- **Docker** : Conteneurisation
- **pytest** : Tests unitaires (si applicable)

---

## 📊 Résultats et performances

### Modèle final
- **Algorithme** : LightGBM Classifier
- **Features** : 804 (après agrégation de 7 tables)
- **Seuil optimal** : 0.370 (optimisé selon ratio coût 10:1)

### Méthodologie
- **Gestion déséquilibre** : SMOTE + ajustement des poids de classe
- **Optimisation** : Minimisation du coût métier (FN coûte 10x plus que FP)
- **Explainability** : SHAP values pour chaque prédiction

### Monitoring
- **Data drift** : Monitoring avec Evidently pour détecter les dérives
- **Rapports** : Génération automatique de rapports HTML

---

## 📝 Notes importantes

### Données exclues du repository

Les fichiers suivants sont exclus du repository Git (voir `.gitignore`) :
- `/data/` : Données sources (plusieurs Go)
- `/artifacts/` : Modèles entraînés (peuvent être régénérés)

Ces exclusions respectent les bonnes pratiques Git (pas de fichiers volumineux).

### Reproductibilité

Le projet est entièrement reproductible :
1. Télécharger les données depuis Kaggle
2. Exécuter les notebooks 01 à 07 dans l'ordre
3. Les artifacts seront générés automatiquement

---

## 🎓 Compétences développées

- Développement d'un modèle de scoring avec gestion du déséquilibre
- Optimisation selon des contraintes métier
- Déploiement d'une API de prédiction
- Création d'un dashboard avec explainability
- Monitoring des performances en production
- Conteneurisation avec Docker

---

## 📧 Contact

**Stéphane BARRE**  
Étudiant - OpenClassrooms Data Scientist  
GitHub : [stephanebarre13-boop](https://github.com/stephanebarre13-boop)

---

## 📄 Licence

Ce projet est réalisé dans le cadre de la formation OpenClassrooms Data Scientist.
