markdown# Artifacts

Ce dossier contient le modèle entraîné pour le système de scoring crédit.

## Fichier inclus

- `meilleur_modele.joblib` (1.7 Mo) : LightGBM Classifier optimisé

## Génération

Ce modèle a été généré par les notebooks suivants :
1. **NB01** - Agrégation des tables
2. **NB02** - Pipeline de préparation
3. **NB03** - Comparaison des modèles
4. **NB04** - Gestion du déséquilibre (SMOTE)
5. **NB05** - Optimisation du seuil métier

## Fichiers exclus

Les fichiers suivants sont exclus du repository car volumineux (peuvent être régénérés) :
- `preprocesseur.joblib`
- `X_train_processed.joblib`
- `X_valid_processed.joblib`
- `y_train.joblib`
- `y_valid.joblib`
- `data_split.joblib`
- Autres fichiers de données intermédiaires

## Régénération

Pour régénérer tous les artifacts :
```bash
# Exécuter les notebooks dans l'ordre
jupyter notebook Barre_Stephane_P7_01_aggregation_tables.ipynb
jupyter notebook Barre_Stephane_P7_02_preparation_pipeline.ipynb
jupyter notebook Barre_Stephane_P7_03_comparaison_modeles.ipynb
jupyter notebook Barre_Stephane_P7_04_desequilibre.ipynb
jupyter notebook Barre_Stephane_P7_05_optimisation_seuil.ipynb
```

## Utilisation avec l'API

L'API FastAPI charge automatiquement ce modèle :
```python
MODEL_PATH = ARTIFACTS_DIR / "meilleur_modele.joblib"
model = joblib.load(MODEL_PATH)
```

## Note

Le modèle nécessite les données preprocessées selon le pipeline défini dans le notebook 02.