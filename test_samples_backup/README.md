# Échantillons de Test - API Credit Scoring

Ce dossier contient des fichiers JSON prêts à envoyer à l'API pour tester les prédictions.

## Fichiers disponibles

1. **client_zeros.json** : Client avec toutes les features à 0
2. **client_low_risk.json** : Client à faible risque (features négatives)
3. **client_high_risk.json** : Client à risque élevé (features positives)
4. **client_mixed.json** : Client avec features variées
5. **batch_clients.json** : Lot de 5 clients pour tests batch

## Utilisation

### PowerShell (Windows)
```powershell
# Charger un échantillon
$body = Get-Content test_samples/client_low_risk.json -Raw

# Envoyer à l'API
$response = Invoke-RestMethod "http://127.0.0.1:8000/predict" -Method Post -ContentType "application/json" -Body $body

# Afficher le résultat
$response | ConvertTo-Json -Depth 10
```

### Bash (Linux/Mac)
```bash
# Tester un client
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d @test_samples/client_low_risk.json
```

### Python
```python
import json
import requests

# Charger l'échantillon
with open('test_samples/client_low_risk.json') as f:
    data = json.load(f)

# Envoyer à l'API
response = requests.post('http://127.0.0.1:8000/predict', json=data)
print(response.json())
```

## Résultats attendus

- **client_zeros** : Devrait donner une probabilité autour de 0.46
- **client_low_risk** : Probabilité < 0.3 (ACCORD recommandé)
- **client_high_risk** : Probabilité > 0.6 (REFUS recommandé)
- **client_mixed** : Probabilité variable selon features
