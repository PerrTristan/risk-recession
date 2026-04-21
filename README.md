# Recession Risk Dashboard

Pipeline complet d'analyse du risque de récession aux États-Unis, basé sur les données FRED (Federal Reserve Economic Data).

## Architecture du projet

```
recession-risk-dashboard/
├── data/
│   └── build_fred.py          # Télécharge et stocke les données FRED dans DuckDB
├── src/
│   └── recession_analysis.py  # Classe principale d'analyse et scoring
├── notebooks/
│   ├── analysis.qmd           # Notebook Quarto : analyse détaillée
│   └── user_guide.qmd         # Notebook Quarto : guide utilisateur
├── pyproject.toml             # Dépendances du projet (uv)
└── README.md                  # Ce fichier
```

## Indicateurs utilisés

| Indicateur       | Série FRED | Description                                     |
|------------------|------------|-------------------------------------------------|
| Courbe des taux  | T10Y2Y     | Spread 10 ans - 2 ans (inversé = signal fort)   |
| Chômage          | UNRATE     | Taux de chômage mensuel                         |
| ISM Manufacture  | MANEMP     | Emploi manufacturier (proxy ISM)                |
| Ventes au détail | RSAFS      | Ventes retail ajustées saisonnièrement          |
| Indice Sahm      | SAHMREALTIME | Règle de Sahm (signal récession précoce)      |
| Permis construire| PERMIT     | Permis de construire résidentiels               |
| Conf. consommateur| UMCSENT   | Indice Michigan de confiance des ménages        |
| Spread crédit    | BAMLH0A0HYM2 | Spread High Yield (stress financier)          |

## Score de risque

Le `RecessionAnalyzer` calcule un **score composite de 0 à 100** :

- **0–30** : Risque faible (expansion économique)
- **30–60** : Risque modéré (ralentissement possible)
- **60–80** : Risque élevé (récession probable)
- **80–100** : Risque très élevé (récession imminente)

## Installation

```bash
# Avec uv (recommandé)
uv sync

# Ou avec pip
pip install -r requirements.txt
```

## Utilisation rapide

```python
# 1. Construire la base de données
python data/build_fred.py

# 2. Lancer l'analyse
python src/recession_analysis.py

# 3. Générer les notebooks
quarto render notebooks/analysis.qmd
quarto render notebooks/user_guide.qmd
```

## Données

Les données sont stockées dans `data/fred_data.duckdb` (base DuckDB locale).
Elles sont récupérées via l'API FRED — une clé API gratuite est nécessaire :
https://fred.stlouisfed.org/docs/api/api_key.html

Mettre la clé dans une variable d'environnement :
```bash
export FRED_API_KEY="votre_cle_ici"
```

## Références

- [FRED API Docs](https://fred.stlouisfed.org/docs/api/fred/)
- [Règle de Sahm (Claudia Sahm, 2019)](https://www.hamiltonproject.org/papers/direct_stimulus_payments_to_individuals)
- [Yield Curve & Recessions (NY Fed)](https://www.newyorkfed.org/research/capital_markets/ycfaq)
