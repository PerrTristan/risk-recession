"""
data/build_fred.py
==================
Pipeline de collecte des données macro-économiques via l'API FRED.

Ce script :
  1. Lit la clé API FRED depuis les variables d'environnement
  2. Télécharge 8 séries temporelles clés (indicateurs de récession)
  3. Stocke tout dans une base DuckDB locale (data/fred_data.duckdb)
  4. Affiche un résumé de ce qui a été collecté

Usage :
    export FRED_API_KEY="votre_cle_ici"
    python data/build_fred.py

    # Ou avec un fichier .env :
    echo "FRED_API_KEY=votre_cle_ici" > .env
    python data/build_fred.py
"""

import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd
import requests
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# ─── Configuration ────────────────────────────────────────────────────────────

# Charger les variables d'environnement depuis .env (si présent)
load_dotenv()

# Clé API FRED (obligatoire)
FRED_API_KEY = os.getenv("FRED_API_KEY")

# URL de base de l'API FRED
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

# Chemin vers la base DuckDB (créée automatiquement si inexistante)
DB_PATH = Path(__file__).parent / "fred_data.duckdb"

# Délai entre les requêtes API pour éviter le rate-limiting (en secondes)
REQUEST_DELAY = 0.5

# Date de début des données historiques
START_DATE = "2000-01-01"

# Console Rich pour un affichage stylisé
console = Console()

# ─── Définition des séries FRED à télécharger ─────────────────────────────────

FRED_SERIES = {
    # Spread courbe des taux : 10 ans - 2 ans
    # Signal classique de récession : inversé = récession dans 6-18 mois
    "T10Y2Y": {
        "name": "Yield Curve Spread (10Y-2Y)",
        "description": "Écart entre le taux 10 ans et 2 ans du Trésor US. "
                       "Négatif = courbe inversée, signal fort de récession.",
        "unit": "%",
        "frequency": "daily",
    },
    # Taux de chômage
    "UNRATE": {
        "name": "Taux de chômage",
        "description": "Taux de chômage mensuel aux États-Unis (données BLS). "
                       "Une remontée de +0.5pp sur 3 mois déclenche la règle de Sahm.",
        "unit": "%",
        "frequency": "monthly",
    },
    # Indice Sahm (règle de récession de Claudia Sahm, 2019)
    "SAHMREALTIME": {
        "name": "Sahm Rule Recession Indicator",
        "description": "Différence entre le taux de chômage moyen sur 3 mois "
                       "et son minimum sur 12 mois. Seuil : 0.5 = récession.",
        "unit": "pp",
        "frequency": "monthly",
    },
    # Ventes au détail (proxy de la demande des ménages)
    "RSAFS": {
        "name": "Ventes au détail (ajustées)",
        "description": "Ventes retail avancées, ajustées saisonnièrement. "
                       "Indicateur clé de la consommation des ménages (~70% du PIB).",
        "unit": "Millions USD",
        "frequency": "monthly",
    },
    # Emploi manufacturier (proxy de l'activité industrielle)
    "MANEMP": {
        "name": "Emploi manufacturier",
        "description": "Nombre de salariés dans l'industrie manufacturière. "
                       "Baisse soutenue = contraction du secteur productif.",
        "unit": "Milliers",
        "frequency": "monthly",
    },
    # Permis de construire (indicateur avancé du marché immobilier)
    "PERMIT": {
        "name": "Permis de construire résidentiels",
        "description": "Nombre de permis de construire délivrés. "
                       "Indicateur avancé : le logement anticipe souvent l'économie.",
        "unit": "Milliers",
        "frequency": "monthly",
    },
    # Confiance des consommateurs (Université du Michigan)
    "UMCSENT": {
        "name": "Confiance des consommateurs (Michigan)",
        "description": "Indice de sentiment des ménages. "
                       "Chute rapide = pessimisme qui précède souvent la récession.",
        "unit": "Index 1966=100",
        "frequency": "monthly",
    },
    # Spread crédit High Yield (stress sur les marchés financiers)
    "BAMLH0A0HYM2": {
        "name": "Spread High Yield (ICE BofA)",
        "description": "Écart de rendement entre obligations HY et Treasuries. "
                       "Hausse = stress financier, fuite vers la qualité.",
        "unit": "%",
        "frequency": "daily",
    },
}


# ─── Fonctions utilitaires ─────────────────────────────────────────────────────

def check_api_key() -> None:
    """Vérifie que la clé API FRED est bien configurée."""
    if not FRED_API_KEY:
        console.print(
            "[bold red]Erreur :[/bold red] Variable d'environnement FRED_API_KEY non définie.\n"
            "Obtenez une clé gratuite sur : https://fred.stlouisfed.org/docs/api/api_key.html\n"
            "Puis : [cyan]export FRED_API_KEY='votre_cle'[/cyan]"
        )
        sys.exit(1)


def fetch_fred_series(series_id: str, start_date: str = START_DATE) -> pd.DataFrame:
    """
    Télécharge une série temporelle depuis l'API FRED.

    Paramètres
    ----------
    series_id : str
        Identifiant FRED de la série (ex: "T10Y2Y")
    start_date : str
        Date de début au format YYYY-MM-DD

    Retourne
    --------
    pd.DataFrame avec colonnes : date (datetime), value (float), series_id (str)

    Lève
    ----
    requests.HTTPError si l'API retourne un code d'erreur
    ValueError si la série est introuvable ou vide
    """
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": start_date,
        "observation_end": datetime.today().strftime("%Y-%m-%d"),
    }

    response = requests.get(FRED_BASE_URL, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()

    if "observations" not in data or not data["observations"]:
        raise ValueError(f"Aucune donnée retournée pour la série {series_id}")

    df = pd.DataFrame(data["observations"])

    # Nettoyage : FRED retourne "." pour les valeurs manquantes
    df = df[df["value"] != "."].copy()
    df["value"] = pd.to_numeric(df["value"])
    df["date"] = pd.to_datetime(df["date"])
    df["series_id"] = series_id

    # Ne garder que les colonnes utiles
    return df[["date", "value", "series_id"]].reset_index(drop=True)


def init_database(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Initialise le schéma de la base DuckDB.

    Tables créées :
    - fred_observations : données brutes (date, value, series_id)
    - fred_series_meta  : métadonnées des séries (nom, description, unité)
    - build_log         : journal des mises à jour
    """
    conn.execute("""
        -- Table principale : une ligne par observation
        CREATE TABLE IF NOT EXISTS fred_observations (
            date        DATE NOT NULL,
            value       DOUBLE NOT NULL,
            series_id   VARCHAR NOT NULL,
            inserted_at TIMESTAMP DEFAULT current_timestamp,
            PRIMARY KEY (date, series_id)
        )
    """)

    conn.execute("""
        -- Métadonnées des séries (nom lisible, description, unité)
        CREATE TABLE IF NOT EXISTS fred_series_meta (
            series_id   VARCHAR PRIMARY KEY,
            name        VARCHAR NOT NULL,
            description VARCHAR,
            unit        VARCHAR,
            frequency   VARCHAR,
            updated_at  TIMESTAMP DEFAULT current_timestamp
        )
    """)

    conn.execute("""
        -- Journal de chaque exécution du script
        CREATE TABLE IF NOT EXISTS build_log (
            run_id      INTEGER PRIMARY KEY,
            run_at      TIMESTAMP DEFAULT current_timestamp,
            series_id   VARCHAR NOT NULL,
            rows_added  INTEGER,
            status      VARCHAR,   -- 'success' | 'error'
            message     VARCHAR
        )
    """)


def upsert_series(
    conn: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
    series_id: str,
    meta: dict,
) -> int:
    """
    Insère ou met à jour les observations d'une série dans DuckDB.

    Utilise INSERT OR REPLACE pour éviter les doublons sur (date, series_id).

    Retourne le nombre de lignes insérées.
    """
    # Insérer les observations
    conn.execute("""
        INSERT OR REPLACE INTO fred_observations (date, value, series_id)
        SELECT date, value, series_id FROM df
    """)

    # Mettre à jour les métadonnées
    conn.execute("""
        INSERT OR REPLACE INTO fred_series_meta (series_id, name, description, unit, frequency)
        VALUES (?, ?, ?, ?, ?)
    """, [series_id, meta["name"], meta["description"], meta["unit"], meta["frequency"]])

    return len(df)


def log_build(
    conn: duckdb.DuckDBPyConnection,
    series_id: str,
    rows: int,
    status: str,
    message: str = "",
) -> None:
    """Enregistre le résultat d'un téléchargement dans build_log."""
    conn.execute("""
        INSERT INTO build_log (series_id, rows_added, status, message)
        VALUES (?, ?, ?, ?)
    """, [series_id, rows, status, message])


def print_summary(conn: duckdb.DuckDBPyConnection) -> None:
    """Affiche un tableau récapitulatif de ce qui est dans la base."""
    result = conn.execute("""
        SELECT
            m.series_id,
            m.name,
            m.unit,
            COUNT(o.value)         AS nb_observations,
            MIN(o.date)::VARCHAR   AS date_debut,
            MAX(o.date)::VARCHAR   AS date_fin,
            ROUND(AVG(o.value), 3) AS moyenne
        FROM fred_series_meta m
        JOIN fred_observations o USING (series_id)
        GROUP BY m.series_id, m.name, m.unit
        ORDER BY m.series_id
    """).fetchdf()

    table = Table(
        title="📊 Données FRED en base",
        show_header=True,
        header_style="bold cyan",
    )
    for col in ["Série", "Nom", "Unité", "N obs.", "Début", "Fin", "Moyenne"]:
        table.add_column(col)

    for _, row in result.iterrows():
        table.add_row(
            row["series_id"],
            row["name"][:35] + "…" if len(row["name"]) > 35 else row["name"],
            row["unit"],
            str(row["nb_observations"]),
            row["date_debut"],
            row["date_fin"],
            str(row["moyenne"]),
        )

    console.print(table)


# ─── Point d'entrée principal ──────────────────────────────────────────────────

def main() -> None:
    console.rule("[bold blue]Recession Risk Dashboard — Build FRED Data[/bold blue]")
    console.print(f"[dim]Démarrage : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]\n")

    # 1. Vérifier la clé API
    check_api_key()

    # 2. Ouvrir (ou créer) la base DuckDB
    console.print(f"[cyan]Base de données :[/cyan] {DB_PATH}")
    conn = duckdb.connect(str(DB_PATH))
    init_database(conn)

    # 3. Télécharger chaque série
    errors = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Téléchargement...", total=len(FRED_SERIES))

        for series_id, meta in FRED_SERIES.items():
            progress.update(task, description=f"[cyan]{series_id}[/cyan] — {meta['name']}")

            try:
                df = fetch_fred_series(series_id)
                rows = upsert_series(conn, df, series_id, meta)
                log_build(conn, series_id, rows, "success")
                console.print(f"  ✅ [green]{series_id}[/green] : {rows} observations")

            except Exception as exc:
                errors.append((series_id, str(exc)))
                log_build(conn, series_id, 0, "error", str(exc))
                console.print(f"  ❌ [red]{series_id}[/red] : {exc}")

            # Pause pour respecter le rate limit de l'API FRED
            time.sleep(REQUEST_DELAY)
            progress.advance(task)

    # 4. Résumé final
    console.print()
    print_summary(conn)

    if errors:
        console.print(f"\n[yellow]⚠ {len(errors)} erreur(s) rencontrée(s).[/yellow]")
        for sid, msg in errors:
            console.print(f"  • {sid} : {msg}")
    else:
        console.print("\n[bold green]✓ Toutes les séries ont été téléchargées avec succès.[/bold green]")

    conn.close()
    console.print(f"\n[dim]Fin : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]")


if __name__ == "__main__":
    main()
