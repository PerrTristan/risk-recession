"""
src/recession_analysis.py
==========================
Classe principale d'analyse du risque de récession.

Ce module fournit :
  - RecessionAnalyzer : charge les données, calcule les signaux, produit un score composite
  - Méthodes de visualisation (Plotly) pour explorer chaque indicateur
  - Un rapport console Rich avec le verdict final

Logique de scoring
------------------
Chaque indicateur est transformé en un signal normalisé entre 0 et 1,
où 1 = signal de récession maximal. Les signaux sont combinés en un score
composite pondéré (0-100). Les poids reflètent la littérature empirique :

  - Courbe des taux (T10Y2Y)    → 25%  (prédicteur le plus robuste, NY Fed)
  - Règle de Sahm               → 25%  (signal précoce sur l'emploi)
  - Taux de chômage (momentum)  → 15%  (niveau + tendance)
  - Spread High Yield           → 15%  (stress financier)
  - Confiance consommateurs     → 10%  (anticipations des ménages)
  - Ventes au détail            → 5%   (demande courante)
  - Emploi manufacturier        → 3%   (secteur productif)
  - Permis de construire        → 2%   (indicateur avancé logement)

Interprétation du score
-----------------------
  0–30   : Risque faible   — expansion économique
  30–60  : Risque modéré   — ralentissement possible
  60–80  : Risque élevé    — récession probable
  80–100 : Risque critique — récession imminente

Usage
-----
    from src.recession_analysis import RecessionAnalyzer

    analyzer = RecessionAnalyzer("data/fred_data.duckdb")
    analyzer.load_data()
    score = analyzer.compute_risk_score()
    print(f"Score de risque : {score:.1f}/100")
    analyzer.print_report()
    analyzer.plot_dashboard()
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# ─── Configuration globale ─────────────────────────────────────────────────────

# Thème Plotly par défaut
pio.templates.default = "plotly_white"

# Console Rich
console = Console()

# ─── Poids des indicateurs dans le score composite ────────────────────────────

INDICATOR_WEIGHTS: dict[str, float] = {
    "yield_curve":       0.25,   # Courbe des taux (10Y-2Y)
    "sahm_rule":         0.25,   # Règle de Sahm
    "unemployment":      0.15,   # Taux de chômage (momentum)
    "hy_spread":         0.15,   # Spread High Yield
    "consumer_conf":     0.10,   # Confiance consommateurs Michigan
    "retail_sales":      0.05,   # Ventes au détail
    "manufacturing":     0.03,   # Emploi manufacturier
    "building_permits":  0.02,   # Permis de construire
}

# Seuil de déclenchement de la Règle de Sahm (Claudia Sahm, 2019)
SAHM_THRESHOLD = 0.5

# Seuil pour courbe des taux inversée (en points de %)
YIELD_CURVE_INVERSION_THRESHOLD = 0.0

# Nombre de mois pour calculer le momentum du chômage
UNEMPLOYMENT_MOMENTUM_MONTHS = 6

# Fenêtre glissante pour les calculs de z-score (en observations)
ZSCORE_WINDOW = 120  # ~10 ans


# ─── Dataclasses pour structurer les résultats ────────────────────────────────

@dataclass
class IndicatorSignal:
    """Résultat normalisé d'un indicateur individuel."""
    name: str          # Nom lisible de l'indicateur
    series_id: str     # Identifiant FRED
    raw_value: float   # Dernière valeur brute
    signal: float      # Signal normalisé entre 0 (pas de risque) et 1 (risque max)
    weight: float      # Poids dans le score composite
    contribution: float  # Contribution au score final (signal * weight * 100)
    unit: str          # Unité de la série
    interpretation: str  # Texte explicatif court


@dataclass
class RiskReport:
    """Rapport complet du score de risque de récession."""
    score: float                      # Score composite 0-100
    risk_level: str                   # 'Faible' | 'Modéré' | 'Élevé' | 'Critique'
    risk_color: str                   # Couleur associée (pour affichage)
    signals: list[IndicatorSignal]    # Détail par indicateur
    computation_date: str             # Date du calcul
    data_as_of: str                   # Date des dernières données disponibles


# ─── Classe principale ─────────────────────────────────────────────────────────

class RecessionAnalyzer:
    """
    Analyseur de risque de récession basé sur 8 indicateurs macro-économiques FRED.

    Paramètres
    ----------
    db_path : str | Path
        Chemin vers la base DuckDB créée par data/build_fred.py.
        Par défaut : "data/fred_data.duckdb" (relatif au répertoire courant).

    Attributs
    ---------
    data : dict[str, pd.DataFrame]
        Données brutes par série FRED (chargées par load_data())
    report : RiskReport | None
        Dernier rapport calculé (disponible après compute_risk_score())

    Exemple
    -------
    >>> analyzer = RecessionAnalyzer()
    >>> analyzer.load_data()
    >>> score = analyzer.compute_risk_score()
    >>> analyzer.print_report()
    >>> fig = analyzer.plot_dashboard()
    >>> fig.show()
    """

    def __init__(self, db_path: str | Path = "data/fred_data.duckdb") -> None:
        self.db_path = Path(db_path)
        self.data: dict[str, pd.DataFrame] = {}
        self.report: RiskReport | None = None
        self._conn: duckdb.DuckDBPyConnection | None = None

    # ── Connexion et chargement ────────────────────────────────────────────────

    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        """Retourne une connexion DuckDB (lecture seule)."""
        if self._conn is None:
            if not self.db_path.exists():
                raise FileNotFoundError(
                    f"Base de données introuvable : {self.db_path}\n"
                    "Exécutez d'abord : python data/build_fred.py"
                )
            self._conn = duckdb.connect(str(self.db_path), read_only=True)
        return self._conn

    def load_data(self) -> None:
        """
        Charge toutes les séries FRED depuis DuckDB dans self.data.

        Après cette méthode, self.data[series_id] est un DataFrame
        avec colonnes : date (index datetime), value (float).
        """
        conn = self._get_connection()

        series_ids = conn.execute(
            "SELECT DISTINCT series_id FROM fred_observations"
        ).fetchdf()["series_id"].tolist()

        for sid in series_ids:
            df = conn.execute(
                "SELECT date, value FROM fred_observations WHERE series_id = ? ORDER BY date",
                [sid],
            ).fetchdf()
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            self.data[sid] = df

        console.print(
            f"[green]✓[/green] {len(self.data)} séries chargées depuis [cyan]{self.db_path}[/cyan]"
        )

    def _require_data(self) -> None:
        """Vérifie que load_data() a été appelé."""
        if not self.data:
            raise RuntimeError("Appelez d'abord load_data() avant de calculer les signaux.")

    # ── Calcul des signaux individuels ────────────────────────────────────────

    def _signal_yield_curve(self) -> IndicatorSignal:
        """
        Signal basé sur la courbe des taux (T10Y2Y).

        Logique :
          - Courbe fortement inversée (< -1%) → signal = 1.0
          - Courbe plate (0%) → signal = 0.5
          - Courbe normale (> 1.5%) → signal = 0.0
        """
        series = self.data.get("T10Y2Y")
        if series is None or series.empty:
            return self._empty_signal("yield_curve", "T10Y2Y", "Courbe des taux", "%")

        # Moyenne sur les 20 derniers jours ouvrés (stabilise le signal)
        recent_value = float(series["value"].iloc[-20:].mean())
        last_value = float(series["value"].iloc[-1])

        # Normalisation : -1.5% → 1.0 ; +1.5% → 0.0
        signal = float(max(0.0, min(1.0, (-recent_value + 1.5) / 3.0)))

        interpretation = (
            f"Courbe inversée ({last_value:+.2f}%) — signal fort de récession"
            if last_value < 0
            else f"Courbe normale ({last_value:+.2f}%) — pas de signal"
        )

        return IndicatorSignal(
            name="Courbe des taux (10Y-2Y)",
            series_id="T10Y2Y",
            raw_value=last_value,
            signal=signal,
            weight=INDICATOR_WEIGHTS["yield_curve"],
            contribution=signal * INDICATOR_WEIGHTS["yield_curve"] * 100,
            unit="%",
            interpretation=interpretation,
        )

    def _signal_sahm_rule(self) -> IndicatorSignal:
        """
        Signal basé sur la Règle de Sahm (SAHMREALTIME).

        Claudia Sahm (2019) : si la moyenne mobile 3 mois du chômage
        dépasse de 0.5pp son minimum des 12 derniers mois → récession.

        Logique :
          - Valeur >= 0.5 → signal = 1.0 (règle déclenchée)
          - Valeur entre 0 et 0.5 → signal proportionnel
        """
        series = self.data.get("SAHMREALTIME")
        if series is None or series.empty:
            return self._empty_signal("sahm_rule", "SAHMREALTIME", "Règle de Sahm", "pp")

        last_value = float(series["value"].iloc[-1])

        # Normalisation : 0 → 0.0 ; 0.5+ → 1.0
        signal = float(min(1.0, max(0.0, last_value / SAHM_THRESHOLD)))

        interpretation = (
            f"Règle de Sahm DÉCLENCHÉE ({last_value:.2f} pp ≥ {SAHM_THRESHOLD}) — récession probable"
            if last_value >= SAHM_THRESHOLD
            else f"Règle de Sahm inactive ({last_value:.2f} pp < {SAHM_THRESHOLD})"
        )

        return IndicatorSignal(
            name="Règle de Sahm",
            series_id="SAHMREALTIME",
            raw_value=last_value,
            signal=signal,
            weight=INDICATOR_WEIGHTS["sahm_rule"],
            contribution=signal * INDICATOR_WEIGHTS["sahm_rule"] * 100,
            unit="pp",
            interpretation=interpretation,
        )

    def _signal_unemployment(self) -> IndicatorSignal:
        """
        Signal basé sur le taux de chômage (UNRATE) — niveau + momentum.

        Combine deux sous-signaux :
          - Niveau absolu : chômage > 6% = signal élevé
          - Momentum : variation sur 6 mois (accélération = mauvais signe)
        """
        series = self.data.get("UNRATE")
        if series is None or series.empty:
            return self._empty_signal("unemployment", "UNRATE", "Taux de chômage", "%")

        last_value = float(series["value"].iloc[-1])

        # Sous-signal 1 : niveau (4% = 0.0 ; 8% = 1.0)
        level_signal = float(max(0.0, min(1.0, (last_value - 4.0) / 4.0)))

        # Sous-signal 2 : momentum sur UNEMPLOYMENT_MOMENTUM_MONTHS mois
        if len(series) >= UNEMPLOYMENT_MOMENTUM_MONTHS:
            prev_value = float(series["value"].iloc[-UNEMPLOYMENT_MOMENTUM_MONTHS])
            delta = last_value - prev_value
            # Hausse de 2pp = signal = 1.0 ; baisse = signal = 0.0
            momentum_signal = float(max(0.0, min(1.0, delta / 2.0)))
        else:
            momentum_signal = 0.0

        # Moyenne pondérée des deux sous-signaux
        signal = 0.6 * level_signal + 0.4 * momentum_signal

        interpretation = (
            f"Chômage à {last_value:.1f}% "
            f"({'↑' if momentum_signal > 0.2 else '→' if momentum_signal > 0.05 else '↓'} "
            f"sur {UNEMPLOYMENT_MOMENTUM_MONTHS} mois)"
        )

        return IndicatorSignal(
            name="Taux de chômage",
            series_id="UNRATE",
            raw_value=last_value,
            signal=signal,
            weight=INDICATOR_WEIGHTS["unemployment"],
            contribution=signal * INDICATOR_WEIGHTS["unemployment"] * 100,
            unit="%",
            interpretation=interpretation,
        )

    def _signal_hy_spread(self) -> IndicatorSignal:
        """
        Signal basé sur le spread High Yield (BAMLH0A0HYM2).

        Un spread élevé reflète le stress sur les marchés du crédit.
        Seuils historiques :
          - < 4%    : faible stress (expansion)
          - 4-8%    : stress modéré
          - > 10%   : crise crédit (ex: 2008, 2020)
        """
        series = self.data.get("BAMLH0A0HYM2")
        if series is None or series.empty:
            return self._empty_signal("hy_spread", "BAMLH0A0HYM2", "Spread High Yield", "%")

        # Moyenne sur 10 jours pour lisser la volatilité quotidienne
        recent_avg = float(series["value"].iloc[-10:].mean())
        last_value = float(series["value"].iloc[-1])

        # Normalisation : 3% → 0.0 ; 10% → 1.0
        signal = float(max(0.0, min(1.0, (recent_avg - 3.0) / 7.0)))

        interpretation = (
            f"Spread HY à {last_value:.1f}% — "
            + ("stress financier élevé" if last_value > 7 else
               "stress modéré" if last_value > 5 else
               "marchés calmes")
        )

        return IndicatorSignal(
            name="Spread High Yield",
            series_id="BAMLH0A0HYM2",
            raw_value=last_value,
            signal=signal,
            weight=INDICATOR_WEIGHTS["hy_spread"],
            contribution=signal * INDICATOR_WEIGHTS["hy_spread"] * 100,
            unit="%",
            interpretation=interpretation,
        )

    def _signal_consumer_confidence(self) -> IndicatorSignal:
        """
        Signal basé sur la confiance des consommateurs (UMCSENT).

        Compare le niveau actuel à la moyenne historique normalisée.
        Chute rapide > 15 points sur 3 mois = signal fort.
        """
        series = self.data.get("UMCSENT")
        if series is None or series.empty:
            return self._empty_signal("consumer_conf", "UMCSENT", "Confiance consommateurs", "Index")

        last_value = float(series["value"].iloc[-1])

        # Médiane historique pour normaliser
        historical_median = float(series["value"].median())
        historical_std = float(series["value"].std())

        # Z-score inversé (faible confiance = signal élevé)
        z_score = (historical_median - last_value) / historical_std if historical_std > 0 else 0
        signal = float(max(0.0, min(1.0, z_score / 2.0)))

        # Momentum sur 3 mois
        if len(series) >= 3:
            delta_3m = last_value - float(series["value"].iloc[-3])
            trend_str = f", variation 3 mois : {delta_3m:+.1f} pts"
        else:
            trend_str = ""

        interpretation = f"Confiance à {last_value:.1f} (médiane hist. : {historical_median:.1f}){trend_str}"

        return IndicatorSignal(
            name="Confiance consommateurs",
            series_id="UMCSENT",
            raw_value=last_value,
            signal=signal,
            weight=INDICATOR_WEIGHTS["consumer_conf"],
            contribution=signal * INDICATOR_WEIGHTS["consumer_conf"] * 100,
            unit="Index",
            interpretation=interpretation,
        )

    def _signal_retail_sales(self) -> IndicatorSignal:
        """
        Signal basé sur les ventes au détail (RSAFS).

        Utilise la variation annuelle (YoY) :
          - Croissance > 3% → faible risque
          - Croissance négative → signal élevé
        """
        series = self.data.get("RSAFS")
        if series is None or series.empty:
            return self._empty_signal("retail_sales", "RSAFS", "Ventes au détail", "M$")

        last_value = float(series["value"].iloc[-1])

        # Variation annualisée sur 12 mois
        if len(series) >= 12:
            prev_year = float(series["value"].iloc[-12])
            yoy = (last_value - prev_year) / prev_year * 100 if prev_year != 0 else 0.0
        else:
            yoy = 0.0

        # Normalisation : > 3% YoY → 0.0 ; < -3% YoY → 1.0
        signal = float(max(0.0, min(1.0, (-yoy + 3.0) / 6.0)))

        interpretation = (
            f"Ventes retail : {last_value:,.0f} M$ "
            f"(variation annuelle : {yoy:+.1f}%)"
        )

        return IndicatorSignal(
            name="Ventes au détail",
            series_id="RSAFS",
            raw_value=last_value,
            signal=signal,
            weight=INDICATOR_WEIGHTS["retail_sales"],
            contribution=signal * INDICATOR_WEIGHTS["retail_sales"] * 100,
            unit="M$",
            interpretation=interpretation,
        )

    def _signal_manufacturing(self) -> IndicatorSignal:
        """
        Signal basé sur l'emploi manufacturier (MANEMP).

        Variation annuelle négative = signal de contraction industrielle.
        """
        series = self.data.get("MANEMP")
        if series is None or series.empty:
            return self._empty_signal("manufacturing", "MANEMP", "Emploi manufacturier", "K")

        last_value = float(series["value"].iloc[-1])

        if len(series) >= 12:
            prev_year = float(series["value"].iloc[-12])
            yoy = (last_value - prev_year) / prev_year * 100 if prev_year != 0 else 0.0
        else:
            yoy = 0.0

        # Normalisation : > 1% YoY → 0.0 ; < -3% YoY → 1.0
        signal = float(max(0.0, min(1.0, (-yoy + 1.0) / 4.0)))

        interpretation = f"Emploi manufact. : {last_value:,.0f} K (YoY : {yoy:+.1f}%)"

        return IndicatorSignal(
            name="Emploi manufacturier",
            series_id="MANEMP",
            raw_value=last_value,
            signal=signal,
            weight=INDICATOR_WEIGHTS["manufacturing"],
            contribution=signal * INDICATOR_WEIGHTS["manufacturing"] * 100,
            unit="K",
            interpretation=interpretation,
        )

    def _signal_building_permits(self) -> IndicatorSignal:
        """
        Signal basé sur les permis de construire (PERMIT).

        Indicateur avancé : chute des permis précède souvent la récession
        de 6 à 12 mois.
        """
        series = self.data.get("PERMIT")
        if series is None or series.empty:
            return self._empty_signal("building_permits", "PERMIT", "Permis de construire", "K")

        last_value = float(series["value"].iloc[-1])

        if len(series) >= 12:
            prev_year = float(series["value"].iloc[-12])
            yoy = (last_value - prev_year) / prev_year * 100 if prev_year != 0 else 0.0
        else:
            yoy = 0.0

        # Normalisation : > 5% YoY → 0.0 ; < -20% YoY → 1.0
        signal = float(max(0.0, min(1.0, (-yoy + 5.0) / 25.0)))

        interpretation = f"Permis : {last_value:,.0f} K (YoY : {yoy:+.1f}%)"

        return IndicatorSignal(
            name="Permis de construire",
            series_id="PERMIT",
            raw_value=last_value,
            signal=signal,
            weight=INDICATOR_WEIGHTS["building_permits"],
            contribution=signal * INDICATOR_WEIGHTS["building_permits"] * 100,
            unit="K",
            interpretation=interpretation,
        )

    @staticmethod
    def _empty_signal(key: str, series_id: str, name: str, unit: str) -> IndicatorSignal:
        """Retourne un signal neutre quand une série est manquante."""
        return IndicatorSignal(
            name=name,
            series_id=series_id,
            raw_value=float("nan"),
            signal=0.0,
            weight=INDICATOR_WEIGHTS.get(key, 0.0),
            contribution=0.0,
            unit=unit,
            interpretation="Données non disponibles",
        )

    # ── Score composite ────────────────────────────────────────────────────────

    def compute_risk_score(self) -> float:
        """
        Calcule le score composite de risque de récession (0–100).

        Les 8 signaux individuels sont pondérés selon INDICATOR_WEIGHTS,
        puis additionnés pour former le score final.

        Retourne
        --------
        float : score entre 0 (aucun risque) et 100 (récession certaine)

        Effet de bord
        -------------
        Stocke le rapport complet dans self.report.
        """
        self._require_data()

        # Calculer les 8 signaux
        signals = [
            self._signal_yield_curve(),
            self._signal_sahm_rule(),
            self._signal_unemployment(),
            self._signal_hy_spread(),
            self._signal_consumer_confidence(),
            self._signal_retail_sales(),
            self._signal_manufacturing(),
            self._signal_building_permits(),
        ]

        # Score composite = somme pondérée * 100
        score = sum(s.signal * s.weight for s in signals) * 100

        # Classification du risque
        if score < 30:
            risk_level, risk_color = "Faible", "green"
        elif score < 60:
            risk_level, risk_color = "Modéré", "yellow"
        elif score < 80:
            risk_level, risk_color = "Élevé", "orange"
        else:
            risk_level, risk_color = "Critique", "red"

        # Déterminer la date des données les plus récentes
        latest_dates = [
            df.index[-1].strftime("%Y-%m-%d")
            for df in self.data.values()
            if not df.empty
        ]
        data_as_of = max(latest_dates) if latest_dates else "N/A"

        # Construire et stocker le rapport
        from datetime import datetime
        self.report = RiskReport(
            score=score,
            risk_level=risk_level,
            risk_color=risk_color,
            signals=signals,
            computation_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            data_as_of=data_as_of,
        )

        return score

    # ── Affichage console ──────────────────────────────────────────────────────

    def print_report(self) -> None:
        """
        Affiche le rapport complet dans le terminal (via Rich).

        Prérequis : compute_risk_score() doit avoir été appelé.
        """
        if self.report is None:
            raise RuntimeError("Appelez d'abord compute_risk_score().")

        r = self.report

        # Panneau principal avec le score
        score_bar = self._text_score_bar(r.score)
        panel_content = (
            f"Score de risque : [bold]{r.score:.1f} / 100[/bold]\n"
            f"Niveau          : [bold {r.risk_color}]{r.risk_level}[/bold {r.risk_color}]\n"
            f"Barre           : {score_bar}\n"
            f"Données au      : {r.data_as_of}\n"
            f"Calculé le      : {r.computation_date}"
        )
        console.print(Panel(panel_content, title="📊 Recession Risk Dashboard", border_style="blue"))

        # Tableau des signaux détaillés
        table = Table(show_header=True, header_style="bold cyan", title="Détail des indicateurs")
        table.add_column("Indicateur", style="white", min_width=25)
        table.add_column("Valeur", justify="right")
        table.add_column("Signal", justify="right")
        table.add_column("Poids", justify="right")
        table.add_column("Contrib.", justify="right")
        table.add_column("Interprétation", style="dim")

        for sig in sorted(r.signals, key=lambda s: s.contribution, reverse=True):
            # Couleur du signal : vert < 0.33 ; jaune < 0.66 ; rouge >=0.66
            if sig.signal < 0.33:
                signal_style = "green"
            elif sig.signal < 0.66:
                signal_style = "yellow"
            else:
                signal_style = "red"

            table.add_row(
                sig.name,
                f"{sig.raw_value:.2f} {sig.unit}" if not pd.isna(sig.raw_value) else "N/A",
                f"[{signal_style}]{sig.signal:.2f}[/{signal_style}]",
                f"{sig.weight:.0%}",
                f"{sig.contribution:.1f} pts",
                sig.interpretation[:55] + "…" if len(sig.interpretation) > 55 else sig.interpretation,
            )

        console.print(table)

    @staticmethod
    def _text_score_bar(score: float, width: int = 30) -> str:
        """Génère une barre de progression texte pour le score."""
        filled = int(score / 100 * width)
        bar = "█" * filled + "░" * (width - filled)
        if score < 30:
            return f"[green]{bar}[/green]"
        elif score < 60:
            return f"[yellow]{bar}[/yellow]"
        elif score < 80:
            return f"[orange1]{bar}[/orange1]"
        else:
            return f"[red]{bar}[/red]"

    # ── Visualisations Plotly ──────────────────────────────────────────────────

    def plot_dashboard(self, show: bool = True) -> go.Figure:
        """
        Génère un dashboard Plotly interactif avec tous les indicateurs.

        Contenu :
          - Ligne 1 : Jauge du score + Courbe des taux
          - Ligne 2 : Règle de Sahm + Taux de chômage
          - Ligne 3 : Spread High Yield + Confiance consommateurs
          - Ligne 4 : Contributions des indicateurs (bar chart)

        Paramètres
        ----------
        show : bool
            Si True, affiche la figure dans le navigateur.

        Retourne
        --------
        go.Figure : objet Plotly
        """
        if self.report is None:
            raise RuntimeError("Appelez d'abord compute_risk_score().")

        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=(
                "Score de Risque Global",
                "Courbe des Taux (T10Y2Y)",
                "Règle de Sahm",
                "Taux de Chômage (UNRATE)",
                "Spread High Yield",
                "Confiance Consommateurs",
                "Contributions par Indicateur",
                "",
            ),
            specs=[
                [{"type": "indicator"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "bar"},      {"type": "scatter"}],
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.08,
        )

        score = self.report.score

        # 1. Jauge du score global ────────────────────────────────────────────
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=score,
                title={"text": f"Risque : {self.report.risk_level}", "font": {"size": 16}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [0, 30],   "color": "#2ecc71"},
                        {"range": [30, 60],  "color": "#f39c12"},
                        {"range": [60, 80],  "color": "#e67e22"},
                        {"range": [80, 100], "color": "#e74c3c"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 4},
                        "thickness": 0.75,
                        "value": score,
                    },
                },
                number={"suffix": " / 100", "font": {"size": 24}},
            ),
            row=1, col=1,
        )

        # 2. Courbe des taux ──────────────────────────────────────────────────
        if "T10Y2Y" in self.data:
            self._add_time_series(
                fig, "T10Y2Y",
                color="#3498db",
                row=1, col=2,
                add_zero_line=True,
                fill_negative=True,
            )

        # 3. Règle de Sahm ────────────────────────────────────────────────────
        if "SAHMREALTIME" in self.data:
            self._add_time_series(
                fig, "SAHMREALTIME",
                color="#e74c3c",
                row=2, col=1,
                add_threshold=SAHM_THRESHOLD,
            )

        # 4. Taux de chômage ──────────────────────────────────────────────────
        if "UNRATE" in self.data:
            self._add_time_series(fig, "UNRATE", color="#9b59b6", row=2, col=2)

        # 5. Spread High Yield ────────────────────────────────────────────────
        if "BAMLH0A0HYM2" in self.data:
            self._add_time_series(fig, "BAMLH0A0HYM2", color="#e67e22", row=3, col=1)

        # 6. Confiance consommateurs ──────────────────────────────────────────
        if "UMCSENT" in self.data:
            self._add_time_series(fig, "UMCSENT", color="#1abc9c", row=3, col=2)

        # 7. Contributions par indicateur (bar chart horizontal) ──────────────
        sorted_signals = sorted(self.report.signals, key=lambda s: s.contribution)
        fig.add_trace(
            go.Bar(
                x=[s.contribution for s in sorted_signals],
                y=[s.name for s in sorted_signals],
                orientation="h",
                marker_color=[
                    "#2ecc71" if s.signal < 0.33 else
                    "#f39c12" if s.signal < 0.66 else
                    "#e74c3c"
                    for s in sorted_signals
                ],
                text=[f"{s.contribution:.1f} pts" for s in sorted_signals],
                textposition="outside",
            ),
            row=4, col=1,
        )

        # Mise en forme globale
        fig.update_layout(
            title={
                "text": f"Recession Risk Dashboard — Score : {score:.1f}/100 ({self.report.risk_level})",
                "x": 0.5,
                "font": {"size": 20},
            },
            height=1200,
            showlegend=False,
            template="plotly_white",
            annotations=[
                go.layout.Annotation(
                    text=f"Données au {self.report.data_as_of} | Sources : FRED (Federal Reserve)",
                    xref="paper", yref="paper",
                    x=0.5, y=-0.04,
                    showarrow=False,
                    font={"size": 10, "color": "gray"},
                )
            ],
        )

        if show:
            fig.show()

        return fig

    def _add_time_series(
        self,
        fig: go.Figure,
        series_id: str,
        color: str,
        row: int,
        col: int,
        add_zero_line: bool = False,
        add_threshold: float | None = None,
        fill_negative: bool = False,
    ) -> None:
        """
        Ajoute une série temporelle à un subplot Plotly.

        Paramètres
        ----------
        fig : go.Figure
        series_id : str
        color : str           Couleur de la ligne (hex)
        row, col : int        Position dans la grille de subplots
        add_zero_line : bool  Ajouter une ligne horizontale à y=0
        add_threshold : float Ajouter une ligne horizontale de seuil
        fill_negative : bool  Remplir la zone sous zéro en rouge
        """
        df = self.data[series_id].last("10Y")  # Dernières 10 années
        x = df.index
        y = df["value"]

        # Ligne principale
        fig.add_trace(
            go.Scatter(
                x=x, y=y,
                mode="lines",
                line={"color": color, "width": 1.5},
                name=series_id,
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>",
            ),
            row=row, col=col,
        )

        # Zone négative (pour courbe des taux)
        if fill_negative:
            y_neg = y.clip(upper=0)
            fig.add_trace(
                go.Scatter(
                    x=x, y=y_neg,
                    fill="tozeroy",
                    fillcolor="rgba(231, 76, 60, 0.2)",
                    line={"color": "rgba(0,0,0,0)"},
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row, col=col,
            )

        # Ligne à zéro
        if add_zero_line:
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=row, col=col)

        # Ligne de seuil (ex: Sahm 0.5)
        if add_threshold is not None:
            fig.add_hline(
                y=add_threshold,
                line_dash="dot",
                line_color="red",
                annotation_text=f"Seuil {add_threshold}",
                row=row, col=col,
            )

    def save_dashboard(self, output_path: str = "recession_dashboard.html") -> str:
        """
        Sauvegarde le dashboard Plotly en fichier HTML autonome.

        Paramètres
        ----------
        output_path : str   Chemin du fichier HTML de sortie

        Retourne
        --------
        str : chemin absolu du fichier créé
        """
        if self.report is None:
            raise RuntimeError("Appelez d'abord compute_risk_score().")

        fig = self.plot_dashboard(show=False)
        fig.write_html(output_path, include_plotlyjs="cdn")

        abs_path = str(Path(output_path).resolve())
        console.print(f"[green]✓[/green] Dashboard sauvegardé : [cyan]{abs_path}[/cyan]")
        return abs_path


# ─── Point d'entrée ───────────────────────────────────────────────────────────

def main() -> None:
    """Lancer l'analyse complète et afficher le rapport."""
    console.rule("[bold blue]Recession Risk Analyzer[/bold blue]")

    analyzer = RecessionAnalyzer()

    # Charger les données depuis DuckDB
    analyzer.load_data()

    # Calculer le score de risque
    score = analyzer.compute_risk_score()

    # Afficher le rapport dans le terminal
    analyzer.print_report()

    # Générer et ouvrir le dashboard interactif
    console.print("\n[cyan]Génération du dashboard Plotly...[/cyan]")
    analyzer.save_dashboard("recession_dashboard.html")


if __name__ == "__main__":
    main()
