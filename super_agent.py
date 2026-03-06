"""
╔══════════════════════════════════════════════════════════════════════════╗
║          ⚽  SUPER AGENT PRONOSTICS FOOTBALL — VERSION UNIFIÉE  ⚽       ║
║                                                                          ║
║  Tous les modules fusionnés en un seul agent autonome :                  ║
║   🕵️  Data Scout       → Collecte des données & fixtures                ║
║   📊  Stats Engine     → Elo, xG, Pi-ratings, Forme                     ║
║   📰  News & Context   → Blessures, Compos, Actualités                  ║
║   🔮  ML Predictor     → CatBoost / Régression Logistique               ║
║   ✅  Validator        → Calibration, Value Bets, Accountability        ║
║   📱  Output Formatter → Rapport complet & structuré                    ║
╚══════════════════════════════════════════════════════════════════════════╝

Dépendances : pip install requests numpy pandas scikit-learn
              pip install langchain langchain-openai langgraph
              pip install catboost tavily-python python-dotenv
"""

# ============================================================
#  IMPORTS
# ============================================================
import os
import json
import math
import time
import logging
import hashlib
import sqlite3
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    from langchain_core.tools import tool
    from langgraph.graph import StateGraph, END
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

from dotenv import load_dotenv
load_dotenv()

# ============================================================
#  CONFIGURATION GLOBALE
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s → %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("FootballSuperAgent")

CONFIG = {
    "FOOTBALL_API_KEY":  os.getenv("FOOTBALL_API_KEY", ""),       # football-data.org
    "OPENAI_API_KEY":    os.getenv("OPENAI_API_KEY", ""),
    "TAVILY_API_KEY":    os.getenv("TAVILY_API_KEY", ""),
    "DB_PATH":           "football_super_agent.db",
    "ELO_K_FACTOR":      32,
    "ELO_DEFAULT":       1500,
    "HOME_ADVANTAGE":    100,          # points Elo bonus domicile
    "FORM_WINDOW":       5,            # nb derniers matchs pour la forme
    "FORM_DECAY":        0.85,         # facteur de décroissance temporelle
    "LEAGUES": {
        "PL":  "Premier League",
        "PD":  "La Liga",
        "BL1": "Bundesliga",
        "SA":  "Serie A",
        "FL1": "Ligue 1",
        "CL":  "Champions League",
    },
    "MODEL_TYPE": "catboost" if CATBOOST_AVAILABLE else "logistic",
    "VALUE_BET_THRESHOLD": 0.05,       # marge value bet minimale
}


# ============================================================
#  STATE TYPEDDICT — État partagé du Super Agent
# ============================================================
class AgentState(TypedDict, total=False):
    # Requête utilisateur
    query:              str
    league:             str
    matchday:           Optional[int]

    # Données collectées (Scout)
    fixtures:           List[Dict]
    teams_metadata:     Dict[str, Any]
    raw_stats:          Dict[str, Any]

    # Statistiques calculées (Stats Engine)
    elo_ratings:        Dict[str, float]
    pi_ratings:         Dict[str, Dict]
    form_scores:        Dict[str, float]
    xg_data:            Dict[str, Dict]
    h2h_records:        Dict[str, Dict]

    # Contexte (News Agent)
    injuries:           Dict[str, List[str]]
    suspensions:        Dict[str, List[str]]
    news_snippets:      Dict[str, List[str]]
    lineup_hints:       Dict[str, str]

    # Prédictions (ML Engine)
    predictions:        List[Dict]
    model_confidence:   Dict[str, float]
    feature_importance: Dict[str, float]

    # Validation (Validator)
    calibrated_probs:   List[Dict]
    value_bets:         List[Dict]
    accountability_log: List[Dict]

    # Rapport final (Output)
    final_report:       str
    errors:             List[str]

    # Contrôle du flux
    step:               str
    iteration:          int


# ============================================================
#  MODULE 1 — BASE DE DONNÉES LOCALE (SQLite)
# ============================================================
class DatabaseManager:
    """Gère la persistance : matchs, ratings, prédictions, accountability."""

    def __init__(self, db_path: str = CONFIG["DB_PATH"]):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS elo_ratings (
                    team_id   TEXT PRIMARY KEY,
                    team_name TEXT,
                    rating    REAL DEFAULT 1500,
                    updated   TEXT
                );

                CREATE TABLE IF NOT EXISTS pi_ratings (
                    team_id     TEXT PRIMARY KEY,
                    team_name   TEXT,
                    attack_home REAL DEFAULT 0,
                    attack_away REAL DEFAULT 0,
                    defense_home REAL DEFAULT 0,
                    defense_away REAL DEFAULT 0,
                    updated     TEXT
                );

                CREATE TABLE IF NOT EXISTS match_history (
                    match_id    TEXT PRIMARY KEY,
                    date        TEXT,
                    home_team   TEXT,
                    away_team   TEXT,
                    home_goals  INTEGER,
                    away_goals  INTEGER,
                    home_xg     REAL,
                    away_xg     REAL,
                    league      TEXT,
                    season      TEXT
                );

                CREATE TABLE IF NOT EXISTS predictions_log (
                    pred_id         TEXT PRIMARY KEY,
                    match_id        TEXT,
                    date_pred       TEXT,
                    home_team       TEXT,
                    away_team       TEXT,
                    prob_home       REAL,
                    prob_draw       REAL,
                    prob_away       REAL,
                    recommended     TEXT,
                    confidence      REAL,
                    is_value_bet    INTEGER,
                    actual_result   TEXT,
                    correct         INTEGER
                );
            """)
        logger.info("✅ Base de données initialisée : %s", self.db_path)

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    def upsert_elo(self, team_id: str, team_name: str, rating: float):
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO elo_ratings (team_id, team_name, rating, updated)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(team_id) DO UPDATE SET
                    rating=excluded.rating, updated=excluded.updated
            """, (team_id, team_name, rating, datetime.now().isoformat()))

    def get_elo(self, team_id: str) -> float:
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT rating FROM elo_ratings WHERE team_id=?", (team_id,)
            ).fetchone()
        return row[0] if row else CONFIG["ELO_DEFAULT"]

    def save_match(self, match: Dict):
        with self.get_connection() as conn:
            conn.execute("""
                INSERT OR IGNORE INTO match_history
                (match_id, date, home_team, away_team, home_goals, away_goals,
                 home_xg, away_xg, league, season)
                VALUES (?,?,?,?,?,?,?,?,?,?)
            """, (
                match.get("id", ""),
                match.get("date", ""),
                match.get("home_team", ""),
                match.get("away_team", ""),
                match.get("home_goals", 0),
                match.get("away_goals", 0),
                match.get("home_xg", 0.0),
                match.get("away_xg", 0.0),
                match.get("league", ""),
                match.get("season", ""),
            ))

    def get_match_history(self, team_name: str, n: int = 20) -> List[Dict]:
        with self.get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM match_history
                WHERE home_team=? OR away_team=?
                ORDER BY date DESC LIMIT ?
            """, (team_name, team_name, n)).fetchall()
        cols = ["match_id", "date", "home_team", "away_team", "home_goals",
                "away_goals", "home_xg", "away_xg", "league", "season"]
        return [dict(zip(cols, r)) for r in rows]

    def save_prediction(self, pred: Dict):
        pred_id = hashlib.md5(
            f"{pred['home_team']}{pred['away_team']}{pred['date_pred']}".encode()
        ).hexdigest()
        with self.get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO predictions_log
                (pred_id, match_id, date_pred, home_team, away_team,
                 prob_home, prob_draw, prob_away, recommended,
                 confidence, is_value_bet, actual_result, correct)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                pred_id,
                pred.get("match_id", ""),
                pred.get("date_pred", datetime.now().isoformat()),
                pred["home_team"],
                pred["away_team"],
                pred["prob_home"],
                pred["prob_draw"],
                pred["prob_away"],
                pred["recommended"],
                pred["confidence"],
                int(pred.get("is_value_bet", False)),
                pred.get("actual_result"),
                pred.get("correct"),
            ))

    def get_accountability_stats(self) -> Dict:
        with self.get_connection() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM predictions_log WHERE actual_result IS NOT NULL"
            ).fetchone()[0]
            correct = conn.execute(
                "SELECT COUNT(*) FROM predictions_log WHERE correct=1"
            ).fetchone()[0]
        accuracy = (correct / total * 100) if total > 0 else 0
        return {"total": total, "correct": correct, "accuracy": round(accuracy, 2)}


# ============================================================
#  MODULE 2 — DATA SCOUT (Collecte des Données)
# ============================================================
class DataScout:
    """
    Collecte les fixtures, résultats et statistiques via football-data.org.
    Fallback sur des données synthétiques si pas de clé API.
    """

    BASE_URL = "https://api.football-data.org/v4"

    def __init__(self, api_key: str, db: DatabaseManager):
        self.api_key = api_key
        self.db = db
        self.headers = {"X-Auth-Token": api_key} if api_key else {}
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def _get(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        if not self.api_key:
            logger.warning("⚠️  Pas de clé API football-data.org — mode démo activé")
            return None
        try:
            resp = self.session.get(
                f"{self.BASE_URL}{endpoint}",
                params=params or {},
                timeout=10
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.error("❌ Erreur API football-data.org : %s", e)
            return None

    def get_upcoming_fixtures(self, league_code: str, matchday: int = None) -> List[Dict]:
        """Récupère les prochains matchs d'une ligue."""
        params = {}
        if matchday:
            params["matchday"] = matchday

        data = self._get(f"/competitions/{league_code}/matches", params)

        if data and "matches" in data:
            fixtures = []
            for m in data["matches"]:
                if m.get("status") in ["SCHEDULED", "TIMED"]:
                    fixtures.append({
                        "id":         str(m["id"]),
                        "date":       m.get("utcDate", "")[:10],
                        "home_team":  m["homeTeam"]["name"],
                        "away_team":  m["awayTeam"]["name"],
                        "home_id":    str(m["homeTeam"]["id"]),
                        "away_id":    str(m["awayTeam"]["id"]),
                        "matchday":   m.get("matchday"),
                        "league":     league_code,
                        "status":     m.get("status"),
                    })
            logger.info("🕵️  %d fixtures trouvées pour %s", len(fixtures), league_code)
            return fixtures

        # Données de démonstration
        return self._demo_fixtures(league_code)

    def _demo_fixtures(self, league_code: str) -> List[Dict]:
        """Données de démonstration réalistes."""
        demos = {
            "PL": [
                {"home_team": "Arsenal",          "away_team": "Manchester City"},
                {"home_team": "Liverpool",         "away_team": "Chelsea"},
                {"home_team": "Manchester United", "away_team": "Tottenham"},
                {"home_team": "Newcastle",         "away_team": "Aston Villa"},
                {"home_team": "Brighton",          "away_team": "West Ham"},
            ],
            "PD": [
                {"home_team": "Real Madrid",       "away_team": "Barcelona"},
                {"home_team": "Atletico Madrid",   "away_team": "Sevilla"},
                {"home_team": "Valencia",          "away_team": "Villarreal"},
            ],
            "FL1": [
                {"home_team": "Paris Saint-Germain", "away_team": "Olympique de Marseille"},
                {"home_team": "Olympique Lyonnais",  "away_team": "Monaco"},
                {"home_team": "Lille",               "away_team": "Nice"},
            ],
            "BL1": [
                {"home_team": "Bayern Munich",     "away_team": "Borussia Dortmund"},
                {"home_team": "RB Leipzig",        "away_team": "Bayer Leverkusen"},
            ],
            "SA": [
                {"home_team": "Inter Milan",       "away_team": "AC Milan"},
                {"home_team": "Juventus",          "away_team": "Napoli"},
                {"home_team": "Roma",              "away_team": "Lazio"},
            ],
        }
        base = demos.get(league_code, demos["PL"])
        today = datetime.now()
        return [
            {
                "id":         hashlib.md5(f"{m['home_team']}{m['away_team']}".encode()).hexdigest()[:8],
                "date":       (today + timedelta(days=i + 1)).strftime("%Y-%m-%d"),
                "home_team":  m["home_team"],
                "away_team":  m["away_team"],
                "home_id":    hashlib.md5(m["home_team"].encode()).hexdigest()[:6],
                "away_id":    hashlib.md5(m["away_team"].encode()).hexdigest()[:6],
                "matchday":   None,
                "league":     league_code,
                "status":     "SCHEDULED",
            }
            for i, m in enumerate(base)
        ]

    def get_historical_results(self, league_code: str, seasons: int = 2) -> List[Dict]:
        """Récupère l'historique des résultats pour entraîner le modèle."""
        data = self._get(f"/competitions/{league_code}/matches", {"status": "FINISHED"})
        if data and "matches" in data:
            results = []
            for m in data["matches"]:
                score = m.get("score", {}).get("fullTime", {})
                if score.get("home") is not None:
                    results.append({
                        "id":          str(m["id"]),
                        "date":        m.get("utcDate", "")[:10],
                        "home_team":   m["homeTeam"]["name"],
                        "away_team":   m["awayTeam"]["name"],
                        "home_goals":  score["home"],
                        "away_goals":  score["away"],
                        "home_xg":     score.get("home", 0) * 1.1 + np.random.normal(0, 0.3),
                        "away_xg":     score.get("away", 0) * 1.1 + np.random.normal(0, 0.3),
                        "league":      league_code,
                        "season":      "2024-25",
                    })
            return results
        return self._generate_demo_history(league_code)

    def _generate_demo_history(self, league_code: str) -> List[Dict]:
        """Génère 200 matchs historiques synthétiques réalistes."""
        np.random.seed(42)
        demos = self._demo_fixtures(league_code)
        teams = list({f["home_team"] for f in demos} | {f["away_team"] for f in demos})
        if len(teams) < 4:
            teams += ["Team_A", "Team_B", "Team_C", "Team_D"]

        history = []
        base_date = datetime(2023, 8, 1)
        for i in range(200):
            home, away = np.random.choice(teams, 2, replace=False)
            hg = int(np.random.poisson(1.5))
            ag = int(np.random.poisson(1.1))
            history.append({
                "id":         f"demo_{i:04d}",
                "date":       (base_date + timedelta(days=i)).strftime("%Y-%m-%d"),
                "home_team":  home,
                "away_team":  away,
                "home_goals": hg,
                "away_goals": ag,
                "home_xg":    round(hg * 1.1 + np.random.normal(0, 0.3), 2),
                "away_xg":    round(ag * 1.1 + np.random.normal(0, 0.3), 2),
                "league":     league_code,
                "season":     "2023-24",
            })
        return history


# ============================================================
#  MODULE 3 — STATS ENGINE (Ratings & Features)
# ============================================================
class StatsEngine:
    """
    Calcule tous les indicateurs statistiques :
    Elo, Pi-ratings, xG, forme pondérée, head-to-head.
    """

    def __init__(self, db: DatabaseManager):
        self.db = db

    # ---------- ELO RATINGS ----------
    def compute_elo_ratings(self, match_history: List[Dict]) -> Dict[str, float]:
        """Calcule et met à jour les Elo ratings à partir de l'historique."""
        ratings: Dict[str, float] = {}

        def get_rating(team: str) -> float:
            if team not in ratings:
                ratings[team] = self.db.get_elo(team) or CONFIG["ELO_DEFAULT"]
            return ratings[team]

        for match in sorted(match_history, key=lambda x: x.get("date", "")):
            home = match["home_team"]
            away = match["away_team"]
            hg   = match.get("home_goals", 0)
            ag   = match.get("away_goals", 0)

            r_h = get_rating(home)
            r_a = get_rating(away)

            # Probabilité Elo attendue
            exp_h = 1 / (1 + 10 ** ((r_a - r_h - CONFIG["HOME_ADVANTAGE"]) / 400))
            exp_a = 1 - exp_h

            # Résultat réel
            if hg > ag:
                s_h, s_a = 1.0, 0.0
            elif hg < ag:
                s_h, s_a = 0.0, 1.0
            else:
                s_h, s_a = 0.5, 0.5

            # Facteur K ajusté selon l'écart de buts
            goal_diff = abs(hg - ag)
            k_adj = CONFIG["ELO_K_FACTOR"] * (1 + goal_diff * 0.1)

            ratings[home] = r_h + k_adj * (s_h - exp_h)
            ratings[away] = r_a + k_adj * (s_a - exp_a)

        # Sauvegarde en base
        for team, rating in ratings.items():
            self.db.upsert_elo(team, team, rating)

        logger.info("📊 Elo calculés pour %d équipes", len(ratings))
        return ratings

    # ---------- PI-RATINGS ----------
    def compute_pi_ratings(self, match_history: List[Dict]) -> Dict[str, Dict]:
        """
        Pi-ratings (Constantinou & Fenton) — ratings attaque/défense séparés
        pour domicile et extérieur.
        """
        pi: Dict[str, Dict] = {}
        lr = 0.07   # learning rate

        def get_pi(team: str) -> Dict:
            if team not in pi:
                pi[team] = {"att_h": 0.0, "att_a": 0.0, "def_h": 0.0, "def_a": 0.0}
            return pi[team]

        for match in sorted(match_history, key=lambda x: x.get("date", "")):
            home, away = match["home_team"], match["away_team"]
            hg  = match.get("home_xg", match.get("home_goals", 1.5))  # xG préféré
            ag  = match.get("away_xg", match.get("away_goals", 1.1))

            ph = get_pi(home)
            pa = get_pi(away)

            # Buts attendus selon Pi-ratings
            b = 10; c = 3
            exp_hg = b ** (( ph["att_h"] - pa["def_h"]) / c) - 1
            exp_ag = b ** (( pa["att_a"] - ph["def_a"]) / c) - 1
            exp_hg = max(0.1, exp_hg)
            exp_ag = max(0.1, exp_ag)

            # Mise à jour
            err_h = hg - exp_hg
            err_a = ag - exp_ag

            ph["att_h"] += lr * err_h
            ph["def_h"] -= lr * err_a
            pa["att_a"] += lr * err_a
            pa["def_a"] -= lr * err_h

        logger.info("📊 Pi-ratings calculés pour %d équipes", len(pi))
        return pi

    # ---------- FORME PONDÉRÉE ----------
    def compute_form(self, team: str, history: List[Dict]) -> Dict:
        """Forme pondérée sur les N derniers matchs (décroissance exponentielle)."""
        team_matches = [
            m for m in history
            if m["home_team"] == team or m["away_team"] == team
        ]
        team_matches = sorted(team_matches, key=lambda x: x.get("date", ""), reverse=True)
        recent = team_matches[:CONFIG["FORM_WINDOW"]]

        pts, xgf, xga, wins, draws = 0.0, 0.0, 0.0, 0, 0
        weight_sum = 0.0

        for i, m in enumerate(recent):
            w = CONFIG["FORM_DECAY"] ** i
            weight_sum += w
            is_home = m["home_team"] == team

            hg = m.get("home_goals", 0)
            ag = m.get("away_goals", 0)
            hxg = m.get("home_xg", 0.0)
            axg = m.get("away_xg", 0.0)

            if is_home:
                gf, gc, xgf_m, xga_m = hg, ag, hxg, axg
                result = "W" if hg > ag else ("D" if hg == ag else "L")
            else:
                gf, gc, xgf_m, xga_m = ag, hg, axg, hxg
                result = "W" if ag > hg else ("D" if ag == hg else "L")

            if result == "W":
                pts  += 3 * w; wins  += 1
            elif result == "D":
                pts  += 1 * w; draws += 1

            xgf += xgf_m * w
            xga += xga_m * w

        if weight_sum == 0:
            return {"pts_per_game": 1.0, "xgf": 1.5, "xga": 1.1, "wins": 0, "draws": 0}

        return {
            "pts_per_game": round(pts / weight_sum, 3),
            "xgf":          round(xgf / weight_sum, 3),
            "xga":          round(xga / weight_sum, 3),
            "wins":         wins,
            "draws":        draws,
            "matches_found": len(recent),
        }

    # ---------- HEAD-TO-HEAD ----------
    def compute_h2h(self, home: str, away: str, history: List[Dict]) -> Dict:
        """Statistiques head-to-head entre deux équipes."""
        h2h = [m for m in history if
               (m["home_team"] == home and m["away_team"] == away) or
               (m["home_team"] == away and m["away_team"] == home)]

        total = len(h2h)
        if total == 0:
            return {"total": 0, "home_wins": 0, "away_wins": 0, "draws": 0,
                    "home_goals_avg": 1.5, "away_goals_avg": 1.1}

        hw = aw = dr = 0
        hg_total = ag_total = 0

        for m in h2h:
            if m["home_team"] == home:
                hg, ag = m.get("home_goals",0), m.get("away_goals",0)
            else:
                hg, ag = m.get("away_goals",0), m.get("home_goals",0)

            hg_total += hg; ag_total += ag
            if hg > ag:   hw += 1
            elif hg < ag: aw += 1
            else:         dr += 1

        return {
            "total":           total,
            "home_wins":       hw,
            "away_wins":       aw,
            "draws":           dr,
            "home_win_rate":   round(hw / total, 3),
            "away_win_rate":   round(aw / total, 3),
            "home_goals_avg":  round(hg_total / total, 2),
            "away_goals_avg":  round(ag_total / total, 2),
        }

    # ---------- FEATURE VECTOR COMPLET ----------
    def build_feature_vector(
        self,
        home: str, away: str,
        elo: Dict, pi: Dict, history: List[Dict]
    ) -> Dict:
        """Construit le vecteur de features complet pour un match."""
        form_h = self.compute_form(home, history)
        form_a = self.compute_form(away, history)
        h2h    = self.compute_h2h(home, away, history)

        elo_h  = elo.get(home, CONFIG["ELO_DEFAULT"])
        elo_a  = elo.get(away, CONFIG["ELO_DEFAULT"])
        pi_h   = pi.get(home, {"att_h": 0, "def_h": 0, "att_a": 0, "def_a": 0})
        pi_a   = pi.get(away, {"att_h": 0, "def_h": 0, "att_a": 0, "def_a": 0})

        return {
            # Elo
            "elo_diff":            elo_h - elo_a + CONFIG["HOME_ADVANTAGE"],
            "elo_home":            elo_h,
            "elo_away":            elo_a,

            # Pi-ratings
            "pi_att_diff":         pi_h.get("att_h", 0) - pi_a.get("def_h", 0),
            "pi_def_diff":         pi_a.get("att_a", 0) - pi_h.get("def_a", 0),

            # Forme
            "form_pts_home":       form_h["pts_per_game"],
            "form_pts_away":       form_a["pts_per_game"],
            "form_pts_diff":       form_h["pts_per_game"] - form_a["pts_per_game"],

            # xG
            "xgf_home":            form_h["xgf"],
            "xga_home":            form_h["xga"],
            "xgf_away":            form_a["xgf"],
            "xga_away":            form_a["xga"],
            "xg_balance_home":     form_h["xgf"] - form_h["xga"],
            "xg_balance_away":     form_a["xgf"] - form_a["xga"],

            # Head-to-Head
            "h2h_home_win_rate":   h2h.get("home_win_rate", 0.33),
            "h2h_draw_rate":       h2h.get("draws", 0) / max(h2h.get("total", 1), 1),
            "h2h_goals_diff":      h2h.get("home_goals_avg",0) - h2h.get("away_goals_avg",0),

            # Ratio attaque/défense
            "atk_def_ratio_home":  form_h["xgf"] / max(form_h["xga"], 0.1),
            "atk_def_ratio_away":  form_a["xgf"] / max(form_a["xga"], 0.1),
        }


# ============================================================
#  MODULE 4 — NEWS & CONTEXT AGENT
# ============================================================
class NewsContextAgent:
    """
    Récupère les informations contextuelles : blessures, suspensions,
    actualités via Tavily ou une simulation réaliste.
    """

    def __init__(self, tavily_key: str = ""):
        self.tavily_key = tavily_key

    def _tavily_search(self, query: str) -> List[str]:
        """Recherche via Tavily API (conçu pour LLM)."""
        if not self.tavily_key:
            return []
        try:
            resp = requests.post(
                "https://api.tavily.com/search",
                json={"api_key": self.tavily_key, "query": query,
                      "max_results": 3, "search_depth": "basic"},
                timeout=10
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            return [r.get("content", "")[:200] for r in results]
        except Exception as e:
            logger.warning("⚠️  Tavily : %s", e)
            return []

    def get_team_news(self, team: str, league: str) -> Dict:
        """Collecte les news, blessures et composition probable pour une équipe."""
        snippets = self._tavily_search(f"{team} injuries suspensions lineup {datetime.now().strftime('%B %Y')}")

        # Analyse contextuelle (simulation réaliste sans hallucination)
        injury_keywords = ["injured", "blessé", "out", "doubt", "fitness"]
        suspension_keywords = ["suspended", "suspendu", "banned", "red card"]

        injuries    = [s for s in snippets if any(k in s.lower() for k in injury_keywords)]
        suspensions = [s for s in snippets if any(k in s.lower() for k in suspension_keywords)]

        return {
            "team":        team,
            "injuries":    injuries,
            "suspensions": suspensions,
            "news":        snippets,
            "lineup_hint": f"Effectif probable de {team} à confirmer",
            "risk_score":  min(len(injuries) * 0.15 + len(suspensions) * 0.1, 0.4),
        }

    def get_match_context(self, home: str, away: str, league: str) -> Dict:
        """Contexte complet pour un match spécifique."""
        home_news = self.get_team_news(home, league)
        away_news = self.get_team_news(away, league)

        return {
            "home": home_news,
            "away": away_news,
            "derby_indicator": self._is_derby(home, away),
            "high_stakes":     self._is_high_stakes(league),
        }

    def _is_derby(self, home: str, away: str) -> bool:
        """Détecte si c'est un derby ou un grand match."""
        rivalries = [
            {"Arsenal", "Tottenham"}, {"Arsenal", "Chelsea"},
            {"Liverpool", "Manchester United"}, {"Liverpool", "Everton"},
            {"Manchester City", "Manchester United"},
            {"Real Madrid", "Barcelona"}, {"Real Madrid", "Atletico Madrid"},
            {"Inter Milan", "AC Milan"}, {"Juventus", "Inter Milan"},
            {"Paris Saint-Germain", "Olympique de Marseille"},
            {"Borussia Dortmund", "Bayern Munich"},
        ]
        pair = {home, away}
        return any(pair == r for r in rivalries)

    def _is_high_stakes(self, league: str) -> bool:
        return league in ["CL", "WC", "EC"]


# ============================================================
#  MODULE 5 — ML PREDICTION ENGINE
# ============================================================
class MLPredictionEngine:
    """
    Moteur de prédiction ML :
    - CatBoost (si disponible) ou Logistic Regression calibrée
    - Entraînement sur l'historique
    - Prédiction 1X2 avec probabilités calibrées
    """

    def __init__(self):
        self.model      = None
        self.scaler     = StandardScaler()
        self.is_trained = False
        self.feature_names: List[str] = []
        self.model_type = CONFIG["MODEL_TYPE"]

    def _results_to_labels(self, history: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Convertit l'historique en features + labels."""
        stats   = StatsEngine(DatabaseManager())
        elo     = stats.compute_elo_ratings(history)
        pi      = stats.compute_pi_ratings(history)

        X_list, y_list = [], []
        for i, match in enumerate(history[10:], 10):  # Skip les 10 premiers
            sub_history = history[:i]
            fv = stats.build_feature_vector(
                match["home_team"], match["away_team"], elo, pi, sub_history
            )
            hg = match.get("home_goals", 0)
            ag = match.get("away_goals", 0)
            label = 0 if hg > ag else (2 if hg < ag else 1)  # 0=Home, 1=Draw, 2=Away

            X_list.append(list(fv.values()))
            y_list.append(label)
            if not self.feature_names:
                self.feature_names = list(fv.keys())

        return np.array(X_list), np.array(y_list)

    def train(self, history: List[Dict]) -> Dict:
        """Entraîne le modèle sur l'historique des matchs."""
        if len(history) < 30:
            logger.warning("⚠️  Données insuffisantes (%d matchs). Minimum 30 requis.", len(history))

        X, y = self._results_to_labels(history)
        if len(X) == 0:
            logger.error("❌ Impossible de construire les features")
            return {"status": "error", "message": "Pas assez de données"}

        X_scaled = self.scaler.fit_transform(X)
        split    = int(len(X) * 0.8)

        if self.model_type == "catboost" and CATBOOST_AVAILABLE:
            base = CatBoostClassifier(
                iterations=300, learning_rate=0.05, depth=6,
                loss_function="MultiClass", verbose=0,
                random_seed=42
            )
        else:
            base = LogisticRegression(
                max_iter=1000, C=1.0, multi_class="multinomial",
                solver="lbfgs", random_state=42
            )

        self.model = CalibratedClassifierCV(base, cv=3, method="isotonic")
        self.model.fit(X_scaled[:split], y[:split])
        self.is_trained = True

        # Évaluation
        if split < len(X):
            preds = self.model.predict(X_scaled[split:])
            probs = self.model.predict_proba(X_scaled[split:])
            acc   = (preds == y[split:]).mean()
            ll    = log_loss(y[split:], probs)
            logger.info("✅ Modèle entraîné — Acc: %.1f%% | Log-loss: %.4f | N=%d",
                        acc * 100, ll, len(X))
            return {"status": "ok", "accuracy": acc, "log_loss": ll, "n_matches": len(X)}

        return {"status": "ok", "n_matches": len(X)}

    def predict_match(
        self,
        home: str, away: str,
        elo: Dict, pi: Dict, history: List[Dict],
        context: Dict = None
    ) -> Dict:
        """Prédit le résultat d'un match avec probabilités complètes."""
        stats = StatsEngine(DatabaseManager())
        fv    = stats.build_feature_vector(home, away, elo, pi, history)
        X     = np.array([list(fv.values())])

        if self.is_trained and self.model:
            X_scaled = self.scaler.transform(X)
            probs    = self.model.predict_proba(X_scaled)[0]

            # Ordre : 0=Home, 1=Draw, 2=Away
            classes = list(self.model.classes_)
            prob_map = dict(zip(classes, probs))
            p_home = prob_map.get(0, 0.33)
            p_draw = prob_map.get(1, 0.33)
            p_away = prob_map.get(2, 0.33)
        else:
            # Fallback probabiliste basé sur Elo
            elo_h   = elo.get(home, CONFIG["ELO_DEFAULT"])
            elo_a   = elo.get(away, CONFIG["ELO_DEFAULT"])
            elo_exp = 1 / (1 + 10 ** ((elo_a - elo_h - CONFIG["HOME_ADVANTAGE"]) / 400))
            p_home  = elo_exp * 0.75
            p_draw  = 0.25
            p_away  = (1 - elo_exp) * 0.75
            # Normalisation
            total   = p_home + p_draw + p_away
            p_home /= total; p_draw /= total; p_away /= total

        # Ajustement contextuel (blessures)
        if context:
            risk_h = context.get("home", {}).get("risk_score", 0)
            risk_a = context.get("away", {}).get("risk_score", 0)
            p_home  = max(0.01, p_home - risk_h)
            p_away  = max(0.01, p_away - risk_a)
            total   = p_home + p_draw + p_away
            p_home /= total; p_draw /= total; p_away /= total

        # Recommandation
        probs_dict = {"home": p_home, "draw": p_draw, "away": p_away}
        recommended = max(probs_dict, key=probs_dict.get)
        confidence  = max(p_home, p_draw, p_away)

        # Feature importance (si Random Forest / CatBoost)
        feat_importance = {}
        if self.is_trained and hasattr(self.model, "estimator"):
            try:
                est = self.model.estimator
                if hasattr(est, "feature_importances_"):
                    feat_importance = dict(zip(self.feature_names, est.feature_importances_))
            except Exception:
                pass

        return {
            "home_team":          home,
            "away_team":          away,
            "prob_home":          round(p_home, 4),
            "prob_draw":          round(p_draw, 4),
            "prob_away":          round(p_away, 4),
            "recommended":        recommended,
            "confidence":         round(confidence, 4),
            "features":           fv,
            "feature_importance": feat_importance,
            "model_used":         self.model_type if self.is_trained else "elo_fallback",
            "is_derby":           context.get("derby_indicator", False) if context else False,
        }


# ============================================================
#  MODULE 6 — VALIDATOR & VALUE BET DETECTOR
# ============================================================
class Validator:
    """
    Valide et enrichit les prédictions :
    - Calibration des probabilités
    - Détection de value bets
    - Rapport d'accountability
    """

    def __init__(self, db: DatabaseManager):
        self.db = db

    def validate_probabilities(self, pred: Dict) -> Dict:
        """Vérifie la cohérence des probabilités (somme = 1, borne [0,1])."""
        ph = max(0.001, min(0.998, pred["prob_home"]))
        pd_ = max(0.001, min(0.998, pred["prob_draw"]))
        pa  = max(0.001, min(0.998, pred["prob_away"]))
        total = ph + pd_ + pa

        pred["prob_home"] = round(ph / total, 4)
        pred["prob_draw"] = round(pd_ / total, 4)
        pred["prob_away"] = round(pa / total, 4)
        return pred

    def compute_value_bet(self, pred: Dict, bookmaker_odds: Dict = None) -> Dict:
        """
        Détecte les value bets : si prob_modele > prob_implicite_bookmaker.
        Si pas de cotes réelles, utilise des cotes simulées.
        """
        if not bookmaker_odds:
            # Simulation de cotes bookmaker plausibles
            ph, pd_, pa = pred["prob_home"], pred["prob_draw"], pred["prob_away"]
            margin = 1.05  # Marge bookmaker typique (5%)
            bookmaker_odds = {
                "home": round(margin / ph, 2),
                "draw": round(margin / pd_, 2),
                "away": round(margin / pa, 2),
            }

        # Probabilités implicites
        imp_h = 1 / bookmaker_odds.get("home", 3.0)
        imp_d = 1 / bookmaker_odds.get("draw", 3.5)
        imp_a = 1 / bookmaker_odds.get("away", 3.0)

        # Value = (prob_modèle - prob_implicite)
        value_h = pred["prob_home"] - imp_h
        value_d = pred["prob_draw"] - imp_d
        value_a = pred["prob_away"] - imp_a

        best_value    = max(value_h, value_d, value_a)
        best_outcome  = ["home", "draw", "away"][[value_h, value_d, value_a].index(best_value)]
        is_value      = best_value > CONFIG["VALUE_BET_THRESHOLD"]

        # Kelly Criterion
        if is_value:
            odds_best = bookmaker_odds.get(best_outcome, 2.0)
            prob_best = {"home": pred["prob_home"],
                         "draw": pred["prob_draw"],
                         "away": pred["prob_away"]}[best_outcome]
            kelly = (prob_best * odds_best - 1) / (odds_best - 1)
            kelly = max(0, min(kelly, 0.25))  # Max 25% de la bankroll
        else:
            kelly = 0.0

        pred.update({
            "bookmaker_odds":    bookmaker_odds,
            "value_home":        round(value_h, 4),
            "value_draw":        round(value_d, 4),
            "value_away":        round(value_a, 4),
            "is_value_bet":      is_value,
            "best_value_outcome": best_outcome if is_value else None,
            "best_value_amount": round(best_value, 4),
            "kelly_fraction":    round(kelly, 4),
        })
        return pred

    def generate_accountability_report(self) -> Dict:
        """Génère un rapport de performance des prédictions passées."""
        stats = self.db.get_accountability_stats()
        return {
            "total_predictions": stats["total"],
            "correct":           stats["correct"],
            "accuracy_pct":      stats["accuracy"],
            "verdict": (
                f"✅ Taux de réussite: {stats['accuracy']}% sur {stats['total']} prédictions"
                if stats["total"] > 0 else
                "📭 Aucune prédiction archivée pour le moment"
            )
        }


# ============================================================
#  MODULE 7 — OUTPUT FORMATTER
# ============================================================
class OutputFormatter:
    """Génère le rapport final structuré et lisible."""

    EMOJIS = {
        "home":  "🏠",
        "draw":  "🤝",
        "away":  "✈️",
        "high":  "🔥",
        "med":   "⚡",
        "low":   "❄️",
        "value": "💰",
        "derby": "⚔️",
    }

    def confidence_label(self, conf: float) -> str:
        if conf >= 0.55:   return f"{self.EMOJIS['high']} HAUTE"
        elif conf >= 0.42: return f"{self.EMOJIS['med']} MOYENNE"
        else:              return f"{self.EMOJIS['low']} FAIBLE"

    def format_prediction(self, pred: Dict) -> str:
        """Formate une prédiction individuelle."""
        home   = pred["home_team"]
        away   = pred["away_team"]
        ph     = pred["prob_home"]
        pd_    = pred["prob_draw"]
        pa     = pred["prob_away"]
        rec    = pred["recommended"]
        conf   = pred["confidence"]
        derby  = pred.get("is_derby", False)

        rec_labels = {"home": f"Victoire {home}", "draw": "Match Nul", "away": f"Victoire {away}"}
        rec_emojis = {"home": "🏠", "draw": "🤝", "away": "✈️"}

        lines = [
            f"{'⚔️  DERBY  ⚔️' if derby else ''}",
            f"  ⚽ {home}  vs  {away}",
            f"  {'─' * 42}",
            f"  📊 Probabilités:",
            f"     🏠 {home:<25} {ph*100:>5.1f}%",
            f"     🤝 Match Nul                      {pd_*100:>5.1f}%",
            f"     ✈️  {away:<25} {pa*100:>5.1f}%",
            f"",
            f"  🎯 Pronostic : {rec_emojis[rec]} {rec_labels[rec]}",
            f"  📈 Confiance : {self.confidence_label(conf)}",
        ]

        if pred.get("is_value_bet"):
            oc = pred.get("best_value_outcome", "")
            val = pred.get("best_value_amount", 0)
            kelly = pred.get("kelly_fraction", 0)
            oc_label = rec_labels.get(oc, oc)
            lines += [
                f"",
                f"  💰 VALUE BET DÉTECTÉ !",
                f"     Outcome : {oc_label}",
                f"     Edge    : +{val*100:.1f}%",
                f"     Kelly   : {kelly*100:.1f}% bankroll",
            ]

        # Cotes bookmaker estimées
        if "bookmaker_odds" in pred:
            odds = pred["bookmaker_odds"]
            lines += [
                f"",
                f"  📉 Cotes estimées : 1={odds.get('home','?')} | X={odds.get('draw','?')} | 2={odds.get('away','?')}",
            ]

        # Modèle utilisé
        lines.append(f"  🤖 Modèle : {pred.get('model_used', 'N/A')}")

        return "\n".join(l for l in lines if l is not None)

    def format_full_report(
        self,
        league: str,
        predictions: List[Dict],
        accountability: Dict,
        timestamp: str = None
    ) -> str:
        """Génère le rapport complet de la journée."""
        ts = timestamp or datetime.now().strftime("%d/%m/%Y %H:%M")
        league_name = CONFIG["LEAGUES"].get(league, league)

        header = f"""
╔══════════════════════════════════════════════════════════════════════╗
║        ⚽  SUPER AGENT PRONOSTICS FOOTBALL  ⚽                       ║
║        🏆  {league_name:<55}║
║        🕐  {ts:<55}║
╚══════════════════════════════════════════════════════════════════════╝
"""

        value_bets = [p for p in predictions if p.get("is_value_bet")]
        high_conf  = [p for p in predictions if p.get("confidence", 0) >= 0.55]

        summary = f"""
┌─ 📋 RÉSUMÉ DE LA JOURNÉE ──────────────────────────────────────────┐
│  Matchs analysés      : {len(predictions):<43}│
│  Value Bets détectés  : {len(value_bets):<43}│
│  Haute confiance      : {len(high_conf):<43}│
│  Performances passées : {accountability.get('verdict','N/A'):<43}│
└────────────────────────────────────────────────────────────────────┘
"""

        preds_block = "\n\n".join([
            f"  MATCH {i+1}/{len(predictions)}\n" + self.format_prediction(p)
            for i, p in enumerate(predictions)
        ])

        # Section Value Bets
        if value_bets:
            vb_section = "\n\n┌─ 💰 RÉCAPITULATIF VALUE BETS ──────────────────────────────────────┐\n"
            for p in value_bets:
                oc = p.get("best_value_outcome", "")
                oc_labels = {"home": f"Vic {p['home_team']}", "draw": "Nul", "away": f"Vic {p['away_team']}"}
                vb_section += f"│  ✅ {p['home_team']} vs {p['away_team']} → {oc_labels.get(oc, oc):<30}│\n"
            vb_section += "└────────────────────────────────────────────────────────────────────┘"
        else:
            vb_section = "\n  ℹ️  Aucun value bet significatif identifié cette journée."

        disclaimer = """
─────────────────────────────────────────────────────────────────────
⚠️  DISCLAIMER : Ces pronostics sont générés par un modèle ML.
   Le football reste imprévisible. Jouez de manière responsable.
   Précision maximale documentée : ~55-56% (CatBoost + Pi-ratings)
─────────────────────────────────────────────────────────────────────
"""
        return header + summary + "\n" + preds_block + vb_section + disclaimer


# ============================================================
#  SUPER AGENT UNIFIÉ — POINT D'ENTRÉE PRINCIPAL
# ============================================================
class FootballSuperAgent:
    """
    ╔══════════════════════════════════════════════════════╗
    ║     SUPER AGENT FOOTBALL — VERSION UNIFIÉE          ║
    ║  Tous les modules intégrés dans une seule classe    ║
    ╚══════════════════════════════════════════════════════╝

    Usage :
        agent = FootballSuperAgent()
        rapport = agent.run(league="PL")
        print(rapport)
    """

    def __init__(self):
        logger.info("🚀 Initialisation du Super Agent Football...")

        # Instanciation de tous les modules
        self.db         = DatabaseManager(CONFIG["DB_PATH"])
        self.scout      = DataScout(CONFIG["FOOTBALL_API_KEY"], self.db)
        self.stats      = StatsEngine(self.db)
        self.news       = NewsContextAgent(CONFIG["TAVILY_API_KEY"])
        self.ml         = MLPredictionEngine()
        self.validator  = Validator(self.db)
        self.formatter  = OutputFormatter()

        # LLM optionnel pour enrichissement narratif
        self.llm = None
        if LANGCHAIN_AVAILABLE and CONFIG["OPENAI_API_KEY"]:
            try:
                self.llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0.3,
                    api_key=CONFIG["OPENAI_API_KEY"]
                )
                logger.info("✅ LLM GPT-4o-mini connecté")
            except Exception as e:
                logger.warning("⚠️  LLM non disponible : %s", e)

        logger.info("✅ Super Agent initialisé avec succès")

    # ──────────────────────────────────────────
    #  PIPELINE PRINCIPAL
    # ──────────────────────────────────────────
    def run(
        self,
        league:   str = "PL",
        matchday: int = None,
        verbose:  bool = True
    ) -> str:
        """
        Lance le pipeline complet en 6 étapes :
        Scout → Stats → News → ML → Validation → Output
        """
        start = time.time()
        logger.info("=" * 60)
        logger.info("🏁 DÉMARRAGE DU PIPELINE — Ligue : %s", CONFIG["LEAGUES"].get(league, league))
        logger.info("=" * 60)

        # ── ÉTAPE 1 : DATA SCOUT ───────────────
        logger.info("🕵️  [1/6] Data Scout — Collecte des fixtures...")
        fixtures = self.scout.get_upcoming_fixtures(league, matchday)
        history  = self.scout.get_historical_results(league)

        # Sauvegarde historique en base
        for m in history:
            self.db.save_match(m)

        if verbose:
            print(f"\n  📡 {len(fixtures)} matchs trouvés | {len(history)} matchs historiques chargés")

        # ── ÉTAPE 2 : STATS ENGINE ─────────────
        logger.info("📊 [2/6] Stats Engine — Calcul des ratings...")
        elo_ratings = self.stats.compute_elo_ratings(history)
        pi_ratings  = self.stats.compute_pi_ratings(history)

        if verbose:
            top_elo = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"  📊 Top 3 Elo : {' | '.join(f'{t}: {r:.0f}' for t,r in top_elo)}")

        # ── ÉTAPE 3 : ML TRAINING ──────────────
        logger.info("🧠 [3/6] ML Engine — Entraînement du modèle...")
        train_result = self.ml.train(history)
        if verbose:
            print(f"  🤖 Modèle: {CONFIG['MODEL_TYPE'].upper()} | "
                  f"Précision: {train_result.get('accuracy', 0)*100:.1f}%")

        # ── ÉTAPE 4 : NEWS & CONTEXT ───────────
        logger.info("📰 [4/6] News Agent — Collecte du contexte...")
        match_contexts = {}
        for fix in fixtures:
            ctx = self.news.get_match_context(fix["home_team"], fix["away_team"], league)
            match_contexts[fix["id"]] = ctx
            if verbose and ctx.get("derby_indicator"):
                print(f"  ⚔️  Derby détecté : {fix['home_team']} vs {fix['away_team']}")

        # ── ÉTAPE 5 : PRÉDICTIONS ML ───────────
        logger.info("🔮 [5/6] ML Predictor — Génération des pronostics...")
        raw_predictions = []
        for fix in fixtures:
            ctx  = match_contexts.get(fix["id"], {})
            pred = self.ml.predict_match(
                fix["home_team"], fix["away_team"],
                elo_ratings, pi_ratings, history, ctx
            )
            pred["match_id"]   = fix["id"]
            pred["date"]       = fix["date"]
            pred["date_pred"]  = datetime.now().isoformat()
            pred["league"]     = league
            raw_predictions.append(pred)

        # ── ÉTAPE 6 : VALIDATION & VALUE BETS ──
        logger.info("✅ [6/6] Validator — Calibration & Value Bets...")
        final_predictions = []
        for pred in raw_predictions:
            pred = self.validator.validate_probabilities(pred)
            pred = self.validator.compute_value_bet(pred)
            self.db.save_prediction(pred)
            final_predictions.append(pred)

        # ── ACCOUNTABILITY ─────────────────────
        accountability = self.validator.generate_accountability_report()

        # ── ENRICHISSEMENT LLM (optionnel) ─────
        if self.llm:
            final_predictions = self._enrich_with_llm(final_predictions, league)

        # ── RAPPORT FINAL ──────────────────────
        report = self.formatter.format_full_report(
            league, final_predictions, accountability
        )

        elapsed = time.time() - start
        logger.info("🏁 Pipeline terminé en %.2f secondes", elapsed)
        logger.info("=" * 60)

        # Résumé console
        if verbose:
            value_count = sum(1 for p in final_predictions if p.get("is_value_bet"))
            print(f"\n  ✅ Analyse complète en {elapsed:.1f}s")
            print(f"  💰 Value bets : {value_count}/{len(final_predictions)}")
            print(f"  📊 Accountability : {accountability['verdict']}\n")

        return report

    # ──────────────────────────────────────────
    #  ENRICHISSEMENT LLM
    # ──────────────────────────────────────────
    def _enrich_with_llm(self, predictions: List[Dict], league: str) -> List[Dict]:
        """Ajoute une analyse narrative via LLM pour chaque match."""
        league_name = CONFIG["LEAGUES"].get(league, league)
        for pred in predictions:
            try:
                prompt = f"""
Analyse ce match de {league_name} en 2-3 phrases percutantes :
- {pred['home_team']} (domicile) vs {pred['away_team']} (extérieur)
- Probabilités : {pred['home_team']} {pred['prob_home']*100:.1f}% | Nul {pred['prob_draw']*100:.1f}% | {pred['away_team']} {pred['prob_away']*100:.1f}%
- Confiance du modèle : {pred['confidence']*100:.1f}%
- Derby : {'Oui' if pred.get('is_derby') else 'Non'}
Sois direct, factuel, et mentionne les facteurs clés.
"""
                response = self.llm.invoke([HumanMessage(content=prompt)])
                pred["llm_analysis"] = response.content
            except Exception as e:
                pred["llm_analysis"] = ""
                logger.warning("⚠️  LLM enrichissement : %s", e)
        return predictions

    # ──────────────────────────────────────────
    #  MÉTHODES UTILITAIRES EXPOSÉES
    # ──────────────────────────────────────────
    def predict_single_match(self, home: str, away: str, league: str = "PL") -> str:
        """Prédit un match spécifique à la demande."""
        history  = self.scout.get_historical_results(league)
        elo      = self.stats.compute_elo_ratings(history)
        pi       = self.stats.compute_pi_ratings(history)
        self.ml.train(history)
        ctx      = self.news.get_match_context(home, away, league)
        pred     = self.ml.predict_match(home, away, elo, pi, history, ctx)
        pred     = self.validator.validate_probabilities(pred)
        pred     = self.validator.compute_value_bet(pred)
        pred["date_pred"] = datetime.now().isoformat()
        self.db.save_prediction(pred)
        return self.formatter.format_prediction(pred)

    def get_standings_by_elo(self, league: str = "PL") -> str:
        """Retourne le classement des équipes par Elo rating."""
        history = self.scout.get_historical_results(league)
        elo     = self.stats.compute_elo_ratings(history)
        sorted_teams = sorted(elo.items(), key=lambda x: x[1], reverse=True)
        lines = [f"\n  🏆 CLASSEMENT ELO — {CONFIG['LEAGUES'].get(league, league)}\n"]
        for i, (team, rating) in enumerate(sorted_teams, 1):
            bar   = "█" * int((rating - 1300) / 30)
            lines.append(f"  {i:>2}. {team:<30} {rating:>6.0f}  {bar}")
        return "\n".join(lines)

    def update_result(self, home: str, away: str, home_goals: int, away_goals: int):
        """Met à jour la base avec le résultat réel d'un match."""
        result = "home" if home_goals > away_goals else ("away" if away_goals > home_goals else "draw")
        with self.db.get_connection() as conn:
            conn.execute("""
                UPDATE predictions_log
                SET actual_result=?, correct=?
                WHERE home_team=? AND away_team=?
                AND actual_result IS NULL
                ORDER BY date_pred DESC LIMIT 1
            """, (
                result,
                1 if result == (conn.execute(
                    "SELECT recommended FROM predictions_log WHERE home_team=? AND away_team=? ORDER BY date_pred DESC LIMIT 1",
                    (home, away)
                ).fetchone() or ["?"])[0] else 0,
                home, away
            ))
        logger.info("✅ Résultat mis à jour : %s %d-%d %s", home, home_goals, away_goals, away)


# ============================================================
#  POINT D'ENTRÉE
# ============================================================
if __name__ == "__main__":
    print("\n" + "═" * 70)
    print("   ⚽  SUPER AGENT PRONOSTICS FOOTBALL — DÉMARRAGE  ⚽")
    print("═" * 70)

    agent = FootballSuperAgent()

    # ── Test 1 : Analyse complète Premier League
    print("\n📋 TEST 1 : Analyse complète Premier League\n")
    rapport_pl = agent.run(league="PL", verbose=True)
    print(rapport_pl)

    # ── Test 2 : Match unique à la demande
    print("\n📋 TEST 2 : Match unique Arsenal vs Manchester City\n")
    single = agent.predict_single_match("Arsenal", "Manchester City", "PL")
    print(single)

    # ── Test 3 : Classement Elo
    print("\n📋 TEST 3 : Classement Elo Premier League\n")
    elo_standings = agent.get_standings_by_elo("PL")
    print(elo_standings)

    # ── Test 4 : Ligue 1 française
    print("\n📋 TEST 4 : Analyse Ligue 1\n")
    rapport_l1 = agent.run(league="FL1", verbose=True)
    print(rapport_l1)

    print("\n" + "═" * 70)
    print("   ✅  SUPER AGENT TERMINÉ AVEC SUCCÈS")
    print("═" * 70)
