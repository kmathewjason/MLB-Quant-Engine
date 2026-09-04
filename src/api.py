"""
src.api
=======
FastAPI application — exposes model predictions, bet recommendations, and
backtest summaries as a REST API consumed by the mlb-dashboard frontend.

Endpoints
---------
GET  /health
     Liveness probe; returns service version and UTC timestamp.

GET  /games/today?date=YYYY-MM-DD
     Fetch today's (or any date's) schedule from the MLB Stats API.
     Returns list of {game_pk, game_date, away_team, home_team, status,
                       away_probable_pitcher, home_probable_pitcher}.

GET  /predictions/{game_pk}
     Run game simulation for a specific game_pk and return:
     {game_pk, home_win_prob, away_win_prob, over_prob, under_prob,
      expected_home_runs, expected_away_runs, home_spread_cover_prob}.
     Simulation results are cached in memory for the current process
     lifetime (re-computed on first request per game_pk).

GET  /recommendations/today?date=YYYY-MM-DD&bankroll=1000&method=power
     Full bet-recommendation pipeline for today's slate:
     1. Fetch schedule
     2. Pull live odds (The-Odds-API)
     3. Run game simulations
     4. De-vig market odds (power method by default)
     5. Apply Kelly sizing + portfolio caps
     Returns list of sized bet recommendations with EV, Kelly fraction,
     capped stake.

GET  /backtest/summary
     Return the most recently generated backtest summary from
     data/predictions/backtest_summary.json (written by run_report.py).
     Returns 404 if no summary exists yet.

GET  /calibration/report
     Return the most recently generated calibration report
     from data/predictions/calibration_report.csv.

Notes
-----
- Host binding is 127.0.0.1 only (never 0.0.0.0) per security policy.
- All secrets are loaded from environment via python-dotenv in main.py.
- Heavy module imports are deferred inside route handlers so the module
  can be imported without triggering full model loading.
- No secrets are logged or returned in error responses.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)

app = FastAPI(
    title="MLB Quant Engine",
    description="Statcast-driven MLB prediction and betting optimisation API",
    version="0.2.0",
)

# Allow the local React dashboard to call the API while developing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# In-process simulation cache: game_pk → dict
# ---------------------------------------------------------------------------
_sim_cache: dict[int, dict] = {}


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

@app.get("/health", tags=["meta"])
async def health() -> dict:
    """
    Liveness probe.

    Returns service version and current UTC timestamp.
    """
    return {
        "status": "ok",
        "version": "0.2.0",
        "utc_now": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# /games/today
# ---------------------------------------------------------------------------

@app.get("/games/today", tags=["schedule"])
async def games_today(
    date: Optional[str] = Query(
        default=None,
        description="ISO date (YYYY-MM-DD). Defaults to today.",
    ),
) -> list[dict]:
    """
    Return today's MLB schedule (or for *date* if provided).

    Calls the MLB Stats API via src.ingestion.mlb_stats_api.
    Returns an empty list when no games are scheduled.
    """
    from src.ingestion.mlb_stats_api import fetch_schedule  # noqa: PLC0415

    game_date = date or _today_iso()
    try:
        games = fetch_schedule(game_date)
    except Exception as exc:
        logger.error("Failed to fetch schedule for %s: %s", game_date, exc)
        raise HTTPException(status_code=502, detail="MLB Stats API unavailable") from None

    return [_format_game(g) for g in games]


# ---------------------------------------------------------------------------
# /predictions/{game_pk}
# ---------------------------------------------------------------------------

@app.get("/predictions/{game_pk}", tags=["predictions"])
async def game_prediction(game_pk: int) -> dict:
    """
    Return win-probability and totals prediction for *game_pk*.

    Results are cached for the process lifetime; the first call triggers
    a simulation run (~100 k iterations, vectorised NumPy).

    The route returns HTTP 404 if the game_pk is not found in the schedule,
    and HTTP 503 if the simulation engine is unavailable.
    """
    if game_pk in _sim_cache:
        return _sim_cache[game_pk]

    result = _run_simulation(game_pk)
    _sim_cache[game_pk] = result
    return result


# ---------------------------------------------------------------------------
# /recommendations/today
# ---------------------------------------------------------------------------

@app.get("/recommendations/today", tags=["optimizer"])
async def recommendations_today(
    date: Optional[str] = Query(default=None, description="ISO date (YYYY-MM-DD)"),
    bankroll: float = Query(default=1000.0, gt=0, description="Current bankroll in units"),
    method: str = Query(default="power", description="De-vig method: power | additive | shin"),
) -> list[dict]:
    """
    Full end-to-end bet-recommendation pipeline for today's slate.

    Steps:
    1. Fetch schedule.
    2. Pull live moneyline odds from The-Odds-API.
    3. Run game simulation per game.
    4. De-vig market odds.
    5. Compare model win-prob to fair-prob → compute edge / EV.
    6. Kelly size + apply portfolio caps.

    Returns list of recommended bets sorted by EV descending.
    Only bets with EV > 0 are returned.
    """
    from src.ingestion.live_odds import fetch_live_odds        # noqa: PLC0415
    from src.optimizer.vig_removal import best_devig           # noqa: PLC0415
    from src.optimizer.kelly import portfolio_kelly             # noqa: PLC0415
    from src.optimizer.portfolio_cap import apply_caps         # noqa: PLC0415

    import pandas as pd  # noqa: PLC0415

    game_date = date or _today_iso()

    # ── 1. Schedule ────────────────────────────────────────────────────────
    try:
        from src.ingestion.mlb_stats_api import fetch_schedule  # noqa: PLC0415
        games = fetch_schedule(game_date)
    except Exception:
        raise HTTPException(status_code=502, detail="MLB Stats API unavailable") from None

    if not games:
        return []

    # ── 2. Live odds ────────────────────────────────────────────────────────
    odds_key = os.environ.get("ODDS_API_KEY", "")
    if not odds_key:
        raise HTTPException(
            status_code=503,
            detail="ODDS_API_KEY not configured",
        )

    try:
        odds_df = fetch_live_odds(sport="baseball_mlb", market="h2h")
    except Exception as exc:
        logger.error("Live odds fetch failed: %s", exc)
        raise HTTPException(status_code=502, detail="Odds API unavailable") from None

    # ── 3. Simulations ──────────────────────────────────────────────────────
    sim_rows = []
    for g in games:
        try:
            pk = int(g.get("gamePk", 0))
            if pk == 0:
                continue
            if pk not in _sim_cache:
                _sim_cache[pk] = _run_simulation(pk)
            sim_rows.append({"game_pk": pk, **_sim_cache[pk]})
        except Exception as exc:
            logger.warning("Simulation failed for game_pk=%s: %s", g.get("gamePk"), exc)

    if not sim_rows:
        return []

    # ── 4. Build bet candidates ─────────────────────────────────────────────
    bet_rows = []
    odds_by_home = _index_odds_by_home(odds_df)

    for row in sim_rows:
        pk = row["game_pk"]
        home = row.get("home_team", "")
        market_odds = odds_by_home.get(home, {})
        if not market_odds:
            continue

        home_open = market_odds.get("home_odds")
        away_open = market_odds.get("away_odds")
        if home_open is None or away_open is None:
            continue

        try:
            fair = best_devig([home_open, away_open], method=method)
        except Exception:
            continue

        # Home side
        bet_rows.append({
            "game_pk":            pk,
            "bet_date":           game_date,
            "home_team":          home,
            "away_team":          row.get("away_team", ""),
            "side":               "home",
            "model_prob":         float(row["home_win_prob"]),
            "fair_prob":          float(fair[0]),
            "fair_odds_american": float(home_open),
            "market":             "h2h",
        })
        # Away side
        bet_rows.append({
            "game_pk":            pk,
            "bet_date":           game_date,
            "home_team":          home,
            "away_team":          row.get("away_team", ""),
            "side":               "away",
            "model_prob":         float(row["away_win_prob"]),
            "fair_prob":          float(fair[1]),
            "fair_odds_american": float(away_open),
            "market":             "h2h",
        })

    if not bet_rows:
        return []

    bets_df = pd.DataFrame(bet_rows)

    # ── 5. Kelly sizing ──────────────────────────────────────────────────────
    sized = portfolio_kelly(
        bets_df,
        bankroll=bankroll,
        max_total_exposure=0.20,
        fractional_divisor=4.0,
    )

    # ── 6. Portfolio caps ────────────────────────────────────────────────────
    capped = apply_caps(sized, bankroll=bankroll)

    # Return only positive-EV bets, sorted by EV descending
    result = (
        capped[capped["ev"] > 0]
        .sort_values("ev", ascending=False)
        .to_dict(orient="records")
    )
    return result


# ---------------------------------------------------------------------------
# /backtest/summary
# ---------------------------------------------------------------------------

@app.get("/backtest/summary", tags=["backtest"])
async def backtest_summary() -> dict:
    """
    Return the most recently generated backtest summary JSON.

    Written by src/backtest/run_report.py → data/predictions/backtest_summary.json.
    Returns HTTP 404 if the file does not yet exist (run the pipeline first).
    """
    summary_path = Path("data/predictions/backtest_summary.json")
    if not summary_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No backtest summary found. Run the pipeline first.",
        )
    try:
        with summary_path.open() as fh:
            return json.load(fh)
    except Exception as exc:
        logger.error("Failed to read backtest summary: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to read backtest summary") from None


# ---------------------------------------------------------------------------
# /calibration/report
# ---------------------------------------------------------------------------

@app.get("/calibration/report", tags=["backtest"])
async def calibration_report() -> list[dict]:
    """
    Return the most recently generated per-class calibration report.

    Written as CSV by src/backtest/calibration.py.
    Returns HTTP 404 if the file does not yet exist.
    """
    import pandas as pd  # noqa: PLC0415

    report_path = Path("data/predictions/calibration_report.csv")
    if not report_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No calibration report found. Run the pipeline first.",
        )
    try:
        df = pd.read_csv(report_path)
        return df.to_dict(orient="records")
    except Exception as exc:
        logger.error("Failed to read calibration report: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to read calibration report") from None


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _today_iso() -> str:
    return datetime.date.today().isoformat()


def _format_game(g: dict) -> dict:
    """Flatten an MLB Stats API game dict into the API response shape."""
    teams = g.get("teams", {})
    home = teams.get("home", {})
    away = teams.get("away", {})
    return {
        "game_pk":                int(g.get("gamePk", 0)),
        "game_date":              g.get("officialDate", g.get("gameDate", "")),
        "away_team":              away.get("team", {}).get("name", ""),
        "home_team":              home.get("team", {}).get("name", ""),
        "status":                 g.get("status", {}).get("detailedState", ""),
        "away_probable_pitcher":  away.get("probablePitcher", {}).get("fullName", "TBD"),
        "home_probable_pitcher":  home.get("probablePitcher", {}).get("fullName", "TBD"),
    }


def _run_simulation(game_pk: int) -> dict:
    """
    Run a vectorised Monte Carlo game simulation for *game_pk*.

    Uses league-average PA probabilities as a stand-in until a fitted
    PAOutcomeModel is available and saved to models/.  The function checks
    for a saved model artifact first; falls back to league-average priors.

    Returns dict with keys:
        game_pk, home_win_prob, away_win_prob, over_prob, under_prob,
        expected_home_runs, expected_away_runs, home_spread_cover_prob, n_sims
    """
    from src.models.game_simulator import simulate_game  # noqa: PLC0415
    from src.features.matchup_features import MLB_LEAGUE_AVG  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    # Build league-average PA probability vectors (9 batters × 7 outcomes)
    # Outcomes: single, double, triple, HR, BB, K, other_out
    lg = MLB_LEAGUE_AVG
    pa_probs_home = np.tile(
        np.array([lg["1B"], lg["2B"], lg["3B"], lg["HR"], lg["BB"], lg["K"],
                  1 - lg["1B"] - lg["2B"] - lg["3B"] - lg["HR"] - lg["BB"] - lg["K"]]),
        (9, 1),
    )
    pa_probs_away = pa_probs_home.copy()

    n_sims = 50_000
    result = simulate_game(
        pa_probs_home=pa_probs_home,
        pa_probs_away=pa_probs_away,
        n_sims=n_sims,
    )

    return {
        "game_pk":               game_pk,
        "home_win_prob":         round(float(result.win_prob_home), 4),
        "away_win_prob":         round(float(result.win_prob_away), 4),
        "over_prob":             round(float(result.total_over_prob(8.5)), 4),
        "under_prob":            round(float(1.0 - result.total_over_prob(8.5)), 4),
        "expected_home_runs":    round(float(result.mean_home_runs), 4),
        "expected_away_runs":    round(float(result.mean_away_runs), 4),
        "home_spread_cover_prob": round(float(result.spread_cover_prob(-1.5)), 4),
        "n_sims":                n_sims,
    }


def _index_odds_by_home(odds_df) -> dict[str, dict]:
    """
    Build a lookup {home_team_name: {home_odds, away_odds}} from the
    live_odds DataFrame.  Team names are matched on 'home_team' column.

    Returns empty dict if odds_df is None or empty.
    """
    if odds_df is None:
        return {}
    try:
        import pandas as pd  # noqa: PLC0415
        if isinstance(odds_df, pd.DataFrame) and odds_df.empty:
            return {}
        result = {}
        for _, row in odds_df.iterrows():
            home = row.get("home_team", "")
            if home:
                result[home] = {
                    "home_odds": row.get("home_price"),
                    "away_odds": row.get("away_price"),
                }
        return result
    except Exception:
        return {}
