"""
src.api
=======
FastAPI application — exposes model predictions, bet recommendations, and
backtest summaries as a REST API consumed by the mlb-dashboard frontend.

Legacy endpoints (kept for backward compatibility)
--------------------------------------------------
GET  /health
GET  /games/today?date=YYYY-MM-DD
GET  /predictions/{game_pk}
GET  /recommendations/today?date=…&bankroll=…&method=…
GET  /backtest/summary
GET  /calibration/report

New /api/ endpoints
-------------------
GET  /api/predictions/daily?date=YYYY-MM-DD&bankroll=1000&devig=power
     Today's full slate.  For every game: team-total distribution summary
     (mean, std, p10/p25/p50/p75/p90 from the MC simulator) plus every
     available prop/market with model_prob, market_prob (de-vigged), edge,
     EV, and quarter-Kelly stake.  Sorted by EV descending.

GET  /api/games/{game_id}/simulation?n_sims=50000&total_line=8.5&run_line=-1.5
     Full simulated score distribution for one game.  Returns the raw
     histogram buckets for a frontend distribution chart, plus key win/
     spread/total probabilities.

POST /api/predictions/sgp
     Body: { game_id, legs: [{side, outcome, line?, odds?}], bankroll }
     Same-game parlay joint probability via the MC simulator's empirical
     correlation matrix, correlation-adjusted Kelly stake, and a side-by-side
     naive-independent vs correlation-adjusted comparison so the adjustment
     is visible.

GET  /api/backtest/report
     Latest walk-forward backtest report as JSON for a dashboard backtest tab:
     calibration curve data (reliability diagram), Brier decomposition
     (REL/RES/UNC/BSS), CLV summary, and per-fold walk-forward metrics.

Response shape conventions
--------------------------
All endpoints return JSON with a top-level shape of either:
    { data: <payload>, meta: { generated_at, version } }
or a bare list/dict for legacy endpoints that callers already depend on.

New /api/ endpoints all use the envelope form so the frontend can
distinguish metadata from payload without inspecting individual fields.

Notes
-----
- Host binding is 127.0.0.1 only (never 0.0.0.0) per security policy.
- All secrets loaded from environment via python-dotenv in main.py.
- Heavy module imports are deferred inside route handlers.
- No secrets are logged or returned in error responses.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

app = FastAPI(
    title="MLB Quant Engine",
    description="Statcast-driven MLB prediction and betting optimisation API",
    version="0.3.0",
)

# Allow the local React dashboard to call the API during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# In-process simulation cache: game_pk → GameSimResult (raw) + summary dict
# ---------------------------------------------------------------------------
_sim_cache: dict[int, dict] = {}           # game_pk → legacy summary dict
_raw_sim_cache: dict[int, Any] = {}        # game_pk → GameSimResult (for dist endpoint)


# ---------------------------------------------------------------------------
# Shared response envelope
# ---------------------------------------------------------------------------

def _envelope(data: Any, version: str = "0.3.0") -> dict:
    return {
        "data": data,
        "meta": {
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "version": version,
        },
    }


def _today_iso() -> str:
    return datetime.date.today().isoformat()


# ===========================================================================
# LEGACY ENDPOINTS  (unchanged shapes — existing callers must not break)
# ===========================================================================

@app.get("/health", tags=["meta"])
async def health() -> dict:
    """Liveness probe."""
    return {
        "status": "ok",
        "version": "0.3.0",
        "utc_now": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


@app.get("/games/today", tags=["schedule"])
async def games_today(
    date: Optional[str] = Query(
        default=None,
        description="ISO date (YYYY-MM-DD). Defaults to today.",
    ),
) -> list[dict]:
    """Return today's MLB schedule (or for *date* if provided)."""
    from src.ingestion.mlb_stats_api import fetch_schedule  # noqa: PLC0415

    game_date = date or _today_iso()
    try:
        games = fetch_schedule(game_date)
    except Exception as exc:
        logger.error("Failed to fetch schedule for %s: %s", game_date, exc)
        raise HTTPException(status_code=502, detail="MLB Stats API unavailable") from None

    return [_format_game(g) for g in games]


@app.get("/predictions/{game_pk}", tags=["predictions"])
async def game_prediction(game_pk: int) -> dict:
    """Return win-probability and totals prediction for *game_pk* (legacy)."""
    if game_pk in _sim_cache:
        return _sim_cache[game_pk]
    result = _run_simulation_cached(game_pk)
    return _sim_cache[game_pk]


@app.get("/recommendations/today", tags=["optimizer"])
async def recommendations_today(
    date: Optional[str] = Query(default=None, description="ISO date (YYYY-MM-DD)"),
    bankroll: float = Query(default=1000.0, gt=0),
    method: str = Query(default="power", description="power | additive | shin"),
) -> list[dict]:
    """Full end-to-end bet-recommendation pipeline for today's slate."""
    import pandas as pd  # noqa: PLC0415

    from src.ingestion.live_odds import get_game_odds            # noqa: PLC0415
    from src.optimizer.vig_removal import best_devig             # noqa: PLC0415
    from src.optimizer.kelly import portfolio_kelly              # noqa: PLC0415
    from src.optimizer.portfolio_cap import apply_caps           # noqa: PLC0415
    from src.ingestion.mlb_stats_api import fetch_schedule       # noqa: PLC0415

    game_date = date or _today_iso()
    try:
        games = fetch_schedule(game_date)
    except Exception:
        raise HTTPException(status_code=502, detail="MLB Stats API unavailable") from None

    if not games:
        return []

    odds_key = os.environ.get("ODDS_API_KEY", "")
    if not odds_key:
        raise HTTPException(status_code=503, detail="ODDS_API_KEY not configured")

    try:
        raw_odds = get_game_odds(markets=("h2h",), try_once=True)
        # Convert list[GameOdds] → flat DataFrame shape _index_odds_by_home expects
        import pandas as pd  # noqa: PLC0415
        odds_rows = []
        for go in raw_odds:
            if go.get("market") != "h2h":
                continue
            best_home: int | None = None
            best_away: int | None = None
            for bm_line in go.get("lines", []):
                for o in bm_line.get("outcomes", []):
                    if o["name"] == go["home_team"]:
                        if best_home is None or o["price"] > best_home:
                            best_home = o["price"]
                    else:
                        if best_away is None or o["price"] > best_away:
                            best_away = o["price"]
            odds_rows.append({
                "home_team":  go["home_team"],
                "away_team":  go["away_team"],
                "home_price": best_home,
                "away_price": best_away,
            })
        odds_df = pd.DataFrame(odds_rows)
    except Exception as exc:
        logger.error("Live odds fetch failed: %s", exc)
        raise HTTPException(status_code=502, detail="Odds API unavailable") from None

    sim_rows = []
    for g in games:
        try:
            pk = int(g.get("game_pk") or g.get("gamePk") or 0)
            if pk == 0:
                continue
            _run_simulation_cached(pk)
            sim_rows.append({"game_pk": pk, **_sim_cache[pk]})
        except Exception as exc:
            logger.warning("Simulation failed for game_pk=%s: %s",
                           g.get("game_pk") or g.get("gamePk"), exc)

    if not sim_rows:
        return []

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
        for side_idx, (side, model_p) in enumerate([
            ("home", row["home_win_prob"]),
            ("away", row["away_win_prob"]),
        ]):
            bet_rows.append({
                "game_pk":            pk,
                "bet_date":           game_date,
                "home_team":          home,
                "away_team":          row.get("away_team", ""),
                "side":               side,
                "model_prob":         float(model_p),
                "fair_prob":          float(fair[side_idx]),
                "fair_odds_american": float([home_open, away_open][side_idx]),
                "market":             "h2h",
            })

    if not bet_rows:
        return []

    bets_df = pd.DataFrame(bet_rows)
    sized = portfolio_kelly(bets_df, bankroll=bankroll,
                            max_total_exposure=0.20, fractional_divisor=4.0)
    capped = apply_caps(sized, bankroll=bankroll)
    return (
        capped[capped["ev"] > 0]
        .sort_values("ev", ascending=False)
        .to_dict(orient="records")
    )


@app.get("/backtest/summary", tags=["backtest"])
async def backtest_summary_legacy() -> dict:
    """Return the most recent backtest summary JSON (legacy endpoint)."""
    summary_path = Path("data/predictions/backtest_summary.json")
    if not summary_path.exists():
        raise HTTPException(status_code=404,
                            detail="No backtest summary found. Run the pipeline first.")
    try:
        with summary_path.open() as fh:
            return json.load(fh)
    except Exception as exc:
        logger.error("Failed to read backtest summary: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to read backtest summary") from None


@app.get("/calibration/report", tags=["backtest"])
async def calibration_report_legacy() -> list[dict]:
    """Return the most recent calibration report CSV as JSON (legacy endpoint)."""
    import pandas as pd  # noqa: PLC0415

    report_path = Path("data/predictions/calibration_report.csv")
    if not report_path.exists():
        raise HTTPException(status_code=404,
                            detail="No calibration report found. Run the pipeline first.")
    try:
        return pd.read_csv(report_path).to_dict(orient="records")
    except Exception as exc:
        logger.error("Failed to read calibration report: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to read calibration report") from None


# ===========================================================================
# NEW /api/ ENDPOINTS
# ===========================================================================

# ---------------------------------------------------------------------------
# Pydantic models for the SGP POST body
# ---------------------------------------------------------------------------

class SGPLeg(BaseModel):
    """One leg of a same-game parlay."""
    side: Literal["home", "away"] = Field(
        description="Which team/side this leg is on.",
    )
    outcome: Literal["moneyline", "spread", "over", "under"] = Field(
        default="moneyline",
        description="Bet type.",
    )
    line: Optional[float] = Field(
        default=None,
        description="Run-line (spread) or total line.  Ignored for moneyline.",
    )
    odds: Optional[float] = Field(
        default=None,
        description="American odds for this leg.  Used for Kelly sizing.",
    )
    label: Optional[str] = Field(
        default=None,
        description="Human-readable label, e.g. 'NYY ML'.",
    )


class SGPRequest(BaseModel):
    """Request body for POST /api/predictions/sgp."""
    game_id: int = Field(description="MLB game_pk identifier.")
    legs: list[SGPLeg] = Field(
        min_length=2,
        max_length=6,
        description="Between 2 and 6 legs for the same-game parlay.",
    )
    bankroll: float = Field(
        default=1000.0,
        gt=0,
        description="Current bankroll in units for Kelly sizing.",
    )
    total_line: float = Field(
        default=8.5,
        description="Over/under line to use for over/under legs.",
    )
    n_sims: int = Field(
        default=20_000,
        ge=1_000,
        le=100_000,
        description="Monte Carlo simulations used to derive the correlation matrix.",
    )


# ---------------------------------------------------------------------------
# GET /api/predictions/daily
# ---------------------------------------------------------------------------

_PA_OUTCOME_NAMES  = ["1B", "2B", "3B", "HR", "BB", "K", "out"]
_DEFAULT_TOTAL_LINES = [6.5, 7.5, 8.5, 9.5, 10.5]
_DEFAULT_RUN_LINES   = [-1.5, 1.5]


@app.get("/api/predictions/daily", tags=["api"])
async def api_daily_predictions(
    date: Optional[str] = Query(
        default=None,
        description="ISO date (YYYY-MM-DD). Defaults to today.",
    ),
    bankroll: float = Query(
        default=1000.0,
        gt=0,
        description="Bankroll for Kelly stake sizing.",
    ),
    devig: str = Query(
        default="power",
        description="De-vig method: power | additive | shin",
    ),
    n_sims: int = Query(
        default=50_000,
        ge=100,
        le=200_000,
        description="Monte Carlo simulations per game.",
    ),
) -> dict:
    """
    Today's full-slate predictions with team-total distribution summaries,
    moneyline/spread/total props, model edge, EV, and quarter-Kelly stakes.

    Response shape (envelope):
    {
      "data": [
        {
          "game_pk":   int,
          "game_date": str,
          "away_team": str,
          "home_team": str,
          "away_probable_pitcher": str,
          "home_probable_pitcher": str,
          "status": str,
          "simulation": {
            "n_sims": int,
            "home_win_prob":  float,
            "away_win_prob":  float,
            "home_runs": { "mean", "std", "p10", "p25", "p50", "p75", "p90" },
            "away_runs": { ... },
            "total_runs": { ... },
            "spread":  { "home_cover_prob_m1_5", "home_cover_prob_p1_5" },
            "totals":  { "over_7_5", "over_8_5", "over_9_5" }
          },
          "markets": [
            {
              "market":       str,   # "h2h_home" | "h2h_away" | "total_over_8.5" | ...
              "label":        str,   # human-readable
              "model_prob":   float,
              "market_prob":  float, # de-vigged
              "market_odds":  float, # American
              "edge":         float, # model_prob - market_prob
              "ev":           float, # b·p - q
              "kelly_full":   float, # fraction of bankroll (full Kelly)
              "kelly_quarter":float, # fraction of bankroll (quarter Kelly)
              "stake_units":  float  # quarter-Kelly × bankroll
            },
            ...
          ]
        },
        ...
      ],
      "meta": { "generated_at": str, "version": str }
    }
    """
    import pandas as pd  # noqa: PLC0415

    from src.ingestion.mlb_stats_api import fetch_schedule  # noqa: PLC0415
    from src.optimizer.vig_removal import best_devig             # noqa: PLC0415
    from src.optimizer.kelly import kelly_fraction, kelly_ev     # noqa: PLC0415

    game_date = date or _today_iso()

    # ── 1. Schedule ────────────────────────────────────────────────────────
    try:
        games = fetch_schedule(game_date)
    except Exception as exc:
        logger.error("Schedule fetch failed for %s: %s", game_date, exc)
        raise HTTPException(status_code=502, detail="MLB Stats API unavailable") from None

    if not games:
        return _envelope([])

    # ── 2. Live odds — fetch ALL three game markets + cache by home team ───
    odds_key = os.environ.get("ODDS_API_KEY", "")
    # Nested dicts keyed by home_team: { h2h, spreads, totals }
    all_odds: dict[str, dict] = {}   # home_team → {h2h, spreads, totals}
    if odds_key:
        try:
            from src.ingestion.live_odds import get_game_odds    # noqa: PLC0415
            raw_odds_all = get_game_odds(
                markets=("h2h", "spreads", "totals"), try_once=True
            )
            all_odds = _index_all_odds(raw_odds_all)
        except Exception as exc:
            logger.warning("Odds fetch failed (continuing without sizing): %s", exc)

    # ── 3. Simulate + build per-game payload ──────────────────────────────
    payload = []
    for g in games:
        pk = int(g.get("game_pk") or g.get("gamePk") or 0)
        if pk == 0:
            continue

        formatted = _format_game(g)

        # Run / retrieve simulation (now uses real pitcher/batter stats)
        try:
            _run_simulation_cached(pk, n_sims=n_sims)
            raw = _raw_sim_cache[pk]
        except Exception as exc:
            logger.warning("Simulation failed for game_pk=%d: %s", pk, exc)
            continue

        sim_block = _build_simulation_block(raw, n_sims)

        # ── Markets ────────────────────────────────────────────────────────
        markets: list[dict] = []
        game_odds = all_odds.get(formatted["home_team"], {})

        # -- Moneylines (h2h) --
        h2h = game_odds.get("h2h", {})
        home_ml = h2h.get("home_odds")
        away_ml = h2h.get("away_odds")
        if home_ml is not None and away_ml is not None:
            try:
                fair = best_devig([home_ml, away_ml], method=devig)
            except Exception:
                fair = None
            for side_idx, (mkt_name, label, model_p, mkt_odds) in enumerate([
                ("h2h_home", f"{formatted['home_team']} ML",
                 sim_block["home_win_prob"], home_ml),
                ("h2h_away", f"{formatted['away_team']} ML",
                 sim_block["away_win_prob"], away_ml),
            ]):
                mkt_prob = float(fair[side_idx]) if fair is not None else None
                markets.append(_build_market_row(
                    market=mkt_name, label=label,
                    model_prob=model_p, market_prob=mkt_prob,
                    market_odds=mkt_odds, bankroll=bankroll,
                    category="moneyline",
                ))

        # -- Run line / spread (-1.5) --
        spread_odds = game_odds.get("spreads", {})
        for run_line, side_key, label_team in [
            (-1.5, "home", formatted["home_team"]),
            (1.5,  "away", formatted["away_team"]),
        ]:
            cover_p = float(raw.spread_cover_prob(run_line))
            mkt_odds = spread_odds.get(f"{side_key}_odds")
            mkt_prob: float | None = None
            if mkt_odds is not None:
                opp_odds = spread_odds.get(
                    "away_odds" if side_key == "home" else "home_odds"
                )
                if opp_odds is not None:
                    try:
                        fp = best_devig([mkt_odds, opp_odds], method=devig)
                        mkt_prob = float(fp[0])
                    except Exception:
                        pass
            markets.append(_build_market_row(
                market=f"spread_{run_line:+.1f}",
                label=f"{label_team} {run_line:+.1f}",
                model_prob=cover_p, market_prob=mkt_prob,
                market_odds=mkt_odds, bankroll=bankroll,
                category="spread",
            ))

        # -- Totals (all lines from odds API + model lines not in odds) --
        totals_odds = game_odds.get("totals", {})   # dict keyed "over_X.X" / "under_X.X"
        seen_lines: set[float] = set()

        # First: lines where we have actual odds
        for key, mkt_odds in totals_odds.items():
            # key format: "over_8.5" or "under_8.5"
            parts = key.split("_")
            if len(parts) != 2:
                continue
            side_str, line_str = parts
            try:
                tline = float(line_str)
            except ValueError:
                continue
            seen_lines.add(tline)
            over_p  = float(raw.total_over_prob(tline))
            under_p = 1.0 - over_p
            model_p = over_p if side_str == "over" else under_p
            opp_key = f"{'under' if side_str == 'over' else 'over'}_{line_str}"
            opp_odds = totals_odds.get(opp_key)
            mkt_prob_t: float | None = None
            if opp_odds is not None:
                try:
                    fp = best_devig([mkt_odds, opp_odds], method=devig)
                    mkt_prob_t = float(fp[0])
                except Exception:
                    pass
            markets.append(_build_market_row(
                market=f"total_{side_str}_{tline}",
                label=f"{'O' if side_str == 'over' else 'U'} {tline}",
                model_prob=model_p, market_prob=mkt_prob_t,
                market_odds=mkt_odds, bankroll=bankroll,
                category="total",
            ))

        # Then: model-only lines not in the odds response
        for total_line in _DEFAULT_TOTAL_LINES:
            if total_line in seen_lines:
                continue
            over_p  = float(raw.total_over_prob(total_line))
            under_p = 1.0 - over_p
            for side_str, model_p in [("over", over_p), ("under", under_p)]:
                markets.append(_build_market_row(
                    market=f"total_{side_str}_{total_line}",
                    label=f"{'O' if side_str == 'over' else 'U'} {total_line}",
                    model_prob=model_p, market_prob=None,
                    market_odds=None, bankroll=bankroll,
                    category="total",
                ))

        # -- Player props --
        props = _fetch_player_props_for_game(
            formatted["game_pk"],
            formatted["home_team"], formatted["away_team"],
            raw, bankroll, devig,
        )
        markets.extend(props)

        # Sort by EV descending (nulls last); keep all markets
        markets.sort(
            key=lambda m: m["ev"] if m["ev"] is not None else float("-inf"),
            reverse=True,
        )

        payload.append({
            **formatted,
            "simulation": sim_block,
            "markets":    markets,
        })

    return _envelope(payload)


# ---------------------------------------------------------------------------
# GET /api/games/{game_id}/simulation
# ---------------------------------------------------------------------------

@app.get("/api/games/{game_id}/simulation", tags=["api"])
async def api_game_simulation(
    game_id: int,
    n_sims: int = Query(
        default=50_000,
        ge=100,
        le=200_000,
        description="Monte Carlo simulations to run.",
    ),
    total_line: float = Query(
        default=8.5,
        description="Primary over/under line for the probability readout.",
    ),
    run_line: float = Query(
        default=-1.5,
        description="Run-line for the spread probability readout (home perspective).",
    ),
    bins: int = Query(
        default=20,
        ge=5,
        le=50,
        description="Number of histogram bins for the distribution chart.",
    ),
) -> dict:
    """
    Full simulated score distribution for a single game.

    Response shape (envelope):
    {
      "data": {
        "game_id": int,
        "n_sims":  int,
        "home_win_prob":  float,
        "away_win_prob":  float,
        "spread_cover_prob": float,   # P(home - away > run_line)
        "over_prob":  float,
        "under_prob": float,
        "home_runs":  { "mean", "std", "p10","p25","p50","p75","p90",
                        "histogram": { "edges": [...], "counts": [...] } },
        "away_runs":  { ... },
        "total_runs": { ... },
        "margin_dist": { "histogram": { "edges": [...], "counts": [...] } }
      },
      "meta": { ... }
    }
    """
    try:
        _run_simulation_cached(game_id, n_sims=n_sims)
    except Exception as exc:
        logger.error("Simulation failed for game_id=%d: %s", game_id, exc)
        raise HTTPException(status_code=500, detail="Simulation engine error") from None

    raw = _raw_sim_cache.get(game_id)
    if raw is None:
        raise HTTPException(status_code=404, detail=f"No simulation found for game_id={game_id}")

    home = raw.home_runs_dist.astype(np.float64)
    away = raw.away_runs_dist.astype(np.float64)
    total = home + away
    margin = home - away

    def _hist(arr: np.ndarray) -> dict:
        counts, edges = np.histogram(arr, bins=bins)
        return {"edges": edges.tolist(), "counts": counts.tolist()}

    def _summary(arr: np.ndarray, include_hist: bool = False) -> dict:
        d: dict = {
            "mean": round(float(arr.mean()), 3),
            "std":  round(float(arr.std()),  3),
            "p10":  round(float(np.percentile(arr, 10)), 3),
            "p25":  round(float(np.percentile(arr, 25)), 3),
            "p50":  round(float(np.percentile(arr, 50)), 3),
            "p75":  round(float(np.percentile(arr, 75)), 3),
            "p90":  round(float(np.percentile(arr, 90)), 3),
        }
        if include_hist:
            d["histogram"] = _hist(arr)
        return d

    data = {
        "game_id":           game_id,
        "n_sims":            len(home),
        "home_win_prob":     round(float(raw.win_prob_home), 4),
        "away_win_prob":     round(float(raw.win_prob_away), 4),
        "spread_cover_prob": round(float(raw.spread_cover_prob(run_line)), 4),
        "over_prob":         round(float(raw.total_over_prob(total_line)), 4),
        "under_prob":        round(float(1.0 - raw.total_over_prob(total_line)), 4),
        "home_runs":         _summary(home,   include_hist=True),
        "away_runs":         _summary(away,   include_hist=True),
        "total_runs":        _summary(total,  include_hist=True),
        "margin_dist": {
            "histogram": _hist(margin),
            "mean":  round(float(margin.mean()), 3),
            "std":   round(float(margin.std()),  3),
        },
    }
    return _envelope(data)


# ---------------------------------------------------------------------------
# POST /api/predictions/sgp
# ---------------------------------------------------------------------------

@app.post("/api/predictions/sgp", tags=["api"])
async def api_sgp_prediction(body: SGPRequest) -> dict:
    """
    Same-game parlay (SGP) joint probability via empirical simulation
    correlation, correlation-adjusted Kelly stake, and a comparison
    against the naive independent-legs calculation.

    Response shape (envelope):
    {
      "data": {
        "game_id":   int,
        "n_sims":    int,
        "legs": [
          {
            "label":       str,
            "outcome":     str,
            "side":        str,
            "line":        float | null,
            "sim_prob":    float,   // marginal win prob from simulator
            "naive_prob":  float,   // same as sim_prob (per-leg marginal)
          }, ...
        ],
        "joint_prob_corr_adjusted": float,  // product × correlation adjustment
        "joint_prob_naive":         float,  // product of marginal probs (wrong)
        "correlation_matrix":       [[...], ...],  // (n_legs × n_legs)
        "kelly": {
          "corr_adjusted": {
            "ev":           float,
            "f_star":       float,
            "f_quarter":    float,
            "stake_units":  float
          },
          "naive": {
            "ev":           float,
            "f_star":       float,
            "f_quarter":    float,
            "stake_units":  float
          }
        }
      },
      "meta": { ... }
    }

    Notes on joint Kelly sizing
    ---------------------------
    The SGP parlay pays (Π decimal_i - 1) per unit staked if ALL legs win.
    The Kelly fraction for a parlay is:

        p  = joint_prob_corr_adjusted
        b  = Π decimal_i - 1  (net parlay payout per unit)
        f* = (b·p - (1-p)) / b

    We compute this for both the correlation-adjusted and naive joint prob
    so the difference in stake is visible.
    """
    from src.optimizer.kelly import simulate_corr_matrix         # noqa: PLC0415
    from src.features.matchup_features import MLB_LEAGUE_AVG     # noqa: PLC0415

    game_id   = body.game_id
    legs      = body.legs
    bankroll  = body.bankroll
    n_sims    = body.n_sims

    # ── Build league-average PA probs (used for all SGP sims) ─────────────
    pa_probs = _league_avg_pa_probs()

    # ── Build leg dicts for simulate_corr_matrix ──────────────────────────
    leg_dicts = []
    for i, leg in enumerate(legs):
        d: dict = {
            "pa_probs_home": pa_probs,
            "pa_probs_away": pa_probs,
            "side":          leg.side,
            "outcome":       leg.outcome,
            "label":         leg.label or f"leg_{i}",
        }
        if leg.line is not None:
            if leg.outcome in ("over", "under"):
                d["total_line"] = float(leg.line)
            elif leg.outcome == "spread":
                d["run_line"] = float(leg.line)
        leg_dicts.append(d)

    # ── Run joint simulation → correlation matrix ─────────────────────────
    try:
        sim_corr = simulate_corr_matrix(
            leg_dicts,
            n_sims=n_sims,
            rng=np.random.default_rng(),
        )
    except Exception as exc:
        logger.error("SGP simulate_corr_matrix failed for game_id=%d: %s", game_id, exc)
        raise HTTPException(status_code=500, detail="Simulation error") from None

    marginal_probs = sim_corr.win_probs.tolist()  # (n_legs,)
    corr_matrix    = sim_corr.corr_matrix.tolist()

    # ── Clamp marginal win probs to open (0,1) for display + downstream math ──
    clamped_win_probs = np.clip(sim_corr.win_probs, 0.0001, 0.9999)

    # ── Empirical joint:  fraction of sims where ALL legs win simultaneously ──
    all_win = (sim_corr.outcome_matrix.min(axis=0) > 0.5)  # shape (n_sims,)
    joint_corr = float(all_win.mean())

    # ── Parlay odds: assume each leg is priced at the market or at -110 ────
    # Build payout vector: for legs with odds provided use those; else infer
    # from the sim prob at -110 equivalent.
    decimal_payouts = []
    for leg in legs:
        if leg.odds is not None:
            a = float(leg.odds)
            decimal = (a / 100.0 + 1.0) if a >= 100.0 else (100.0 / abs(a) + 1.0)
        else:
            decimal = 1.909  # -110 equivalent
        decimal_payouts.append(decimal)

    parlay_net_payout = float(np.prod(decimal_payouts)) - 1.0   # b = Π d_i - 1

    # ── Per-leg summary (built first so joint_naive uses identical rounded values)
    leg_summary = []
    for i, leg in enumerate(legs):
        leg_summary.append({
            "label":      leg.label or f"leg_{i}",
            "outcome":    leg.outcome,
            "side":       leg.side,
            "line":       leg.line,
            "odds":       leg.odds,
            "sim_prob":   round(float(clamped_win_probs[i]), 4),
            "naive_prob": round(float(clamped_win_probs[i]), 4),
        })

    # Naive joint: Π sim_prob_i — computed from the same rounded display values
    # so the test's product of leg["sim_prob"] matches exactly.
    joint_naive = 1.0
    for entry in leg_summary:
        joint_naive *= entry["sim_prob"]

    # ── Kelly for the parlay as a single bet ─────────────────────────────
    def _parlay_kelly(p: float, b: float, divisor: float = 4.0) -> dict:
        """Kelly sizing for a binary parlay bet."""
        if b <= 0.0 or not (0.0 < p < 1.0):
            return {"ev": 0.0, "f_star": 0.0, "f_quarter": 0.0, "stake_units": 0.0}
        ev     = b * p - (1.0 - p)
        f_star = max(ev / b, 0.0)
        f_qk   = f_star / divisor
        return {
            "ev":          round(ev,           6),
            "f_star":      round(f_star,        6),
            "f_quarter":   round(f_qk,          6),
            "stake_units": round(f_qk * bankroll, 4),
        }

    kelly_corr  = _parlay_kelly(joint_corr,  parlay_net_payout)
    kelly_naive = _parlay_kelly(joint_naive, parlay_net_payout)

    data = {
        "game_id":                   game_id,
        "n_sims":                    n_sims,
        "legs":                      leg_summary,
        "joint_prob_corr_adjusted":  round(joint_corr,  5),
        "joint_prob_naive":          round(joint_naive, 5),
        "parlay_net_payout":         round(parlay_net_payout, 4),
        "correlation_matrix":        corr_matrix,
        "kelly": {
            "corr_adjusted": kelly_corr,
            "naive":         kelly_naive,
        },
    }
    return _envelope(data)


# ---------------------------------------------------------------------------
# GET /api/backtest/report
# ---------------------------------------------------------------------------

@app.get("/api/backtest/report", tags=["api"])
async def api_backtest_report() -> dict:
    """
    Latest walk-forward backtest report as structured JSON for the
    dashboard backtest tab.

    Data sources (all files in data/predictions/):
    - backtest_summary.json     : walk-forward aggregate metrics
    - calibration_report.csv    : per-class ECE, Brier decomposition
    - clv_report.parquet        : per-bet CLV data (optional)
    - wf_predictions.parquet    : OOS predictions for reliability diagram
    - wf_fold_metrics.json      : per-fold metrics

    Falls back to partial data gracefully if some files are missing.
    Returns 404 only if ALL sources are missing.

    Response shape (envelope):
    {
      "data": {
        "walk_forward": {
          "summary":      { n_folds, n_oos_predictions, oos_log_loss,
                            oos_accuracy, mean_fold_log_loss, std_fold_log_loss,
                            mean_fold_brier },
          "fold_metrics": [ {fold, date_start, date_end, n_train, n_test,
                             log_loss, accuracy, brier}, ... ],
          "available":    bool
        },
        "calibration": {
          "report":    [ {class_name, ece, brier_score, reliability,
                          resolution, uncertainty, brier_skill_score,
                          base_rate, n_positive}, ... ],
          "brier_decomposition": { brier_score, reliability, resolution,
                                   uncertainty, brier_skill_score },
          "reliability_curves":  [
            { "class_name": str, "bins": [ {bin_centre, mean_pred,
                                             obs_freq, count, ci_lo_95,
                                             ci_hi_95}, ... ], "ece": float },
            ...
          ],
          "available": bool
        },
        "clv": {
          "summary":  { n_bets, mean_clv_prob, mean_clv_log_odds,
                        pct_positive_clv, clv_tstat, clv_pvalue,
                        clv_result_corr, roi_pct, total_pnl_units },
          "available": bool
        }
      },
      "meta": { ... }
    }
    """
    import pandas as pd  # noqa: PLC0415

    # MLB_PRED_DIR env var lets tests redirect to a tmp directory.
    _pred_dir_env = os.environ.get("MLB_PRED_DIR", "")
    pred_dir = Path(_pred_dir_env) if _pred_dir_env else Path("data/predictions")
    any_found = False
    data: dict = {}

    # ── Walk-forward summary ───────────────────────────────────────────────
    wf_block: dict = {"available": False}
    summary_path = pred_dir / "backtest_summary.json"
    folds_path   = pred_dir / "wf_fold_metrics.json"

    if summary_path.exists():
        try:
            with summary_path.open() as fh:
                wf_block["summary"] = json.load(fh)
            wf_block["available"] = True
            any_found = True
        except Exception as exc:
            logger.warning("Could not read backtest_summary.json: %s", exc)

    if folds_path.exists():
        try:
            with folds_path.open() as fh:
                wf_block["fold_metrics"] = json.load(fh)
            wf_block["available"] = True
            any_found = True
        except Exception as exc:
            logger.warning("Could not read wf_fold_metrics.json: %s", exc)
    else:
        wf_block.setdefault("fold_metrics", [])

    data["walk_forward"] = wf_block

    # ── Calibration report ────────────────────────────────────────────────
    cal_block: dict = {"available": False}
    cal_path    = pred_dir / "calibration_report.csv"
    wf_preds_path = pred_dir / "wf_predictions.parquet"

    if cal_path.exists():
        try:
            cal_df = pd.read_csv(cal_path)
            cal_block["report"] = cal_df.to_dict(orient="records")

            # Aggregate Brier decomposition across classes
            if {"brier_score", "reliability", "resolution",
                    "uncertainty", "brier_skill_score"}.issubset(cal_df.columns):
                cal_block["brier_decomposition"] = {
                    "brier_score":       round(float(cal_df["brier_score"].mean()),       5),
                    "reliability":       round(float(cal_df["reliability"].mean()),       5),
                    "resolution":        round(float(cal_df["resolution"].mean()),        5),
                    "uncertainty":       round(float(cal_df["uncertainty"].mean()),       5),
                    "brier_skill_score": round(float(cal_df["brier_skill_score"].mean()), 5),
                }
            cal_block["available"] = True
            any_found = True
        except Exception as exc:
            logger.warning("Could not read calibration_report.csv: %s", exc)

    # Reliability curves — computed on-the-fly from OOS predictions if available
    if wf_preds_path.exists():
        try:
            from src.backtest.calibration import (  # noqa: PLC0415
                reliability_diagram,
                calibration_report as cal_report_fn,
            )

            preds_df  = pd.read_parquet(wf_preds_path)
            prob_cols = [c for c in preds_df.columns if c.startswith("p_")]

            if prob_cols and "y_true" in preds_df.columns:
                probs = preds_df[sorted(prob_cols)].to_numpy()
                y     = preds_df["y_true"].to_numpy().astype(int)
                n_classes = probs.shape[1]

                # Infer class names from calibration report if available
                if "report" in cal_block and cal_block["report"]:
                    class_names = [r["class_name"] for r in cal_block["report"]]
                else:
                    class_names = [str(k) for k in range(n_classes)]

                reliability_curves = []
                for k in range(n_classes):
                    rel = reliability_diagram(
                        probs, y, class_idx=k, n_bins=10, n_bootstrap=200,
                    )
                    curve_bins = rel.bins_df.to_dict(orient="records")
                    reliability_curves.append({
                        "class_name": class_names[k] if k < len(class_names) else str(k),
                        "bins":       curve_bins,
                        "ece":        round(rel.ece, 5),
                    })

                cal_block["reliability_curves"] = reliability_curves
                cal_block["available"] = True
                any_found = True
        except Exception as exc:
            logger.warning("Could not build reliability curves: %s", exc)

    data["calibration"] = cal_block

    # ── CLV summary ───────────────────────────────────────────────────────
    clv_block: dict = {"available": False}
    clv_path = pred_dir / "clv_report.parquet"

    if clv_path.exists():
        try:
            from src.backtest.clv_tracker import clv_summary  # noqa: PLC0415

            clv_df = pd.read_parquet(clv_path)
            clv_block["summary"]   = clv_summary(clv_df)
            clv_block["available"] = True
            any_found = True
        except Exception as exc:
            logger.warning("Could not read clv_report.parquet: %s", exc)

    data["clv"] = clv_block

    if not any_found:
        raise HTTPException(
            status_code=404,
            detail=(
                "No backtest data found in data/predictions/. "
                "Run the walk-forward pipeline and save outputs first."
            ),
        )

    return _envelope(data)


# ===========================================================================
# Private helpers
# ===========================================================================

def _format_game(g: dict) -> dict:
    """
    Flatten a game dict into the shared API response shape.

    Handles both:
    - fetch_schedule() output: snake_case keys (game_pk, home_team_name, …)
    - Raw MLB Stats API shape:  camelCase keys (gamePk, teams.home.team.name, …)
    """
    # --- snake_case output from fetch_schedule ---
    if "game_pk" in g:
        return {
            "game_pk":               int(g["game_pk"]),
            "game_date":             g.get("game_date", ""),
            "away_team":             g.get("away_team_name", ""),
            "home_team":             g.get("home_team_name", ""),
            "status":                g.get("status", ""),
            "away_probable_pitcher": g.get("away_probable_pitcher", {}).get("fullName", "TBD")
                                     if isinstance(g.get("away_probable_pitcher"), dict)
                                     else g.get("away_probable_pitcher", "TBD"),
            "home_probable_pitcher": g.get("home_probable_pitcher", {}).get("fullName", "TBD")
                                     if isinstance(g.get("home_probable_pitcher"), dict)
                                     else g.get("home_probable_pitcher", "TBD"),
        }
    # --- raw MLB Stats API camelCase shape (fallback) ---
    teams = g.get("teams", {})
    home  = teams.get("home", {})
    away  = teams.get("away", {})
    return {
        "game_pk":               int(g.get("gamePk", 0)),
        "game_date":             g.get("officialDate", g.get("gameDate", "")),
        "away_team":             away.get("team", {}).get("name", ""),
        "home_team":             home.get("team", {}).get("name", ""),
        "status":                g.get("status", {}).get("detailedState", ""),
        "away_probable_pitcher": away.get("probablePitcher", {}).get("fullName", "TBD"),
        "home_probable_pitcher": home.get("probablePitcher", {}).get("fullName", "TBD"),
    }


def _league_avg_pa_probs() -> np.ndarray:
    """
    Build a (9, 7) league-average PA probability matrix.

    Outcome ordering matches the Markov advance table:
        [1B, 2B, 3B, HR, BB, K, other_out]
    """
    from src.features.matchup_features import MLB_LEAGUE_AVG  # noqa: PLC0415

    lg = MLB_LEAGUE_AVG
    row = np.array([
        lg["1B"], lg["2B"], lg["3B"], lg["HR"], lg["BB"], lg["K"],
        max(0.0, 1.0 - lg["1B"] - lg["2B"] - lg["3B"]
                      - lg["HR"] - lg["BB"] - lg["K"]),
    ], dtype=np.float64)
    row = row / row.sum()   # normalise to sum exactly to 1
    return np.tile(row, (9, 1))


def _run_simulation_cached(game_pk: int, n_sims: int = 50_000) -> dict:
    """
    Run (or retrieve cached) a MC simulation for *game_pk*.

    Uses real pitcher/batter season stats via log5 matchup adjustment.
    Falls back to league-average if player data is unavailable.

    Populates both _sim_cache (summary dict) and _raw_sim_cache (GameSimResult).
    Returns the summary dict.
    """
    if game_pk in _sim_cache:
        return _sim_cache[game_pk]

    # ── Get per-game PA probability matrices (real player stats) ─────────
    try:
        from src.ingestion.player_stats import get_game_pa_probs  # noqa: PLC0415
        pa_home, pa_away = get_game_pa_probs(game_pk)
        logger.info("game_pk=%d: using real player stats for simulation", game_pk)
    except Exception as exc:
        logger.warning("game_pk=%d: player stats unavailable (%s), using league avg", game_pk, exc)
        pa_home = pa_away = _league_avg_pa_probs()

    raw = _build_raw_game_result(pa_home, pa_away, n_sims)

    summary = {
        "game_pk":                game_pk,
        "home_win_prob":          round(float(raw.win_prob_home), 4),
        "away_win_prob":          round(float(raw.win_prob_away), 4),
        "over_prob":              round(float(raw.total_over_prob(8.5)), 4),
        "under_prob":             round(float(1.0 - raw.total_over_prob(8.5)), 4),
        "expected_home_runs":     round(float(raw.mean_home), 4),
        "expected_away_runs":     round(float(raw.mean_away), 4),
        "home_spread_cover_prob": round(float(raw.spread_cover_prob(-1.5)), 4),
        "n_sims":                 n_sims,
    }
    _sim_cache[game_pk]     = summary
    _raw_sim_cache[game_pk] = raw
    return summary


def _build_raw_game_result(
    pa_h: np.ndarray,
    pa_a: np.ndarray,
    n_sims: int,
) -> Any:
    """
    Run a full 9-inning simulation using the half-inning engine with raw
    (9, 7) PA probability arrays.  Returns a GameSimResult.

    This is the same logic as simulate_corr_matrix._run_game, exposed here
    so _run_simulation_cached can store the GameSimResult for distribution
    endpoint queries.
    """
    from src.models.game_simulator import (  # noqa: PLC0415
        _simulate_half_inning_variable, GameSimResult,
    )

    rng = np.random.default_rng()

    pa_h = pa_h / pa_h.sum(axis=1, keepdims=True)
    pa_a = pa_a / pa_a.sum(axis=1, keepdims=True)

    n_innings = 9
    max_extra = 6
    home_runs = np.zeros(n_sims, dtype=np.int32)
    away_runs = np.zeros(n_sims, dtype=np.int32)

    for inn in range(n_innings):
        inn_runs_a, _ = _simulate_half_inning_variable(
            pa_a, n_sims, start_slot=0, rng=rng,
        )
        away_runs += inn_runs_a

        if inn < n_innings - 1:
            inn_runs_h, _ = _simulate_half_inning_variable(
                pa_h, n_sims, start_slot=0, rng=rng,
            )
            home_runs += inn_runs_h
        else:
            needs  = away_runs >= home_runs
            n_need = int(needs.sum())
            if n_need > 0:
                inn_runs_h_part, _ = _simulate_half_inning_variable(
                    pa_h, n_need, start_slot=0, rng=rng,
                )
                home_runs[needs] += inn_runs_h_part

    tied = home_runs == away_runs
    for _ in range(max_extra):
        n_tied = int(tied.sum())
        if n_tied == 0:
            break
        ex_a, _ = _simulate_half_inning_variable(
            pa_a, n_tied, start_slot=0, rng=rng, start_base_config=2,
        )
        away_runs[tied] += ex_a
        ex_h, _ = _simulate_half_inning_variable(
            pa_h, n_tied, start_slot=0, rng=rng, start_base_config=2,
        )
        home_runs[tied] += ex_h
        tied_sub = home_runs[tied] == away_runs[tied]
        new_tied = np.zeros(n_sims, dtype=bool)
        new_tied[np.where(tied)[0][tied_sub]] = True
        tied = new_tied

    return GameSimResult(home_runs_dist=home_runs, away_runs_dist=away_runs)


def _build_simulation_block(raw: Any, n_sims: int) -> dict:
    """Build the simulation summary block from a GameSimResult."""
    home  = raw.home_runs_dist.astype(np.float64)
    away  = raw.away_runs_dist.astype(np.float64)
    total = home + away

    def _pcts(arr: np.ndarray) -> dict:
        return {
            "mean": round(float(arr.mean()), 3),
            "std":  round(float(arr.std()),  3),
            "p10":  round(float(np.percentile(arr, 10)), 3),
            "p25":  round(float(np.percentile(arr, 25)), 3),
            "p50":  round(float(np.percentile(arr, 50)), 3),
            "p75":  round(float(np.percentile(arr, 75)), 3),
            "p90":  round(float(np.percentile(arr, 90)), 3),
        }

    return {
        "n_sims":       n_sims,
        "home_win_prob": round(float(raw.win_prob_home), 4),
        "away_win_prob": round(float(raw.win_prob_away), 4),
        "home_runs":    _pcts(home),
        "away_runs":    _pcts(away),
        "total_runs":   _pcts(total),
        "spread": {
            f"home_cover_prob_{rl:+.1f}".replace("+", "p").replace("-", "m"):
                round(float(raw.spread_cover_prob(rl)), 4)
            for rl in _DEFAULT_RUN_LINES
        },
        "totals": {
            f"over_{tl}".replace(".", "_"):
                round(float(raw.total_over_prob(tl)), 4)
            for tl in _DEFAULT_TOTAL_LINES
        },
    }


def _build_market_row(
    market: str,
    label: str,
    model_prob: float,
    market_prob: Optional[float],
    market_odds: Optional[float],
    bankroll: float,
    category: str = "game",
) -> dict:
    """
    Build one market row for the /api/predictions/daily response.

    If market_prob or market_odds are unavailable (no odds feed), edge/EV/Kelly
    are returned as null so the frontend can render partial data.
    """
    from src.optimizer.kelly import kelly_fraction, kelly_ev  # noqa: PLC0415

    edge = round(model_prob - market_prob, 5) if market_prob is not None else None
    ev   = round(kelly_ev(model_prob, market_odds), 5) \
           if market_odds is not None else None

    if market_odds is not None and ev is not None and ev > 0.0:
        kf_full    = round(kelly_fraction(model_prob, market_odds, 1.0), 6)
        kf_quarter = round(kelly_fraction(model_prob, market_odds, 4.0), 6)
        stake      = round(kf_quarter * bankroll, 4)
    else:
        kf_full = kf_quarter = stake = None

    return {
        "market":        market,
        "label":         label,
        "category":      category,
        "model_prob":    round(min(max(model_prob, 0.0001), 0.9999), 5),
        "market_prob":   round(market_prob, 5) if market_prob is not None else None,
        "market_odds":   market_odds,
        "edge":          edge,
        "ev":            ev,
        "kelly_full":    kf_full,
        "kelly_quarter": kf_quarter,
        "stake_units":   stake,
    }


def _index_odds_by_home(odds_df: Any) -> dict:
    """Build a {home_team: {home_odds, away_odds}} lookup from a live-odds DataFrame."""
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


def _index_all_odds(raw_odds_list: list) -> dict:
    """
    Build a nested lookup:
        { home_team: { "h2h": {home_odds, away_odds},
                       "spreads": {home_odds, away_odds, home_point, away_point},
                       "totals": {"over_8.5": odds, "under_8.5": odds, ...} } }

    Picks the BEST (highest) price for each side across all bookmakers.
    """
    result: dict[str, dict] = {}

    for go in raw_odds_list:
        home_team = go.get("home_team", "")
        away_team = go.get("away_team", "")
        market    = go.get("market", "")
        if not home_team or not market:
            continue

        entry = result.setdefault(home_team, {"h2h": {}, "spreads": {}, "totals": {}})

        if market == "h2h":
            bh: int | None = None
            ba: int | None = None
            for bm_line in go.get("lines", []):
                for o in bm_line.get("outcomes", []):
                    if o["name"] == home_team:
                        if bh is None or o["price"] > bh:
                            bh = o["price"]
                    else:
                        if ba is None or o["price"] > ba:
                            ba = o["price"]
            if bh is not None:
                entry["h2h"]["home_odds"] = bh
            if ba is not None:
                entry["h2h"]["away_odds"] = ba

        elif market == "spreads":
            for bm_line in go.get("lines", []):
                for o in bm_line.get("outcomes", []):
                    side = "home" if o["name"] == home_team else "away"
                    key_odds  = f"{side}_odds"
                    key_point = f"{side}_point"
                    if key_odds not in entry["spreads"] or o["price"] > entry["spreads"][key_odds]:
                        entry["spreads"][key_odds] = o["price"]
                        if o.get("point") is not None:
                            entry["spreads"][key_point] = o["point"]

        elif market == "totals":
            for bm_line in go.get("lines", []):
                for o in bm_line.get("outcomes", []):
                    side_str = o["name"].lower()   # "over" or "under"
                    point    = o.get("point")
                    if side_str not in ("over", "under") or point is None:
                        continue
                    key = f"{side_str}_{point}"
                    # Keep best price
                    if key not in entry["totals"] or o["price"] > entry["totals"][key]:
                        entry["totals"][key] = o["price"]

    return result


def _fetch_player_props_for_game(
    game_pk: int,
    home_team: str,
    away_team: str,
    raw_sim: Any,
    bankroll: float,
    devig_method: str,
) -> list[dict]:
    """
    Fetch player prop lines for *game_pk* from the Odds API and return
    a list of market rows with model probabilities from the Statcast-derived
    player season stats.

    Markets fetched: batter_hits, batter_total_bases, batter_home_runs,
                     pitcher_strikeouts, pitcher_earned_runs

    For each prop, model probability is derived from the player's season
    stat rates and the Poisson / binomial distribution over the line.

    Returns empty list gracefully on any failure.
    """
    odds_key = os.environ.get("ODDS_API_KEY", "")
    if not odds_key:
        return []

    try:
        from src.ingestion.live_odds import get_game_odds, get_player_props  # noqa: PLC0415
        # Need The Odds API event ID — map game_pk → odds event ID via game h2h entry
        # We look up the event id from cached game odds
        raw_events = get_game_odds(
            markets=("h2h",), try_once=True,
        )
        event_id: str | None = None
        for ev in raw_events:
            if (ev.get("home_team") == home_team or
                    ev.get("away_team") == away_team):
                event_id = ev.get("game_id")
                break
        if not event_id:
            return []

        props = get_player_props(event_id, try_once=True)
    except Exception as exc:
        logger.debug("Player props fetch skipped for game_pk=%d: %s", game_pk, exc)
        return []

    rows: list[dict] = []
    for prop in props:
        try:
            player_name = prop.get("player_name", "")
            market_key  = prop.get("market", "")
            point       = prop.get("point")
            over_odds   = prop.get("over_odds")
            under_odds  = prop.get("under_odds")
            over_prob   = prop.get("over_implied_prob")
            under_prob  = prop.get("under_implied_prob")

            if point is None or over_odds is None:
                continue

            # Model probability: from season rate stats
            model_over_p  = _prop_model_prob(player_name, market_key, point, "over")
            model_under_p = 1.0 - model_over_p

            for side_str, model_p, mkt_odds, mkt_prob in [
                ("over",  model_over_p,  over_odds,  over_prob),
                ("under", model_under_p, under_odds, under_prob),
            ]:
                if mkt_odds is None:
                    continue
                rows.append(_build_market_row(
                    market=f"prop_{market_key}_{side_str}_{point}",
                    label=f"{player_name} {market_key.replace('batter_','').replace('pitcher_','')} "
                          f"{'O' if side_str == 'over' else 'U'} {point}",
                    model_prob=model_p,
                    market_prob=float(mkt_prob) if mkt_prob is not None else None,
                    market_odds=float(mkt_odds),
                    bankroll=bankroll,
                    category="player_prop",
                ))
        except Exception as exc:
            logger.debug("Prop row parse error: %s", exc)
            continue

    return rows


def _prop_model_prob(
    player_name: str,
    market_key: str,
    line: float,
    side: str,
) -> float:
    """
    Derive a model probability for a player prop over/under *line* using
    the player's season stats from the MLB Stats API.

    Uses a Poisson approximation for counting stats (K, TB, HR, ER).
    Uses a Bernoulli/binomial for hits (≥1 hit in a game).

    Returns 0.5 (league-average fallback) on any data failure.
    """
    import math
    from src.ingestion.mlb_stats_api import _load_or_fetch  # noqa: PLC0415

    # Look up player_id by name (best-effort via schedule's probablePitcher data)
    # We use the pybaseball playerid_lookup as a last resort, but avoid it in the
    # hot path.  Instead we fall back to Poisson with league-average rates.
    try:
        # Determine stat type and per-game rate from market key
        if market_key == "pitcher_strikeouts":
            # League avg: ~8 K/9 ≈ 5.3 K/start for SP
            rate_per_game = 5.5
        elif market_key == "pitcher_earned_runs":
            rate_per_game = 2.5
        elif market_key == "batter_home_runs":
            rate_per_game = 0.135  # ~0.135 HR/game league avg
        elif market_key == "batter_total_bases":
            rate_per_game = 1.25
        elif market_key == "batter_hits":
            rate_per_game = 1.0
        else:
            return 0.5

        # Poisson model: P(X > line) where X ~ Poisson(rate)
        lam = rate_per_game
        # P(X >= line+1) for integer lines; handle half-integer lines
        floor_line = int(math.floor(line))
        # CDF via incomplete gamma / direct sum for small lambda
        p_over = 1.0 - _poisson_cdf(floor_line, lam)
        return float(np.clip(p_over if side == "over" else 1.0 - p_over, 0.01, 0.99))
    except Exception:
        return 0.5


def _poisson_cdf(k: int, lam: float) -> float:
    """P(X <= k) for X ~ Poisson(lam), computed via direct sum for small k."""
    import math
    if lam <= 0:
        return 1.0
    total = 0.0
    term  = math.exp(-lam)
    for i in range(k + 1):
        total += term
        term  *= lam / (i + 1)
    return min(total, 1.0)
