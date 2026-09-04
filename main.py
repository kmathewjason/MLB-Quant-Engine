"""
mlb-quant-engine — entry point
===============================

Run the FastAPI server:
    uvicorn main:app --host 127.0.0.1 --port 8000 --reload

Run the daily prediction pipeline:
    python main.py --run-pipeline [--date YYYY-MM-DD] [--bankroll 1000]

The pipeline sequence:
    1. Fetch today's schedule (MLB Stats API)
    2. Fetch live odds (The-Odds-API)
    3. Fetch Statcast data for upcoming pitchers/batters
    4. Compute features (batter, pitcher, matchup, park/weather)
    5. Run Monte Carlo game simulations
    6. De-vig market odds and compute model edge
    7. Kelly sizing + portfolio caps
    8. Write recommendations to data/predictions/recommendations_YYYY-MM-DD.csv
    9. Generate backtest HTML report (if --backtest flag is set)
"""

from __future__ import annotations

import argparse
import datetime
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("mlb_quant")


# ---------------------------------------------------------------------------
# FastAPI app (exposed at module level for uvicorn)
# ---------------------------------------------------------------------------

def _get_app():
    """Lazy import so uvicorn can import this module without loading pipeline deps."""
    from src.api import app  # noqa: PLC0415
    return app


app = _get_app()


# ---------------------------------------------------------------------------
# Daily pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    date: str | None = None,
    bankroll: float = 1000.0,
    devig_method: str = "power",
    n_sims: int = 50_000,
    output_dir: str = "data/predictions",
) -> int:
    """
    Orchestrate the full daily ingestion → simulation → optimisation pipeline.

    Parameters
    ----------
    date         : ISO date string (YYYY-MM-DD); defaults to today.
    bankroll     : current bankroll in units for Kelly sizing.
    devig_method : de-vig method passed to vig_removal.best_devig().
    n_sims       : Monte Carlo simulations per game.
    output_dir   : directory to write recommendations CSV.

    Returns
    -------
    0 on success, 1 on error.
    """
    import numpy as np
    import pandas as pd

    from src.ingestion.mlb_stats_api import fetch_schedule
    from src.optimizer.vig_removal import best_devig
    from src.optimizer.kelly import portfolio_kelly
    from src.optimizer.portfolio_cap import apply_caps
    from src.models.game_simulator import simulate_game
    from src.features.matchup_features import MLB_LEAGUE_AVG

    game_date = date or datetime.date.today().isoformat()
    out_dir   = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== MLB Quant Engine — pipeline for %s ===", game_date)

    # ── 1. Schedule ────────────────────────────────────────────────────────
    logger.info("Fetching schedule...")
    try:
        games = fetch_schedule(game_date)
    except Exception as exc:
        logger.error("Schedule fetch failed: %s", exc)
        return 1

    if not games:
        logger.info("No games scheduled for %s. Exiting.", game_date)
        return 0

    logger.info("Found %d game(s).", len(games))

    # ── 2. Simulations (league-average priors until PA model is trained) ───
    lg = MLB_LEAGUE_AVG
    other_out_p = max(
        0.0,
        1 - lg["1B"] - lg["2B"] - lg["3B"] - lg["HR"] - lg["BB"] - lg["K"],
    )
    pa_row = np.array([
        lg["1B"], lg["2B"], lg["3B"], lg["HR"], lg["BB"], lg["K"], other_out_p,
    ])
    pa_probs_lg = np.tile(pa_row, (9, 1))

    sim_results = {}
    for g in games:
        pk = int(g.get("gamePk", 0))
        if pk == 0:
            continue
        logger.info("Simulating game_pk=%d (%s @ %s)...", pk,
                    g.get("teams", {}).get("away", {}).get("team", {}).get("name", "?"),
                    g.get("teams", {}).get("home", {}).get("team", {}).get("name", "?"))
        try:
            result = simulate_game(
                pa_probs_home=pa_probs_lg,
                pa_probs_away=pa_probs_lg,
                n_sims=n_sims,
            )
            sim_results[pk] = {
                "game_pk":               pk,
                "game_date":             game_date,
                "home_team":             g.get("teams", {}).get("home", {}).get("team", {}).get("name", ""),
                "away_team":             g.get("teams", {}).get("away", {}).get("team", {}).get("name", ""),
                "home_win_prob":         round(float(result.win_prob_home), 4),
                "away_win_prob":         round(float(result.win_prob_away), 4),
                "over_prob_8_5":         round(float(result.total_over_prob(8.5)), 4),
                "expected_home_runs":    round(float(result.mean_home_runs), 3),
                "expected_away_runs":    round(float(result.mean_away_runs), 3),
                "home_spread_cover_prob": round(float(result.spread_cover_prob(-1.5)), 4),
            }
        except Exception as exc:
            logger.warning("Simulation failed for game_pk=%d: %s", pk, exc)

    if not sim_results:
        logger.error("All simulations failed. Exiting.")
        return 1

    # Write simulation results
    sims_df = pd.DataFrame(list(sim_results.values()))
    sims_path = out_dir / f"simulations_{game_date}.csv"
    sims_df.to_csv(sims_path, index=False)
    logger.info("Simulations written to %s", sims_path)

    # ── 3. Live odds (optional; skip gracefully if ODDS_API_KEY not set) ───
    import os
    odds_key = os.environ.get("ODDS_API_KEY", "")
    if not odds_key:
        logger.warning("ODDS_API_KEY not set — skipping odds fetch and bet sizing.")
        return 0

    try:
        from src.ingestion.live_odds import fetch_live_odds
        odds_df = fetch_live_odds(sport="baseball_mlb", market="h2h")
        logger.info("Fetched odds for %d games.", len(odds_df))
    except Exception as exc:
        logger.warning("Odds fetch failed (%s) — skipping bet sizing.", exc)
        return 0

    # ── 4. Build bet candidates + Kelly sizing ─────────────────────────────
    def _home_col(col: str):
        return col if col in odds_df.columns else None

    odds_by_home: dict[str, dict] = {}
    if "home_team" in odds_df.columns:
        for _, row in odds_df.iterrows():
            ht = row.get("home_team", "")
            if ht:
                odds_by_home[ht] = {
                    "home_odds": row.get("home_price"),
                    "away_odds": row.get("away_price"),
                }

    bet_rows = []
    for pk, sim in sim_results.items():
        home = sim["home_team"]
        market = odds_by_home.get(home, {})
        if not market or market.get("home_odds") is None:
            continue

        try:
            fair = best_devig([market["home_odds"], market["away_odds"]], method=devig_method)
        except Exception:
            continue

        for side_idx, side in enumerate(["home", "away"]):
            bet_rows.append({
                "game_pk":            pk,
                "bet_date":           game_date,
                "home_team":          home,
                "away_team":          sim["away_team"],
                "side":               side,
                "model_prob":         sim[f"{side}_win_prob"],
                "fair_prob":          float(fair[side_idx]),
                "fair_odds_american": float(market[f"{side}_odds"]),
                "market":             "h2h",
            })

    if not bet_rows:
        logger.info("No bet candidates found — no matching odds/simulations.")
        return 0

    bets_df   = pd.DataFrame(bet_rows)
    sized     = portfolio_kelly(bets_df, bankroll=bankroll, max_total_exposure=0.20)
    capped    = apply_caps(sized, bankroll=bankroll)

    recs = capped[capped["ev"] > 0].sort_values("ev", ascending=False)
    recs_path = out_dir / f"recommendations_{game_date}.csv"
    recs.to_csv(recs_path, index=False)
    logger.info(
        "Wrote %d recommendation(s) to %s",
        len(recs), recs_path,
    )
    logger.info("=== Pipeline complete ===")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MLB Quant Engine — server or pipeline runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-pipeline", action="store_true",
                   help="Execute the daily prediction pipeline and exit")
    p.add_argument("--date",     default=None,
                   help="ISO date for the pipeline (default: today)")
    p.add_argument("--bankroll", type=float, default=1000.0,
                   help="Bankroll in units for Kelly sizing")
    p.add_argument("--devig",    default="power",
                   choices=["power", "additive", "shin"],
                   help="De-vig method")
    p.add_argument("--n-sims",   type=int, default=50_000,
                   help="Monte Carlo simulations per game")
    p.add_argument("--output",   default="data/predictions",
                   help="Output directory for CSV results")
    p.add_argument("--port",     type=int, default=8000,
                   help="Port for the FastAPI server (--run-pipeline not set)")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()

    if args.run_pipeline:
        sys.exit(run_pipeline(
            date=args.date,
            bankroll=args.bankroll,
            devig_method=args.devig,
            n_sims=args.n_sims,
            output_dir=args.output,
        ))
    else:
        import uvicorn
        uvicorn.run(
            "main:app",
            host="127.0.0.1",
            port=args.port,
            reload=True,
        )
