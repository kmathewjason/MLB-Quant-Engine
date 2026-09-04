"""
backtest.run_dispersion_check
==============================
Compares Monte Carlo simulator run-score variance against a Negative Binomial
regression model fit to historical team-game run totals, and flags games where
the simulator is *significantly underdispersed* relative to the NegBin baseline.

Statistical justification
--------------------------
Runs-per-game (RPG) is count data with overdispersion: the variance exceeds the
mean (σ² > μ), which a Poisson model cannot capture.  The Negative Binomial (NB2
parameterisation in statsmodels) models:

    Var[Y] = μ + α·μ²

where α > 0 is the dispersion parameter estimated from data.

If the Monte Carlo simulator's sample variance over N simulations is substantially
below the NB-implied variance for the same game context, the simulator is
*underdispersed* — it is missing correlation structure in the PA outcome sequencing
(e.g., within-game momentum, bullpen transitions, lineup clustering effects).

Underdispersion test
--------------------
For each game, compute:

    dispersion_ratio = var_mc / var_nb

where:
    var_mc = empirical variance of {home_runs + away_runs} over N simulations
    var_nb = μ_nb + α̂ · μ_nb²   (predicted variance from fitted NB model)
           μ_nb = fitted NB mean for that game context

Flag the game if:
    dispersion_ratio < UNDERDISPERSION_THRESHOLD   (default 0.60)

i.e., the simulator captures less than 60% of the variance predicted by the
historical NB distribution.

Usage
-----
python -m src.backtest.run_dispersion_check \
    --game-logs data/processed/team_game_logs.parquet \
    --sim-results data/predictions/sim_results.parquet \
    --output data/predictions/dispersion_report.parquet

Or import directly:
    from src.backtest.run_dispersion_check import (
        fit_negbin, predict_negbin_variance, flag_underdispersed_games
    )
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

UNDERDISPERSION_THRESHOLD: float = 0.60
MIN_GAME_LOGS: int = 100   # minimum rows needed to fit NegBin reliably


# ---------------------------------------------------------------------------
# Negative Binomial regression
# ---------------------------------------------------------------------------

def fit_negbin(
    game_logs: pd.DataFrame,
    runs_col: str = "runs",
    feature_cols: list[str] | None = None,
) -> object:
    """
    Fit a statsmodels NegativeBinomialP (NB2) regression to team-game run totals.

    Parameters
    ----------
    game_logs    : DataFrame with one row per team-game.
                   Must contain *runs_col* and the feature columns.
    runs_col     : name of the runs column (dependent variable)
    feature_cols : list of covariate names; defaults to ["is_home", "park_factor_runs"]
                   Add more features (opponent ERA, weather, etc.) when available.

    Returns
    -------
    Fitted statsmodels NegativeBinomialP results object.
    """
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
    except ImportError as e:
        raise ImportError("statsmodels is required for dispersion analysis.") from e

    if len(game_logs) < MIN_GAME_LOGS:
        raise ValueError(
            f"Need at least {MIN_GAME_LOGS} game-log rows to fit NegBin; "
            f"got {len(game_logs)}."
        )

    if feature_cols is None:
        feature_cols = [c for c in ["is_home", "park_factor_runs"] if c in game_logs.columns]

    if not feature_cols:
        # Intercept-only model
        X = sm.add_constant(pd.DataFrame(index=game_logs.index))
    else:
        X = sm.add_constant(game_logs[feature_cols].astype(float))

    y = game_logs[runs_col].astype(float)

    model = sm.NegativeBinomial(y, X)
    result = model.fit(disp=False, maxiter=200)

    logger.info(
        "NegBin fit: α=%.4f  AIC=%.1f  n=%d",
        result.params.get("alpha", float("nan")),
        result.aic,
        len(y),
    )
    return result


def predict_negbin_variance(
    fit_result,
    game_context: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> np.ndarray:
    """
    Compute NB-predicted variance for each row of *game_context*.

    Var[Y] = μ + α·μ²   (NB2 variance function)

    Parameters
    ----------
    fit_result   : fitted NegativeBinomialP results from fit_negbin()
    game_context : DataFrame with the same feature columns used in fit_negbin()
    feature_cols : column names (same as fit_negbin); defaults to "is_home", "park_factor_runs"

    Returns
    -------
    np.ndarray of predicted variance, shape (len(game_context),)
    """
    try:
        import statsmodels.api as sm
    except ImportError as e:
        raise ImportError("statsmodels required.") from e

    if feature_cols is None:
        feature_cols = [c for c in ["is_home", "park_factor_runs"] if c in game_context.columns]

    if not feature_cols:
        X = sm.add_constant(pd.DataFrame(index=game_context.index))
    else:
        X = sm.add_constant(game_context[feature_cols].astype(float))

    # Predicted means from the linear predictor
    mu = fit_result.predict(X)

    # Retrieve the dispersion parameter alpha from the fit
    alpha = float(fit_result.params.get("alpha", 0.0))

    # NB2 variance: mu + alpha * mu^2
    var_nb = mu + alpha * mu ** 2
    return np.asarray(var_nb, dtype=np.float64)


# ---------------------------------------------------------------------------
# Dispersion comparison
# ---------------------------------------------------------------------------

def compute_mc_variance(
    sim_results: pd.DataFrame,
    game_pk_col: str = "game_pk",
    home_runs_col: str = "home_runs",
    away_runs_col: str = "away_runs",
) -> pd.DataFrame:
    """
    Aggregate per-game MC simulation results into per-game total-run variance.

    Parameters
    ----------
    sim_results  : DataFrame with one row per simulation per game.
                   Columns: game_pk, home_runs, away_runs.

    Returns
    -------
    DataFrame with columns: game_pk, mc_mean_total, mc_var_total, n_sims
    """
    sim_results = sim_results.copy()
    sim_results["total_runs"] = (
        sim_results[home_runs_col] + sim_results[away_runs_col]
    )
    agg = sim_results.groupby(game_pk_col)["total_runs"].agg(
        mc_mean_total="mean",
        mc_var_total="var",
        n_sims="count",
    ).reset_index()
    return agg


def flag_underdispersed_games(
    game_logs: pd.DataFrame,
    sim_variance_df: pd.DataFrame,
    fit_result,
    game_pk_col: str = "game_pk",
    feature_cols: list[str] | None = None,
    threshold: float = UNDERDISPERSION_THRESHOLD,
) -> pd.DataFrame:
    """
    Join NB-predicted variance with MC variance per game and flag underdispersed games.

    Parameters
    ----------
    game_logs       : one row per game, must contain game_pk + feature columns
    sim_variance_df : output of compute_mc_variance()
    fit_result      : fitted NegBin model from fit_negbin()
    threshold       : flag if dispersion_ratio < threshold

    Returns
    -------
    DataFrame with columns:
        game_pk, mc_mean_total, mc_var_total, nb_var_predicted,
        dispersion_ratio, underdispersed
    """
    merged = game_logs[[game_pk_col] + (feature_cols or [])].merge(
        sim_variance_df, on=game_pk_col, how="inner"
    )

    nb_var = predict_negbin_variance(fit_result, merged, feature_cols=feature_cols)
    merged["nb_var_predicted"] = nb_var
    merged["dispersion_ratio"] = merged["mc_var_total"] / merged["nb_var_predicted"].clip(lower=1e-6)
    merged["underdispersed"] = merged["dispersion_ratio"] < threshold

    n_flagged = int(merged["underdispersed"].sum())
    n_total   = len(merged)
    logger.warning(
        "Underdispersed games: %d / %d (%.1f%%) — "
        "threshold=%.2f.  These games may have missing correlation structure "
        "in the PA-outcome sequencing.",
        n_flagged, n_total, 100.0 * n_flagged / max(n_total, 1), threshold,
    )
    return merged


# ---------------------------------------------------------------------------
# Diagnostic summary
# ---------------------------------------------------------------------------

def dispersion_summary(flagged_df: pd.DataFrame) -> dict:
    """
    Return a dict of summary statistics for the dispersion comparison report.

    Keys: n_games, n_underdispersed, pct_underdispersed,
          mean_dispersion_ratio, median_dispersion_ratio,
          p10_dispersion_ratio, p90_dispersion_ratio
    """
    ratios = flagged_df["dispersion_ratio"].dropna()
    return {
        "n_games":               len(flagged_df),
        "n_underdispersed":      int(flagged_df["underdispersed"].sum()),
        "pct_underdispersed":    round(100.0 * flagged_df["underdispersed"].mean(), 1),
        "mean_dispersion_ratio": round(float(ratios.mean()), 4),
        "median_dispersion_ratio": round(float(ratios.median()), 4),
        "p10_dispersion_ratio":  round(float(ratios.quantile(0.10)), 4),
        "p90_dispersion_ratio":  round(float(ratios.quantile(0.90)), 4),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compare MC simulator variance to Negative Binomial baseline."
    )
    p.add_argument("--game-logs",    required=True, help="Parquet: team-game logs")
    p.add_argument("--sim-results",  required=True, help="Parquet: simulation results")
    p.add_argument("--output",       default="data/predictions/dispersion_report.parquet")
    p.add_argument("--threshold",    type=float, default=UNDERDISPERSION_THRESHOLD)
    p.add_argument("--runs-col",     default="runs")
    p.add_argument("--game-pk-col",  default="game_pk")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_parser().parse_args(argv)

    game_logs   = pd.read_parquet(args.game_logs)
    sim_results = pd.read_parquet(args.sim_results)

    fit = fit_negbin(game_logs, runs_col=args.runs_col)
    mc_var = compute_mc_variance(sim_results, game_pk_col=args.game_pk_col)

    flagged = flag_underdispersed_games(
        game_logs, mc_var, fit,
        game_pk_col=args.game_pk_col,
        threshold=args.threshold,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    flagged.to_parquet(out_path, index=False)

    summary = dispersion_summary(flagged)
    for k, v in summary.items():
        print(f"  {k}: {v}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
