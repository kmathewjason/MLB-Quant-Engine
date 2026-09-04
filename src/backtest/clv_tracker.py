"""
backtest.clv_tracker
====================
Closing-line value (CLV) tracker — the gold-standard metric for validating
a betting edge independent of short-term result variance.

Theory
------
CLV measures whether the model consistently finds lines that are more
accurate than the market's final (closing) consensus.  The closing line
is the best publicly available probability estimate before the event; a
bettor who consistently beats it is obtaining genuine edge, not variance.

Per-bet CLV (probability space)
---------------------------------
Both the bet price and the closing price are de-vigged (power method by
default) before comparison.  CLV is expressed in probability space:

    clv_prob = p_fair_close - p_fair_open

where:
    p_fair_open  = de-vigged probability at the price we took
    p_fair_close = de-vigged closing probability for the same side

Positive CLV → we were smarter than the market at bet-placement time.

CLV in log-odds space (scale-invariant)
-----------------------------------------
    clv_log_odds = logit(p_fair_close) - logit(p_fair_open)

This is scale-invariant (compares equal probability movements at 10% vs 50%
the same way), which is statistically preferred for aggregation.

Summary statistics
------------------
- mean_clv_prob     : average probability-space CLV across all bets
- mean_clv_log_odds : average log-odds CLV (primary metric)
- pct_positive_clv  : fraction of bets with CLV > 0
- clv_tstat         : one-sample t-statistic (H0: mean CLV = 0)
- clv_pvalue        : two-tailed p-value
- clv_result_corr   : Pearson r between per-bet CLV and bet result (+1 / 0)
  → positive correlation distinguishes real edge from variance

Public API
----------
BetRecord                                  — dataclass (one placed bet)
CLVResult                                  — dataclass (per-bet output)
compute_clv(bets, devig_method)           -> pd.DataFrame
clv_summary(clv_df)                       -> dict
rolling_clv(clv_df, window)              -> pd.Series
plot_data_clv(clv_df)                    -> dict  (ready for charting)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BetRecord:
    """
    A single placed bet for CLV tracking.

    Fields
    ------
    bet_id          : unique identifier (game_pk + market + side)
    bet_date        : ISO date string when bet was placed
    market          : 'h2h', 'spreads', 'totals', or prop market name
    side            : 'home', 'away', 'over', 'under', or player name
    open_odds       : American odds at time of bet placement (our side)
    open_odds_other : American odds for the opposing side at bet time
    close_odds      : American closing odds (our side)
    close_odds_other: American closing odds (opposing side)
    result          : 1 = win, 0 = loss, 0.5 = push/void
    stake           : stake in units (default 1.0)
    model_prob      : model's predicted probability for this side
    """
    bet_id: str
    bet_date: str
    market: str
    side: str
    open_odds: float
    open_odds_other: float
    close_odds: float
    close_odds_other: float
    result: float = 0.5
    stake: float = 1.0
    model_prob: float | None = None


# ---------------------------------------------------------------------------
# CLV computation
# ---------------------------------------------------------------------------

def compute_clv(
    bets: list[BetRecord] | pd.DataFrame,
    devig_method: Literal["power", "additive", "shin"] = "power",
) -> pd.DataFrame:
    """
    Compute per-bet CLV for a list of BetRecords (or a DataFrame with the
    same column names).

    De-vigs both the opening and closing two-way prices using the specified
    method, then computes CLV in probability and log-odds space.

    Parameters
    ----------
    bets          : list of BetRecord or equivalent DataFrame
    devig_method  : de-vig method passed to vig_removal.best_devig()

    Returns
    -------
    DataFrame with one row per bet and columns:
        bet_id, bet_date, market, side, open_odds, close_odds, result, stake,
        model_prob,
        p_fair_open, p_fair_close,
        clv_prob, clv_log_odds,
        pnl_units   (profit/loss in units at American odds),
        ev_units    (EV = clv_prob * stake, approximate)
    """
    from src.optimizer.vig_removal import best_devig  # noqa: PLC0415

    if isinstance(bets, pd.DataFrame):
        records = [BetRecord(**{k: row[k] for k in BetRecord.__dataclass_fields__})
                   for _, row in bets.iterrows()]
    else:
        records = list(bets)

    rows = []
    for b in records:
        try:
            # De-vig opening price
            p_open_arr = best_devig([b.open_odds, b.open_odds_other], method=devig_method)
            p_fair_open = float(p_open_arr[0])

            # De-vig closing price
            p_close_arr = best_devig([b.close_odds, b.close_odds_other], method=devig_method)
            p_fair_close = float(p_close_arr[0])

        except Exception as exc:
            # Log and skip malformed records
            import logging
            logging.getLogger(__name__).warning("CLV computation failed for %s: %s", b.bet_id, exc)
            continue

        clv_prob     = p_fair_close - p_fair_open
        clv_log_odds = _logit(p_fair_close) - _logit(p_fair_open)

        # P&L in units: American odds to payout
        if b.result == 1.0:
            pnl = _american_payout(b.open_odds) * b.stake
        elif b.result == 0.0:
            pnl = -b.stake
        else:
            pnl = 0.0   # push / void

        # EV ≈ p_fair_open * payout - (1 - p_fair_open) * stake
        payout_ratio = _american_payout(b.open_odds)
        ev_units = (p_fair_open * payout_ratio - (1.0 - p_fair_open)) * b.stake

        rows.append({
            "bet_id":         b.bet_id,
            "bet_date":       b.bet_date,
            "market":         b.market,
            "side":           b.side,
            "open_odds":      b.open_odds,
            "close_odds":     b.close_odds,
            "result":         b.result,
            "stake":          b.stake,
            "model_prob":     b.model_prob,
            "p_fair_open":    round(p_fair_open,  5),
            "p_fair_close":   round(p_fair_close, 5),
            "clv_prob":       round(clv_prob,      6),
            "clv_log_odds":   round(clv_log_odds,  6),
            "pnl_units":      round(pnl,           4),
            "ev_units":       round(ev_units,       4),
        })

    return pd.DataFrame(rows)


def _logit(p: float) -> float:
    """Numerically stable logit (log-odds)."""
    p = max(1e-9, min(1 - 1e-9, p))
    return math.log(p / (1.0 - p))


def _american_payout(american: float) -> float:
    """
    Return the profit per unit staked for a winning bet at *american* odds.
    e.g. -110 → 0.909..., +150 → 1.5
    """
    a = float(american)
    if a >= 100:
        return a / 100.0
    return 100.0 / abs(a)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def clv_summary(clv_df: pd.DataFrame) -> dict:
    """
    Compute summary statistics from a CLV DataFrame produced by compute_clv().

    Returns
    -------
    dict with keys:
        n_bets, mean_clv_prob, mean_clv_log_odds, pct_positive_clv,
        clv_tstat, clv_pvalue, clv_result_corr, roi_pct,
        mean_pnl_units, total_pnl_units
    """
    if clv_df.empty:
        return {"n_bets": 0}

    clv = clv_df["clv_log_odds"].values
    results = clv_df["result"].values
    pnl = clv_df["pnl_units"].values
    n = len(clv)

    # t-test: H0: mean CLV = 0
    tstat, pvalue = stats.ttest_1samp(clv, popmean=0.0)

    # Pearson correlation between CLV and result
    if len(np.unique(results)) > 1 and np.std(clv) > 0:
        r, p_corr = stats.pearsonr(clv, results)
    else:
        r, p_corr = float("nan"), float("nan")

    total_staked = clv_df["stake"].sum()
    roi = float(pnl.sum() / total_staked * 100.0) if total_staked > 0 else 0.0

    return {
        "n_bets":              n,
        "mean_clv_prob":       round(float(clv_df["clv_prob"].mean()), 5),
        "mean_clv_log_odds":   round(float(clv.mean()), 5),
        "std_clv_log_odds":    round(float(clv.std()),  5),
        "pct_positive_clv":    round(float((clv > 0).mean() * 100), 1),
        "clv_tstat":           round(float(tstat),  4),
        "clv_pvalue":          round(float(pvalue), 4),
        "clv_result_corr":     round(float(r), 4) if not math.isnan(r) else None,
        "roi_pct":             round(roi, 2),
        "mean_pnl_units":      round(float(pnl.mean()), 4),
        "total_pnl_units":     round(float(pnl.sum()),  4),
        "total_staked":        round(float(total_staked), 2),
    }


def rolling_clv(clv_df: pd.DataFrame, window: int = 50) -> pd.Series:
    """
    Compute rolling mean CLV (log-odds space) with *window* bets.

    Returns pd.Series indexed by bet number.
    """
    return clv_df["clv_log_odds"].rolling(window=window, min_periods=1).mean().rename("rolling_clv")


def clv_by_market(clv_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return per-market summary statistics.

    Returns DataFrame indexed by market with summary columns.
    """
    if clv_df.empty:
        return pd.DataFrame()
    rows = []
    for market, grp in clv_df.groupby("market"):
        s = clv_summary(grp)
        s["market"] = market
        rows.append(s)
    return pd.DataFrame(rows).set_index("market")


def plot_data_clv(clv_df: pd.DataFrame, window: int = 50) -> dict:
    """
    Return chart-ready dict for the CLV dashboard panel.

    Keys:
        rolling_clv   : pd.Series — rolling mean CLV by bet number
        cum_pnl       : pd.Series — cumulative P&L in units
        clv_histogram : dict with 'edges' and 'counts'
    """
    df = clv_df.copy().reset_index(drop=True)
    roll = rolling_clv(df, window=window)
    cum_pnl = df["pnl_units"].cumsum().rename("cum_pnl_units")

    hist_vals = df["clv_log_odds"].values
    counts, edges = np.histogram(hist_vals, bins=30)
    return {
        "rolling_clv":   roll,
        "cum_pnl":       cum_pnl,
        "clv_histogram": {"edges": edges.tolist(), "counts": counts.tolist()},
    }
