"""
optimizer.portfolio_cap
=======================
Hard caps and risk guardrails applied after Kelly sizing.

After Kelly computes the theoretically optimal stake fractions, this module
enforces practical risk limits before an order is submitted.

Guardrails implemented
----------------------
1. Per-bet cap              : no single bet exceeds *max_bet_fraction* of bankroll
2. Same-team correlated cap : sum of exposure to any single team on any given
                              day ≤ *max_team_fraction* (avoids doubling up on
                              correlated game-level risk)
3. Daily total risk cap     : total staked across all games on one calendar day
                              ≤ *daily_cap_fraction* of bankroll
4. Drawdown circuit-breaker : if the rolling bankroll has declined by
                              *drawdown_halt_pct*% from its high-water mark,
                              no new bets are opened (return all stakes as 0)

Scaling behaviour
-----------------
Caps are applied sequentially.  When a cap triggers, the affected bets are
scaled DOWN proportionally (not hard-zeroed) so relative sizes are preserved
within the cap.  This avoids discontinuous jumps in portfolio composition.

Public API
----------
CapConfig                              — dataclass (all cap parameters)
apply_caps(sized_bets_df, bankroll,
           config, current_hwm)        -> pd.DataFrame
           (adds 'capped_stake_units' and 'cap_applied' columns)
check_drawdown_halt(bankroll,
                    high_water_mark,
                    config)            -> bool
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CapConfig:
    """
    Risk-cap parameters for the portfolio guardrails.

    Attributes
    ----------
    max_bet_fraction     : maximum stake as fraction of bankroll for any single
                           bet (default 0.03 = 3%).
    max_team_fraction    : maximum total exposure to any one team across all
                           concurrent bets (default 0.06 = 6%).
    daily_cap_fraction   : maximum total daily exposure as fraction of bankroll
                           (default 0.15 = 15%).
    drawdown_halt_pct    : halt new bets if bankroll has fallen this many percent
                           below its high-water mark (default 20.0 = 20%).
    min_ev_threshold     : skip bets with EV (per-unit) below this floor
                           (default 0.0 — only positive-EV bets).
    """
    max_bet_fraction:   float = 0.03
    max_team_fraction:  float = 0.06
    daily_cap_fraction: float = 0.15
    drawdown_halt_pct:  float = 20.0
    min_ev_threshold:   float = 0.0


# ---------------------------------------------------------------------------
# Drawdown circuit-breaker
# ---------------------------------------------------------------------------

def check_drawdown_halt(
    bankroll: float,
    high_water_mark: float,
    config: CapConfig | None = None,
) -> bool:
    """
    Return True if betting should be halted due to drawdown.

    Parameters
    ----------
    bankroll        : current bankroll value
    high_water_mark : highest bankroll value ever recorded
    config          : CapConfig; defaults used if None

    Returns
    -------
    bool — True means HALT (do not place any bets).
    """
    if config is None:
        config = CapConfig()
    if high_water_mark <= 0.0:
        return False
    drawdown_pct = (1.0 - bankroll / high_water_mark) * 100.0
    if drawdown_pct >= config.drawdown_halt_pct:
        logger.warning(
            "Drawdown circuit-breaker TRIGGERED: %.1f%% drawdown from HWM %.2f "
            "(current %.2f).  No new bets will be sized.",
            drawdown_pct, high_water_mark, bankroll,
        )
        return True
    return False


# ---------------------------------------------------------------------------
# Main cap application
# ---------------------------------------------------------------------------

def apply_caps(
    sized_bets_df: pd.DataFrame,
    bankroll: float = 1.0,
    config: CapConfig | None = None,
    current_hwm: float | None = None,
) -> pd.DataFrame:
    """
    Apply sequential risk caps to Kelly-sized bets.

    Input DataFrame must have columns:
        stake_units  : float — stake in bankroll units from portfolio_kelly()
        ev           : float — expected value per unit (optional, used for EV filter)

    Optional columns used for team-level caps:
        home_team    : str — home team identifier
        away_team    : str — away team identifier
        side         : str — 'home' or 'away' (indicates which team is bet on)
        bet_date     : str — ISO date (used for daily cap)

    Parameters
    ----------
    sized_bets_df : output of portfolio_kelly(), one row per bet
    bankroll      : current bankroll in currency units
    config        : CapConfig; defaults used if None
    current_hwm   : current high-water-mark bankroll; if provided and drawdown
                    circuit-breaker triggers, all stakes are zeroed.

    Returns
    -------
    DataFrame with additional columns:
        capped_stake_units : final stake after all caps applied
        cap_applied        : str description of which cap(s) triggered
    """
    if config is None:
        config = CapConfig()

    df = sized_bets_df.copy()
    n  = len(df)

    if n == 0:
        df["capped_stake_units"] = pd.Series(dtype=float)
        df["cap_applied"]        = pd.Series(dtype=str)
        return df

    stakes      = df["stake_units"].to_numpy(dtype=np.float64).copy()
    cap_applied = [""] * n

    # ── Drawdown circuit-breaker ───────────────────────────────────────────
    if current_hwm is not None and check_drawdown_halt(bankroll, current_hwm, config):
        df["capped_stake_units"] = 0.0
        df["cap_applied"]        = "drawdown_halt"
        return df

    # ── EV filter ─────────────────────────────────────────────────────────
    if "ev" in df.columns:
        ev = df["ev"].to_numpy(dtype=np.float64)
        below_ev = ev < config.min_ev_threshold
        stakes[below_ev] = 0.0
        for i in np.where(below_ev)[0]:
            cap_applied[i] = "ev_filter"

    # ── Per-bet cap ────────────────────────────────────────────────────────
    max_stake = config.max_bet_fraction * bankroll
    over_bet  = stakes > max_stake
    if over_bet.any():
        for i in np.where(over_bet)[0]:
            cap_applied[i] = _append_cap(cap_applied[i], "per_bet_cap")
        stakes = np.minimum(stakes, max_stake)

    # ── Same-team correlated cap ───────────────────────────────────────────
    if "side" in df.columns and ("home_team" in df.columns or "away_team" in df.columns):
        bet_teams = _infer_bet_team(df)
        max_team_stake = config.max_team_fraction * bankroll
        unique_teams = {t for t in bet_teams if t is not None}

        for team in unique_teams:
            team_mask = np.array([t == team for t in bet_teams])
            team_total = stakes[team_mask].sum()
            if team_total > max_team_stake:
                scale = max_team_stake / team_total
                stakes[team_mask] *= scale
                for i in np.where(team_mask)[0]:
                    cap_applied[i] = _append_cap(cap_applied[i], "team_cap")

    # ── Daily total risk cap ───────────────────────────────────────────────
    if "bet_date" in df.columns:
        max_daily = config.daily_cap_fraction * bankroll
        for date_val, grp_idx in df.groupby("bet_date").groups.items():
            idx_arr = np.array(grp_idx)
            day_total = stakes[idx_arr].sum()
            if day_total > max_daily:
                scale = max_daily / day_total
                stakes[idx_arr] *= scale
                for i in idx_arr:
                    cap_applied[i] = _append_cap(cap_applied[i], "daily_cap")
    else:
        # No date column: apply daily cap to the whole batch as one day
        max_daily  = config.daily_cap_fraction * bankroll
        batch_total = stakes.sum()
        if batch_total > max_daily:
            scale   = max_daily / batch_total
            stakes *= scale
            cap_applied = [_append_cap(c, "daily_cap") for c in cap_applied]

    df["capped_stake_units"] = np.round(stakes, 6)
    df["cap_applied"]        = [c if c else "none" for c in cap_applied]

    total_after = stakes.sum()
    logger.info(
        "apply_caps: %d bets, total staked=%.4f (%.2f%% of bankroll=%.2f)",
        n, total_after, 100.0 * total_after / max(bankroll, 1e-9), bankroll,
    )

    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _infer_bet_team(df: pd.DataFrame) -> list[str | None]:
    """
    Return the list of teams being bet on (one per row).

    Uses 'side', 'home_team', 'away_team' columns.
    Falls back to None if columns are missing.
    """
    result: list[str | None] = []
    for _, row in df.iterrows():
        side = str(row.get("side", "")).lower()
        if side == "home" and "home_team" in df.columns:
            result.append(str(row["home_team"]))
        elif side == "away" and "away_team" in df.columns:
            result.append(str(row["away_team"]))
        else:
            result.append(None)
    return result


def _append_cap(existing: str, new_cap: str) -> str:
    """Append a cap label, pipe-separated."""
    if not existing:
        return new_cap
    if new_cap in existing:
        return existing
    return f"{existing}|{new_cap}"


# ---------------------------------------------------------------------------
# Bootstrap drawdown simulator
# ---------------------------------------------------------------------------

from dataclasses import dataclass as _dataclass

@_dataclass
class DrawdownSimResult:
    """
    Output of bootstrap_drawdown_simulator() for one Kelly fraction.

    Attributes
    ----------
    kelly_divisor      : fractional-Kelly divisor used (1=full, 2=half, etc.)
    kelly_label        : human-readable label, e.g. "full", "half"
    p95_max_drawdown   : 95th-percentile maximum drawdown across bootstrap paths
    p50_max_drawdown   : median maximum drawdown
    p05_max_drawdown   : 5th-percentile maximum drawdown (optimistic tail)
    mean_max_drawdown  : mean maximum drawdown across paths
    p95_terminal_bankroll : 95th-percentile terminal bankroll (pessimistic tail)
    p50_terminal_bankroll : median terminal bankroll
    mean_terminal_bankroll: mean terminal bankroll
    n_bets             : number of bets per bootstrap path
    n_paths            : number of bootstrap resampled paths
    """
    kelly_divisor:           float
    kelly_label:             str
    p95_max_drawdown:        float   # worst-5% max drawdown fraction (0–1)
    p50_max_drawdown:        float
    p05_max_drawdown:        float
    mean_max_drawdown:       float
    p95_terminal_bankroll:   float   # 5th-percentile terminal bankroll (normalised to 1.0 start)
    p50_terminal_bankroll:   float
    mean_terminal_bankroll:  float
    n_bets:                  int
    n_paths:                 int

    def to_dict(self) -> dict:
        return {
            "kelly_divisor":            self.kelly_divisor,
            "kelly_label":              self.kelly_label,
            "p95_max_drawdown":         round(self.p95_max_drawdown,       4),
            "p50_max_drawdown":         round(self.p50_max_drawdown,       4),
            "p05_max_drawdown":         round(self.p05_max_drawdown,       4),
            "mean_max_drawdown":        round(self.mean_max_drawdown,      4),
            "p95_terminal_bankroll":    round(self.p95_terminal_bankroll,  4),
            "p50_terminal_bankroll":    round(self.p50_terminal_bankroll,  4),
            "mean_terminal_bankroll":   round(self.mean_terminal_bankroll, 4),
            "n_bets":                   self.n_bets,
            "n_paths":                  self.n_paths,
        }


def bootstrap_drawdown_simulator(
    bet_log: "pd.DataFrame",
    n_paths: int = 2_000,
    kelly_divisors: "list[float] | None" = None,
    kelly_labels: "list[str] | None" = None,
    starting_bankroll: float = 1.0,
    rng: "np.random.Generator | None" = None,
) -> "list[DrawdownSimResult]":
    """
    Simulate the bankroll path for multiple Kelly fractions using bootstrap
    resampling of a historical or simulated bet log, and report the
    95th-percentile maximum drawdown at each fraction.

    This lets you pick a Kelly divisor based on an actual risk target instead
    of a fixed convention (e.g. "I accept at most 20% drawdown 95% of the time,
    so I should use the fraction whose p95 drawdown ≤ 0.20").

    Theory
    ------
    The maximum drawdown of a bankroll path B(t) is:

        MDD = max_{0 ≤ s ≤ t} [HWM(s) - B(t)] / HWM(s)

    where HWM(s) = max_{0 ≤ u ≤ s} B(u).

    For fractional Kelly with divisor d, the stake on bet i is:

        stake_i(d) = f*_i / d  (or f_star column from portfolio_kelly)

    We rescale the original stake column by (original_divisor / d) to simulate
    different fractions without re-running the Kelly solver.

    Bootstrap procedure
    -------------------
    1. Draw n_paths bootstrap samples of the bet log (with replacement,
       preserving the original bet-count N).
    2. For each path, simulate the bankroll sequence using the rescaled stakes
       and the bet outcomes (result column).
    3. Record the maximum drawdown fraction for that path.
    4. Report percentiles across the n_paths values.

    Parameters
    ----------
    bet_log : DataFrame with (at minimum) columns:
                  - result         : float, 1 = win, 0 = loss, 0.5 = push
                  - stake_units    : float, stake as fraction of bankroll
                    (from portfolio_kelly() output)
                  - f_kelly        : float, fractional Kelly stake fraction
                    (used to back-out the full-Kelly f* = f_kelly * divisor_used;
                     if this column is absent, stake_units is used directly and
                     all rescaling is skipped — i.e. results are for the single
                     fraction recorded)
                  Optional:
                  - fair_odds_american : float (used to compute payout)
                  - pnl_units      : float (if present, overrides computed P&L)

    n_paths  : number of bootstrap resampled paths (default 2,000)

    kelly_divisors : list of divisors to test (default [1, 2, 4, 8] = full,
                     half, quarter, eighth Kelly).

    kelly_labels   : human-readable labels for each divisor.  Defaults to
                     ["full", "half", "quarter", "eighth"].

    starting_bankroll : normalised starting bankroll (default 1.0).

    rng : numpy random Generator; seeded internally if None.

    Returns
    -------
    List of DrawdownSimResult, one per divisor, in the same order as
    kelly_divisors.  Results are sorted by kelly_divisor ascending.

    Example
    -------
    >>> results = bootstrap_drawdown_simulator(clv_df, n_paths=5000)
    >>> for r in results:
    ...     print(f"{r.kelly_label}: p95 MDD = {r.p95_max_drawdown:.1%},
    ...            p50 terminal = {r.p50_terminal_bankroll:.2f}")
    quarter: p95 MDD = 18.3%, p50 terminal = 1.41
    """
    if kelly_divisors is None:
        kelly_divisors = [1.0, 2.0, 4.0, 8.0]
    if kelly_labels is None:
        kelly_labels = ["full", "half", "quarter", "eighth"]
    if len(kelly_labels) != len(kelly_divisors):
        raise ValueError(
            f"kelly_labels length {len(kelly_labels)} must match "
            f"kelly_divisors length {len(kelly_divisors)}."
        )
    if rng is None:
        rng = np.random.default_rng(42)

    if bet_log.empty:
        return [
            DrawdownSimResult(
                kelly_divisor=d, kelly_label=lab,
                p95_max_drawdown=0.0, p50_max_drawdown=0.0,
                p05_max_drawdown=0.0, mean_max_drawdown=0.0,
                p95_terminal_bankroll=starting_bankroll,
                p50_terminal_bankroll=starting_bankroll,
                mean_terminal_bankroll=starting_bankroll,
                n_bets=0, n_paths=n_paths,
            )
            for d, lab in zip(kelly_divisors, kelly_labels)
        ]

    n_bets   = len(bet_log)
    results_list = []

    # ── Pre-compute the P&L-per-unit vector (independent of Kelly fraction) ─
    # pnl_unit[i] = profit/loss in bankroll units per unit staked on bet i
    if "pnl_units" in bet_log.columns and "stake_units" in bet_log.columns:
        # If we already have pnl_units from CLV tracker, back out the per-unit P&L
        stake_arr = bet_log["stake_units"].to_numpy(dtype=np.float64)
        pnl_arr   = bet_log["pnl_units"].to_numpy(dtype=np.float64)
        # pnl_per_unit[i] = pnl_arr[i] / stake_arr[i] where stake > 0
        safe_stake = np.where(stake_arr > 1e-12, stake_arr, 1.0)
        pnl_per_unit = np.where(stake_arr > 1e-12, pnl_arr / safe_stake, 0.0)
    elif "result" in bet_log.columns:
        result_arr = bet_log["result"].to_numpy(dtype=np.float64)
        if "fair_odds_american" in bet_log.columns:
            from src.optimizer.kelly import american_to_decimal as _a2d  # noqa: PLC0415
            odds_arr = bet_log["fair_odds_american"].to_numpy(dtype=np.float64)
            payout   = np.array([_a2d(o) - 1.0 for o in odds_arr])
        else:
            payout = np.ones(n_bets)   # assume even-money if no odds
        # pnl_per_unit: win → +payout, loss → −1, push → 0
        pnl_per_unit = np.where(
            result_arr > 0.5,   payout,
            np.where(result_arr == 0.5, 0.0, -1.0)
        )
    else:
        raise ValueError(
            "bet_log must contain either ('pnl_units', 'stake_units') "
            "or ('result',) columns."
        )

    # ── Base stake fractions (at the divisor used when the log was generated) ─
    # We assume the log was generated at some reference divisor.  We infer this
    # from the 'f_kelly' column if available, otherwise use 'stake_units' directly.
    if "f_kelly" in bet_log.columns and "stake_units" in bet_log.columns:
        base_stake = bet_log["f_kelly"].to_numpy(dtype=np.float64)
        # f_kelly = f_star / divisor_used; f_star = f_kelly * divisor_used
        # We want to express stakes as a fraction of bankroll at any divisor d:
        # stake(d) = f_star / d = f_kelly * (divisor_used / d)
        # We DON'T know divisor_used, so we normalise base_stake to f* = f_kelly * 4
        # (assume the log was generated at divisor=4 = quarter-Kelly, default).
        # If the log has f_star explicitly, use that instead.
        if "f_star" in bet_log.columns:
            f_star_arr = bet_log["f_star"].to_numpy(dtype=np.float64)
        else:
            # Assume quarter-Kelly was used when generating the log
            f_star_arr = base_stake * 4.0
    elif "stake_units" in bet_log.columns:
        # No f_kelly column; treat stake_units as the baseline.
        # Rescaling will be relative to divisor=4 (quarter-Kelly assumption).
        f_star_arr = bet_log["stake_units"].to_numpy(dtype=np.float64) * 4.0
    else:
        raise ValueError("bet_log must contain 'stake_units' or 'f_kelly' columns.")

    f_star_arr = np.maximum(f_star_arr, 0.0)

    # ── Bootstrap loop ────────────────────────────────────────────────────────
    for divisor, label in zip(kelly_divisors, kelly_labels):
        # Compute stakes for this divisor
        stake_this = f_star_arr / max(divisor, 1e-9)   # fraction of bankroll per bet

        # Vectorised bootstrap: draw n_paths × n_bets indices with replacement
        idx = rng.integers(0, n_bets, size=(n_paths, n_bets))

        # stake_paths[path, bet]
        stake_paths = stake_this[idx]             # (n_paths, n_bets)
        pnl_paths   = pnl_per_unit[idx]           # (n_paths, n_bets) per-unit P&L

        # Bankroll change per bet per path (in bankroll-fraction units)
        pnl_dollar  = stake_paths * pnl_paths     # (n_paths, n_bets)

        # Cumulative bankroll relative to starting_bankroll
        cum_returns = np.cumsum(pnl_dollar, axis=1) / starting_bankroll
        bankroll_paths = starting_bankroll + np.cumsum(pnl_dollar, axis=1)
        # Prepend starting value
        bankroll_full = np.concatenate(
            [np.full((n_paths, 1), starting_bankroll), bankroll_paths],
            axis=1,
        )   # (n_paths, n_bets + 1)

        # ── Maximum drawdown per path ────────────────────────────────────────
        # MDD[path] = max over time of (HWM - current) / HWM
        # Using cumulative max as HWM
        hwm = np.maximum.accumulate(bankroll_full, axis=1)   # (n_paths, n_bets+1)
        drawdown = (hwm - bankroll_full) / np.maximum(hwm, 1e-12)
        mdd      = drawdown.max(axis=1)                       # (n_paths,)

        terminal = bankroll_full[:, -1]                       # (n_paths,)

        results_list.append(DrawdownSimResult(
            kelly_divisor=divisor,
            kelly_label=label,
            p95_max_drawdown=float(np.percentile(mdd,      95)),
            p50_max_drawdown=float(np.percentile(mdd,      50)),
            p05_max_drawdown=float(np.percentile(mdd,       5)),
            mean_max_drawdown=float(mdd.mean()),
            p95_terminal_bankroll=float(np.percentile(terminal,  5)),   # pessimistic 5th pct
            p50_terminal_bankroll=float(np.percentile(terminal, 50)),
            mean_terminal_bankroll=float(terminal.mean()),
            n_bets=n_bets,
            n_paths=n_paths,
        ))

    return results_list
