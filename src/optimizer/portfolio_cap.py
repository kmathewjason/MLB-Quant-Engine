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
