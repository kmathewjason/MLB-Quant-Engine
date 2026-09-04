"""
optimizer.kelly
===============
Kelly criterion bet sizing — single-bet fractional Kelly and
covariance-adjusted portfolio Kelly.

Single-bet Kelly
----------------
For a bet with fair win-probability p and decimal odds b+1 (payout = b per unit
staked on a win):

    f* = (b·p - q) / b      where q = 1 - p

This maximises the long-run geometric growth rate E[log W].
We apply a fractional divisor (default 4 — "quarter-Kelly") to reduce variance
at the cost of slightly lower growth:

    f = f* / divisor

Convert American odds → decimal:
    american ≥ 0  →  decimal = american/100 + 1
    american < 0  →  decimal = 100/|american| + 1

Edge filter
-----------
Only size bets where model_prob > fair_prob (positive expected value).
kelly_fraction() returns 0.0 when there is no edge.

Covariance-adjusted portfolio Kelly
------------------------------------
When sizing multiple simultaneous bets, the correlated-Kelly solution
maximises:

    E[log W] ≈ f·μ - ½ f·Σ·fᵀ

where:
    μ_i = b_i·p_i - q_i      (per-unit edge for bet i)
    Σ    = covariance matrix of bet outcomes

The quadratic-programme solution (unconstrained):

    f* = Σ⁻¹ · μ

We project f* onto the feasible simplex {f ≥ 0, Σf_i ≤ max_total_exposure}
using scipy.optimize.minimize (SLSQP), which handles the inequality constraint.

Reference
---------
Kelly, J.L. (1956). "A New Interpretation of Information Rate."
MacLean, Thorp & Ziemba (2010). "The Kelly Capital Growth Investment Criterion."

Public API
----------
kelly_fraction(model_prob, fair_odds_american, fractional_divisor)  -> float
portfolio_kelly(bets_df, cov_matrix, bankroll, max_total_exposure,
                fractional_divisor)                                  -> pd.DataFrame
american_to_decimal(american)                                        -> float
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Odds converters
# ---------------------------------------------------------------------------

def american_to_decimal(american: float) -> float:
    """
    Convert American odds to decimal odds.

    +150  →  2.50   (profit of 1.50 per unit staked)
    -110  →  1.909…
    """
    a = float(american)
    if a >= 100.0:
        return a / 100.0 + 1.0
    return 100.0 / abs(a) + 1.0


def decimal_to_american(decimal: float) -> float:
    """
    Convert decimal odds to American odds.

    2.50  →  +150
    1.909 →  -110
    """
    if decimal <= 1.0:
        raise ValueError(f"Decimal odds must be > 1.0; got {decimal}")
    if decimal >= 2.0:
        return (decimal - 1.0) * 100.0
    return -100.0 / (decimal - 1.0)


# ---------------------------------------------------------------------------
# Single-bet fractional Kelly
# ---------------------------------------------------------------------------

def kelly_fraction(
    model_prob: float,
    fair_odds_american: float,
    fractional_divisor: float = 4.0,
) -> float:
    """
    Compute the fractional Kelly stake as a fraction of bankroll.

    Parameters
    ----------
    model_prob          : model's estimated win probability for this bet
    fair_odds_american  : American odds at which we place the bet
                          (should be the line we get, already de-vigged
                           externally by the caller using vig_removal)
    fractional_divisor  : divisor applied to full-Kelly stake (default 4).
                          Use 1.0 for full Kelly (maximum growth, high variance).

    Returns
    -------
    float — fraction of bankroll to wager (0.0 if no edge or divisor ≤ 0).

    Notes
    -----
    Returns 0.0 when:
    - model_prob ≤ 0 or ≥ 1 (degenerate probability)
    - computed f* ≤ 0 (model has no edge at this price)
    - divisor ≤ 0
    """
    if fractional_divisor <= 0.0:
        return 0.0
    if not (0.0 < model_prob < 1.0):
        return 0.0

    b = american_to_decimal(fair_odds_american) - 1.0   # net profit per unit
    if b <= 0.0:
        return 0.0

    p = model_prob
    q = 1.0 - p
    f_star = (b * p - q) / b

    if f_star <= 0.0:
        return 0.0

    return float(f_star / fractional_divisor)


def kelly_ev(model_prob: float, fair_odds_american: float) -> float:
    """
    Expected value per unit staked.

    EV = b·p - q   where b = decimal - 1, p = model prob, q = 1-p.

    Positive EV is required before sizing; this function exposes the raw EV
    for logging and filtering.

    Returns float (may be negative — do NOT bet if EV ≤ 0).
    """
    if not (0.0 < model_prob < 1.0):
        return float("-inf")
    b = american_to_decimal(fair_odds_american) - 1.0
    return float(b * model_prob - (1.0 - model_prob))


# ---------------------------------------------------------------------------
# Portfolio covariance-adjusted Kelly
# ---------------------------------------------------------------------------

def portfolio_kelly(
    bets_df: pd.DataFrame,
    cov_matrix: np.ndarray | None = None,
    bankroll: float = 1.0,
    max_total_exposure: float = 0.25,
    fractional_divisor: float = 4.0,
) -> pd.DataFrame:
    """
    Covariance-adjusted Kelly sizing for a portfolio of simultaneous bets.

    Maximises  f·μ - ½ f·Σ·fᵀ  subject to  f ≥ 0,  Σf_i ≤ max_total_exposure.

    Parameters
    ----------
    bets_df : DataFrame with (at minimum) columns:
              - model_prob         : float, model win probability
              - fair_odds_american : float, line we expect to take

              Optional:
              - bet_id             : identifier (preserved in output)

    cov_matrix : (n, n) covariance matrix of bet-outcome indicators.
                 If None, assumed diagonal (independent bets — reverts to
                 single-Kelly sizing scaled to respect total exposure cap).

    bankroll           : total bankroll in currency units (default 1.0 = fractions)
    max_total_exposure : hard cap on total fraction of bankroll across all bets
    fractional_divisor : applied to the portfolio-optimal f* vector

    Returns
    -------
    DataFrame — input bets_df plus extra columns:
        ev            : expected value per unit
        f_star        : unconstrained Kelly fraction
        f_kelly       : fractional Kelly after divisor
        f_final       : final stake fraction (after portfolio cap)
        stake_units   : stake in bankroll units (f_final * bankroll)
    """
    from scipy.optimize import minimize  # noqa: PLC0415

    n = len(bets_df)
    if n == 0:
        return bets_df.copy().assign(ev=[], f_star=[], f_kelly=[], f_final=[], stake_units=[])

    probs = bets_df["model_prob"].to_numpy(dtype=np.float64)
    odds  = bets_df["fair_odds_american"].to_numpy(dtype=np.float64)

    # Net payout per unit (b = decimal - 1)
    b = np.array([american_to_decimal(o) - 1.0 for o in odds])
    q = 1.0 - probs

    # Expected value vector μ_i = b_i * p_i - q_i
    mu = b * probs - q  # (n,)

    # Single-Kelly vector f* = μ_i / b_i  (unconstrained, per-bet)
    f_star_single = np.where(b > 0, mu / b, 0.0)
    f_star_single = np.maximum(f_star_single, 0.0)   # no short-selling

    if cov_matrix is None:
        # Diagonal case: independent bets; use single-Kelly fractions
        f_opt = f_star_single.copy()
    else:
        # Quadratic programme: maximise f·μ - ½ f·Σ·fᵀ
        # subject to f ≥ 0 and sum(f) ≤ max_total_exposure
        Sigma = np.asarray(cov_matrix, dtype=np.float64)
        if Sigma.shape != (n, n):
            raise ValueError(
                f"cov_matrix shape {Sigma.shape} does not match n_bets={n}."
            )

        def neg_obj(f: np.ndarray) -> float:
            return float(-(f @ mu) + 0.5 * (f @ Sigma @ f))

        def neg_grad(f: np.ndarray) -> np.ndarray:
            return -(mu) + Sigma @ f

        bounds     = [(0.0, max_total_exposure)] * n
        constraint = {"type": "ineq", "fun": lambda f: max_total_exposure - f.sum()}
        x0 = np.full(n, max_total_exposure / n)

        res = minimize(
            neg_obj,
            x0,
            jac=neg_grad,
            method="SLSQP",
            bounds=bounds,
            constraints=[constraint],
            options={"ftol": 1e-10, "maxiter": 500},
        )
        f_opt = np.maximum(res.x, 0.0)

    # Apply fractional divisor
    divisor = max(fractional_divisor, 1e-6)
    f_kelly = f_opt / divisor

    # Enforce per-bet positivity and total exposure cap
    f_kelly = np.maximum(f_kelly, 0.0)
    total   = f_kelly.sum()
    if total > max_total_exposure:
        f_kelly = f_kelly * (max_total_exposure / total)

    result = bets_df.copy()
    result["ev"]          = mu.round(6)
    result["f_star"]      = f_star_single.round(6)
    result["f_kelly"]     = f_kelly.round(6)
    result["f_final"]     = f_kelly.round(6)
    result["stake_units"] = (f_kelly * bankroll).round(4)

    return result
