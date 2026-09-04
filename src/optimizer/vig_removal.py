"""
optimizer.vig_removal
=====================
Removes bookmaker vig (overround) from a two-outcome or multi-outcome market
to recover fair (true) implied probabilities.

Three methods are implemented:

1. Power method (Smoczynski & Tomczyk 2010)
   Find exponent k such that Σ p_raw_i^k = 1, then p_fair_i = p_raw_i^k.
   - Preserves the relative odds structure
   - Works well for two-way and three-way markets
   - Our default method for all CLV calculations

2. Additive method (simple proportional de-vig)
   Overround = Σ p_raw_i - 1
   p_fair_i = p_raw_i / Σ p_raw_j
   - Fastest; assumes vig is proportionally distributed across all outcomes
   - Good approximation for balanced markets (moneylines near -110/-110)
   - Biased toward favourites in lopsided markets

3. Shin model (Shin 1992; Jullien & Salanié 1994)
   Accounts for the presence of inside-trader / sharp-money probability
   distortion.  For a two-outcome market:

       p_fair_i = [sqrt(z² + 4(1-z)·p_raw_i² / Σp_raw_j²) - z] / [2(1-z)]

   where z is the fraction of bets placed by informed traders (estimated
   by solving Σ p_fair_i = 1 numerically).  More accurate than the
   power method for lopsided markets but slightly slower.

Public API
----------
american_to_implied_prob(odds)                     -> float  (re-exported convenience)
power_devig(american_odds)                         -> np.ndarray   fair probs
additive_devig(american_odds)                      -> np.ndarray   fair probs
shin_devig(american_odds)                          -> np.ndarray   fair probs
vig_percentage(american_odds)                      -> float        % overround
best_devig(american_odds, method)                  -> np.ndarray   dispatch by method
"""

from __future__ import annotations

import math
from typing import Literal, Sequence

import numpy as np
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# Probability ↔ odds converters (re-exported here for convenience)
# ---------------------------------------------------------------------------

def american_to_implied_prob(american: int | float) -> float:
    """Raw (vigged) implied probability from American odds."""
    a = float(american)
    if a >= 100:
        return 100.0 / (a + 100.0)
    return abs(a) / (abs(a) + 100.0)


def implied_prob_to_american(prob: float) -> float:
    """Convert fair probability to American odds."""
    if not (0 < prob < 1):
        raise ValueError(f"Probability must be in (0, 1); got {prob}")
    if prob >= 0.5:
        return -100.0 * prob / (1.0 - prob)
    return 100.0 * (1.0 - prob) / prob


def vig_percentage(american_odds: Sequence[int | float]) -> float:
    """
    Return the bookmaker's vig (overround) as a percentage.

    vig_pct = (Σ p_raw_i - 1) × 100

    For a balanced -110 / -110 market: vig ≈ 4.76%.
    """
    raw_probs = np.array([american_to_implied_prob(o) for o in american_odds])
    return float((raw_probs.sum() - 1.0) * 100.0)


# ---------------------------------------------------------------------------
# Power de-vig (primary method)
# ---------------------------------------------------------------------------

def power_devig(american_odds: Sequence[int | float]) -> np.ndarray:
    """
    Power-method de-vig (Smoczynski & Tomczyk 2010).

    Find k such that Σ p_raw_i^k = 1, then p_fair_i = p_raw_i^k.

    Solved via Brent's method on the monotone function f(k) = Σ p_i^k - 1.

    Parameters
    ----------
    american_odds : sequence of American odds for each outcome
                    (e.g. [-110, -110] or [-150, 130])

    Returns
    -------
    np.ndarray of fair probabilities, shape (n_outcomes,), sums to 1.

    Raises
    ------
    ValueError  if fewer than 2 odds provided.
    """
    raw = np.array([american_to_implied_prob(o) for o in american_odds])
    if len(raw) < 2:
        raise ValueError("Need at least 2 odds to de-vig.")

    overround = raw.sum()
    if abs(overround - 1.0) < 1e-9:
        return raw  # no vig; return as-is

    # f(k) = Σ p_i^k - 1. We need f(k) = 0.
    # k=1 → overround > 1; k→∞ → 0 (max prob^∞ → 0 if max<1)
    def f(k: float) -> float:
        return float(np.sum(raw ** k)) - 1.0

    # k must be > 0; f(1) = overround - 1 > 0; f(large) ≈ 0-ε < 0
    k_lo, k_hi = 1.0, 20.0
    # Edge case: if all probs are equal (perfectly balanced book), k = 1/n
    try:
        k = brentq(f, k_lo, k_hi, xtol=1e-10, maxiter=100)
    except ValueError:
        # Fallback: bisect in a wider range
        k = brentq(f, 0.5, 50.0, xtol=1e-10, maxiter=200)

    fair = raw ** k
    return fair / fair.sum()   # normalise for floating-point hygiene


# ---------------------------------------------------------------------------
# Additive de-vig (fastest; use for batch processing)
# ---------------------------------------------------------------------------

def additive_devig(american_odds: Sequence[int | float]) -> np.ndarray:
    """
    Additive (proportional) de-vig: p_fair_i = p_raw_i / Σ p_raw_j.

    This assumes vig is distributed proportionally across all outcomes,
    which is the least biased when the market is balanced.

    Returns
    -------
    np.ndarray of fair probabilities, sums to 1.
    """
    raw = np.array([american_to_implied_prob(o) for o in american_odds])
    if len(raw) < 2:
        raise ValueError("Need at least 2 odds to de-vig.")
    return raw / raw.sum()


# ---------------------------------------------------------------------------
# Shin de-vig (most theoretically motivated; best for lopsided markets)
# ---------------------------------------------------------------------------

def shin_devig(american_odds: Sequence[int | float]) -> np.ndarray:
    """
    Shin-model de-vig (Shin 1992; Jullien & Salanié 1994).

    For a market with n outcomes, estimate the proportion z of informed bets.
    Fair probabilities satisfy:

        p_fair_i = [sqrt(z² + 4(1-z)·q_i·p_raw_i) - z] / [2(1-z)]

    where q_i = p_raw_i / Σ p_raw_j  (additive normalisation).

    z is found by solving Σ p_fair_i = 1 numerically.

    Parameters
    ----------
    american_odds : sequence of American odds

    Returns
    -------
    np.ndarray of Shin-fair probabilities, sums to 1.
    """
    raw = np.array([american_to_implied_prob(o) for o in american_odds])
    if len(raw) < 2:
        raise ValueError("Need at least 2 odds to de-vig.")

    n = len(raw)
    overround = raw.sum()
    if abs(overround - 1.0) < 1e-9:
        return raw

    # Normalised raw probs (for the Shin formula)
    q = raw / overround

    def _shin_probs(z: float) -> np.ndarray:
        disc = z * z + 4.0 * (1.0 - z) * q * raw
        # Clip to avoid sqrt(negative) due to numerical noise
        disc = np.maximum(disc, 0.0)
        return (np.sqrt(disc) - z) / (2.0 * (1.0 - z))

    def f(z: float) -> float:
        return float(_shin_probs(z).sum()) - 1.0

    # z must be in (0, 1); at z=0 Shin reduces to additive; at z→1 formula breaks
    # f(0) ≈ overround - 1 > 0; as z increases, sum decreases toward 0
    try:
        z_star = brentq(f, 0.0, 0.999, xtol=1e-12, maxiter=200)
    except ValueError:
        # Fallback to additive if Shin solver fails
        return additive_devig(american_odds)

    fair = _shin_probs(z_star)
    return fair / fair.sum()


# ---------------------------------------------------------------------------
# Unified dispatch
# ---------------------------------------------------------------------------

DeVigMethod = Literal["power", "additive", "shin"]


def best_devig(
    american_odds: Sequence[int | float],
    method: DeVigMethod = "power",
) -> np.ndarray:
    """
    De-vig using the specified method.

    Parameters
    ----------
    american_odds : sequence of American odds
    method        : "power" (default), "additive", or "shin"

    Returns
    -------
    np.ndarray of fair probabilities, sums to 1.
    """
    if method == "power":
        return power_devig(american_odds)
    elif method == "additive":
        return additive_devig(american_odds)
    elif method == "shin":
        return shin_devig(american_odds)
    else:
        raise ValueError(f"Unknown de-vig method '{method}'. Use 'power', 'additive', or 'shin'.")
