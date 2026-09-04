"""
optimizer.vig_removal
=====================
Removes bookmaker vig (overround) from a two-outcome or multi-outcome market
to recover fair (true) implied probabilities.

Three de-vig methods
--------------------
1. Power method (Smoczynski & Tomczyk 2010)
   Two equivalent formulations:

   (a) Find exponent k >= 1 such that sum(p_raw_i^k) = 1;
       p_fair_i = p_raw_i^k.

   (b) Find exponent n = 1/k <= 1 such that sum(p_raw_i^(1/n)) = 1;
       p_fair_i = p_raw_i^(1/n).

   Both solve identically -- only the search variable differs.
   power_devig() solves for k > 1; power_devig_inv_k() is an alias using
   the 1/k framing.  For a balanced -110/-110 market, k ~= 1.047, n ~= 0.955.

   - Preserves the log-odds ratio between outcomes
   - Works for 2-way and multi-outcome markets
   - Default method for all CLV calculations

2. Additive method (simple proportional de-vig)
   p_fair_i = p_raw_i / sum(p_raw_j)
   - Fastest; assumes vig is proportionally distributed
   - Biased toward favourites in lopsided markets (favourite-longshot bias)

3. Shin model (Shin 1992; Jullien & Salanié 1994)
   Accounts for inside-trader / sharp-money distortion.

Divergence diagnostics
-----------------------
The difference between power and additive fair probs is a diagnostic for
favourite-longshot bias (FLB).  A book shading the favourite's line harder
than the longshot's inflates the favourite's raw prob relative to what the
power method assigns.

    flb_index = (p_power_fav - p_additive_fav) / vig_pct

Positive flb_index -> favourite is under-priced by the additive method;
standard FLB pattern.  Near-zero -> vig is evenly distributed.

Public API
----------
american_to_implied_prob(odds)                        -> float
power_devig(american_odds)                            -> np.ndarray
power_devig_inv_k(american_odds)                      -> np.ndarray  (1/k alias)
additive_devig(american_odds)                         -> np.ndarray
shin_devig(american_odds)                             -> np.ndarray
vig_percentage(american_odds)                         -> float
best_devig(american_odds, method)                     -> np.ndarray
method_divergence(american_odds, method_a, method_b)  -> DeVigDivergence
favorite_longshot_bias(american_odds)                 -> FLBResult
"""

from __future__ import annotations

import math
from dataclasses import dataclass
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


# ---------------------------------------------------------------------------
# Inverse-k alias  (1/k formulation — numerically identical to power_devig)
# ---------------------------------------------------------------------------

def power_devig_inv_k(american_odds: Sequence[int | float]) -> np.ndarray:
    """
    Power-method de-vig using the inverse-exponent (1/k) framing.

    Mathematically identical to power_devig().  Solves for n = 1/k in (0, 1]
    such that sum(p_raw_i ^ (1/n)) = 1, then returns p_fair_i = p_raw_i^(1/n).

    Why both exist
    --------------
    power_devig()       solves for k > 1  (raises raw probs to a power > 1,
                        pulling them up toward 1 and reducing overround).
    power_devig_inv_k() frames the same operation as "raise to a power n < 1",
    which some literature describes as shrinking the raw probs toward the
    uniform distribution.  Both are equivalent because p^k = p^(1/(1/k)).

    For a balanced -110/-110 market:
        k  ~= 1.047  (power_devig)
        n  ~= 0.955  (power_devig_inv_k)
        k * n ~= 1.0 (always, by definition)

    Parameters / Returns
    --------------------
    Identical to power_devig().
    """
    return power_devig(american_odds)


# ---------------------------------------------------------------------------
# Divergence diagnostics
# ---------------------------------------------------------------------------

@dataclass
class DeVigDivergence:
    """
    Divergence between two de-vig methods on the same market.

    Attributes
    ----------
    probs_a        : fair probabilities from method_a  (shape n_outcomes,)
    probs_b        : fair probabilities from method_b  (shape n_outcomes,)
    diff           : probs_a - probs_b per outcome (signed)
    l1             : sum(|diff|) — total-variation distance (scaled by 2)
    l_inf          : max(|diff|) — worst-case per-outcome disagreement
    js_divergence  : Jensen-Shannon divergence (base 2, range [0, 1]).
                     0 = identical distributions; 1 = maximally different.
    method_a       : name of method a
    method_b       : name of method b
    """
    probs_a:       np.ndarray
    probs_b:       np.ndarray
    diff:          np.ndarray
    l1:            float
    l_inf:         float
    js_divergence: float
    method_a:      str
    method_b:      str

    def to_dict(self) -> dict:
        return {
            "method_a":      self.method_a,
            "method_b":      self.method_b,
            "l1":            round(self.l1,            6),
            "l_inf":         round(self.l_inf,          6),
            "js_divergence": round(self.js_divergence,  6),
            "diff":          self.diff.tolist(),
            "probs_a":       self.probs_a.tolist(),
            "probs_b":       self.probs_b.tolist(),
        }


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence (base 2, [0, 1]) between two distributions."""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-15, 1.0)
    q = np.clip(np.asarray(q, dtype=np.float64), 1e-15, 1.0)
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * np.log2(p / m)))
    kl_qm = float(np.sum(q * np.log2(q / m)))
    return float(np.clip(0.5 * kl_pm + 0.5 * kl_qm, 0.0, 1.0))


def method_divergence(
    american_odds: Sequence[int | float],
    method_a: DeVigMethod = "power",
    method_b: DeVigMethod = "additive",
) -> "DeVigDivergence":
    """
    Compute the divergence between two de-vig methods on the same market.

    The divergence at the favourite outcome is the primary diagnostic for
    favourite-longshot bias: if power assigns the favourite a materially higher
    probability than additive, the book charges more vig on the longshot's
    side — the standard FLB pattern in sports betting markets.

    Parameters
    ----------
    american_odds : sequence of American odds for each outcome (>= 2 outcomes)
    method_a      : first method  (default "power")
    method_b      : second method (default "additive")

    Returns
    -------
    DeVigDivergence with L1, L-inf, and Jensen-Shannon metrics.

    Example
    -------
    >>> d = method_divergence([-300, 240])
    >>> d.l_inf           # large for lopsided markets
    >>> d.js_divergence   # JSD between the two fair-prob distributions
    >>> d.diff[0]         # how much more/less prob method_a assigns to outcome 0
    """
    pa = best_devig(american_odds, method=method_a)
    pb = best_devig(american_odds, method=method_b)
    diff = pa - pb
    return DeVigDivergence(
        probs_a=pa,
        probs_b=pb,
        diff=diff,
        l1=float(np.abs(diff).sum()),
        l_inf=float(np.abs(diff).max()),
        js_divergence=_js_divergence(pa, pb),
        method_a=method_a,
        method_b=method_b,
    )


@dataclass
class FLBResult:
    """
    Favourite-longshot bias diagnostics for one market.

    Attributes
    ----------
    favourite_idx      : index of the highest raw-implied-probability outcome
    vig_pct            : market overround as a percentage
    power_probs        : power-method fair probabilities
    additive_probs     : additive-method fair probabilities
    flb_index          : (power_fav - additive_fav) / vig_pct
                         > 0 -> standard FLB (favourite under-priced by additive)
                         ~ 0 -> vig distributed evenly
                         < 0 -> reverse FLB (longshot under-priced by additive)
    power_fav_prob     : power fair prob for the favourite outcome
    additive_fav_prob  : additive fair prob for the favourite outcome
    divergence         : full DeVigDivergence (power vs additive)
    """
    favourite_idx:     int
    vig_pct:           float
    power_probs:       np.ndarray
    additive_probs:    np.ndarray
    flb_index:         float
    power_fav_prob:    float
    additive_fav_prob: float
    divergence:        "DeVigDivergence"

    def to_dict(self) -> dict:
        return {
            "favourite_idx":     self.favourite_idx,
            "vig_pct":           round(self.vig_pct,            4),
            "flb_index":         round(self.flb_index,           6),
            "power_fav_prob":    round(self.power_fav_prob,      6),
            "additive_fav_prob": round(self.additive_fav_prob,   6),
            "l1":                round(self.divergence.l1,       6),
            "l_inf":             round(self.divergence.l_inf,    6),
            "js_divergence":     round(self.divergence.js_divergence, 6),
        }


def favorite_longshot_bias(
    american_odds: Sequence[int | float],
) -> "FLBResult":
    """
    Diagnose favourite-longshot bias for a single market.

    Compares power-method and additive-method fair probabilities.  The FLB
    index is the favourite's probability gap normalised by vig:

        flb_index = (p_power_fav - p_additive_fav) / vig_pct

    Interpretation
    --------------
    flb_index > 0  Standard FLB.  The book charges more vig on the longshot
                   side; the favourite appears "cheap" after additive de-vig
                   but "fairly priced" after power de-vig.  Bettors who use
                   additive de-vig will systematically overestimate edge on
                   the longshot and underestimate edge on the favourite.

    flb_index < 0  Reverse FLB (rare in mainstream markets, more common in
                   some exchange or prop markets).

    flb_index ~ 0  Vig is evenly distributed; both methods agree.

    Parameters
    ----------
    american_odds : sequence of American odds (>= 2 outcomes)

    Returns
    -------
    FLBResult dataclass.
    """
    raw     = np.array([american_to_implied_prob(o) for o in american_odds])
    fav_idx = int(raw.argmax())
    vig     = vig_percentage(american_odds)

    p_power    = power_devig(american_odds)
    p_additive = additive_devig(american_odds)
    div        = method_divergence(american_odds, method_a="power", method_b="additive")

    flb = float(p_power[fav_idx] - p_additive[fav_idx]) / max(abs(vig), 1e-9)

    return FLBResult(
        favourite_idx=fav_idx,
        vig_pct=vig,
        power_probs=p_power,
        additive_probs=p_additive,
        flb_index=flb,
        power_fav_prob=float(p_power[fav_idx]),
        additive_fav_prob=float(p_additive[fav_idx]),
        divergence=div,
    )
