"""
features.matchup_features
=========================
Batter-vs-pitcher matchup probability adjustment using the log5 / odds-ratio
method, generalised to a multinomial plate-appearance outcome space.

Statistical justification
--------------------------
For a binary event, Bill James' log5 formula gives the probability that
player A beats player B given their individual rates and a league baseline:

    P(A|B) = (a·b/c) / [a·b/c + (1-a)(1-b)/(1-c)]

where a = batter rate, b = pitcher rate, c = league rate.

This is algebraically equivalent to the *odds-ratio* formula:

    odds(A|B) = odds(a) · odds(b) / odds(c)

where odds(p) = p / (1-p).

The odds-ratio generalizes cleanly to the multinomial case.  For each
outcome category k in {K, BB, HBP, 1B, 2B, 3B, HR, OUT}, form the
relative-risk (risk ratio vs. the league):

    RR_k(batter)  = b_k / lg_k
    RR_k(pitcher) = p_k / lg_k

Then the matchup probability for outcome k is proportional to:

    q_k ∝ lg_k · RR_k(batter) · RR_k(pitcher)
         = b_k · p_k / lg_k

Normalise so q sums to 1.  This is the direct multinomial extension of log5.

Equivalently in odds space (for each outcome vs. all others collapsed):

    odds_matchup_k = odds_batter_k · odds_pitcher_k / odds_league_k

then convert back to probabilities via softmax on log-odds.

Both forms are implemented and numerically identical (within floating point).

Public API
----------
Outcome = Literal["K", "BB", "HBP", "1B", "2B", "3B", "HR", "OUT"]

log5_binary(batter_rate, pitcher_rate, league_rate)   -> float
log5_multinomial(batter_probs, pitcher_probs, league_probs) -> np.ndarray
odds_ratio_multinomial(...)                           -> np.ndarray  (same result)
matchup_pa_probs(batter_dict, pitcher_dict, league_dict) -> dict[Outcome, float]
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Canonical PA outcome ordering (used throughout the engine)
# ---------------------------------------------------------------------------

PA_OUTCOMES: tuple[str, ...] = ("K", "BB", "HBP", "1B", "2B", "3B", "HR", "OUT")
N_OUTCOMES: int = len(PA_OUTCOMES)

# MLB approximate league-average PA outcome distribution (2018-2022)
MLB_LEAGUE_AVG: dict[str, float] = {
    "K":   0.2217,
    "BB":  0.0845,
    "HBP": 0.0103,
    "1B":  0.1479,
    "2B":  0.0472,
    "3B":  0.0046,
    "HR":  0.0337,
    "OUT": 0.4501,   # all other outs (non-K, non-BB)
}


# ---------------------------------------------------------------------------
# Binary log5
# ---------------------------------------------------------------------------

def log5_binary(batter_rate: float, pitcher_rate: float, league_rate: float) -> float:
    """
    Bill James log5 formula for a single binary event.

    Parameters
    ----------
    batter_rate  : batter's observed / shrunk rate for this event
    pitcher_rate : pitcher's observed / shrunk rate
    league_rate  : league-average rate (denominator baseline)

    Returns
    -------
    Matchup-adjusted probability ∈ (0, 1)

    Mathematical identity
    ---------------------
    log5(a, b, c) ≡ odds_ratio(a, b, c)
    = (a·b/c) / [a·b/c + (1-a)(1-b)/(1-c)]
    """
    if not (0 < league_rate < 1):
        raise ValueError(f"league_rate must be in (0, 1); got {league_rate}")
    if batter_rate < 0 or pitcher_rate < 0:
        raise ValueError("Rates must be non-negative")

    # Clip to (ε, 1-ε) to avoid log/division explosions
    eps = 1e-9
    a = float(np.clip(batter_rate, eps, 1 - eps))
    b = float(np.clip(pitcher_rate, eps, 1 - eps))
    c = float(np.clip(league_rate, eps, 1 - eps))

    num = a * b / c
    den = num + (1.0 - a) * (1.0 - b) / (1.0 - c)
    return num / den


# ---------------------------------------------------------------------------
# Multinomial log5 / odds-ratio  (primary method)
# ---------------------------------------------------------------------------

def log5_multinomial(
    batter_probs: np.ndarray | Sequence[float],
    pitcher_probs: np.ndarray | Sequence[float],
    league_probs: np.ndarray | Sequence[float],
) -> np.ndarray:
    """
    Multinomial generalisation of log5 via relative-risk composition.

    For each outcome k:  q_k ∝ batter_k · pitcher_k / league_k

    Normalise so sum(q) = 1.

    Parameters
    ----------
    batter_probs  : (K,) array — batter's rate per outcome category
    pitcher_probs : (K,) array — pitcher's rate per outcome category
    league_probs  : (K,) array — league average per outcome category

    Returns
    -------
    (K,) normalised matchup probability vector.
    """
    b = np.asarray(batter_probs, dtype=np.float64)
    p = np.asarray(pitcher_probs, dtype=np.float64)
    lg = np.asarray(league_probs, dtype=np.float64)

    if b.shape != p.shape or b.shape != lg.shape:
        raise ValueError("batter_probs, pitcher_probs, league_probs must have the same shape")
    if np.any(lg <= 0):
        raise ValueError("league_probs must be strictly positive for all outcomes")

    # Raw unnormalised matchup probabilities
    q = b * p / lg

    total = q.sum()
    if total <= 0:
        raise ValueError("Matchup probability vector sums to zero — check input rates")
    return q / total


def odds_ratio_multinomial(
    batter_probs: np.ndarray | Sequence[float],
    pitcher_probs: np.ndarray | Sequence[float],
    league_probs: np.ndarray | Sequence[float],
) -> np.ndarray:
    """
    Odds-ratio form of the multinomial matchup — mathematically equivalent
    to log5_multinomial, implemented via log-space for numerical stability.

    For each outcome k:
        log_odds_k = log_odds(b_k) + log_odds(p_k) - log_odds(lg_k)
    Convert back via softmax on log-odds space.

    See Davenport & Woolner (2004), "Basics of the Odds Ratio" for
    justification that log5 = odds-ratio method.
    """
    b  = np.asarray(batter_probs,  dtype=np.float64)
    p  = np.asarray(pitcher_probs, dtype=np.float64)
    lg = np.asarray(league_probs,  dtype=np.float64)

    eps = 1e-9
    b  = np.clip(b,  eps, 1 - eps)
    p  = np.clip(p,  eps, 1 - eps)
    lg = np.clip(lg, eps, 1 - eps)

    log_odds = (np.log(b) - np.log(1 - b)
               + np.log(p) - np.log(1 - p)
               - np.log(lg) + np.log(1 - lg))

    # Softmax to convert log-odds vector back to probabilities
    # (Subtract max for numerical stability)
    log_odds -= log_odds.max()
    q = np.exp(log_odds)
    return q / q.sum()


# ---------------------------------------------------------------------------
# Dict-based convenience API
# ---------------------------------------------------------------------------

def matchup_pa_probs(
    batter_dict: dict[str, float],
    pitcher_dict: dict[str, float],
    league_dict: dict[str, float] | None = None,
    method: Literal["log5", "odds_ratio"] = "log5",
) -> dict[str, float]:
    """
    Compute matchup-adjusted PA outcome probabilities from rate dicts.

    Parameters
    ----------
    batter_dict  : {outcome: rate} for the batter
    pitcher_dict : {outcome: rate} for the pitcher
    league_dict  : {outcome: rate}; defaults to MLB_LEAGUE_AVG
    method       : "log5" or "odds_ratio" (identical results, different impl)

    Returns
    -------
    dict mapping each PA outcome to its matchup probability, summing to 1.

    Notes
    -----
    Both dicts are normalised internally, so you can pass raw counts
    (e.g. numerators) and they will be converted to rates.  However, for
    best results pass shrunk posterior rates from bayesian_shrinkage.py.
    """
    if league_dict is None:
        league_dict = MLB_LEAGUE_AVG

    # Build aligned arrays in canonical outcome order
    def _to_array(d: dict[str, float]) -> np.ndarray:
        arr = np.array([d.get(k, 0.0) for k in PA_OUTCOMES], dtype=np.float64)
        total = arr.sum()
        if total <= 0:
            raise ValueError(f"Rate dict sums to zero: {d}")
        return arr / total   # normalise to probability simplex

    b  = _to_array(batter_dict)
    p  = _to_array(pitcher_dict)
    lg = _to_array(league_dict)

    # Ensure league probabilities are strictly positive (avoid /0 for rare outcomes)
    eps = 1e-6
    lg = np.maximum(lg, eps)
    lg = lg / lg.sum()

    if method == "log5":
        q = log5_multinomial(b, p, lg)
    elif method == "odds_ratio":
        q = odds_ratio_multinomial(b, p, lg)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'log5' or 'odds_ratio'.")

    return {outcome: float(q[i]) for i, outcome in enumerate(PA_OUTCOMES)}


# ---------------------------------------------------------------------------
# Sensitivity checks (useful for debugging — not part of public API)
# ---------------------------------------------------------------------------

def log5_sensitivity(batter_rate: float, league_rate: float, n_pts: int = 9) -> np.ndarray:
    """
    Return log5 outputs as pitcher_rate sweeps from 0 to 1 (useful for sanity-checking).
    When batter_rate == league_rate the output should always equal league_rate (identity).
    """
    pitcher_rates = np.linspace(0.01, 0.99, n_pts)
    return np.array([log5_binary(batter_rate, pr, league_rate) for pr in pitcher_rates])
