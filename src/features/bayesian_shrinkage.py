"""
features.bayesian_shrinkage
============================
Empirical Bayes shrinkage for binary-rate statistics (K%, BB%, HR/FB, etc.)
using a Beta prior whose parameters are estimated from the league-wide
distribution of observed rates.

Statistical justification
--------------------------
A batter's true talent rate θ is modelled as drawn from a Beta(α, β)
prior across the population.  After observing *x* events in *n* trials,
the posterior is Beta(α + x, β + n - x) and the posterior mean is:

    θ̂ = (x + α) / (n + α + β)
      = (n · x̄ + (α + β) · μ₀) / (n + α + β)

where μ₀ = α / (α + β) is the prior mean (≈ league average) and
α + β = κ is the *effective prior sample size* (stabilisation point).

α and β are estimated from the full league-wide distribution of
per-player rates using *method of moments* on the Beta distribution:

    μ̂  = mean(x̄ᵢ)
    σ̂² = var(x̄ᵢ)

    α = μ̂ · (μ̂(1-μ̂)/σ̂² - 1)
    β = (1-μ̂) · (μ̂(1-μ̂)/σ̂² - 1)

Equivalently:  κ = α + β = μ̂(1-μ̂)/σ̂² - 1

This matches Tango's empirical stabilisation-point framework: κ is the
PA count at which observed rate and shrunk estimate have equal weight.

Public API
----------
fit_beta_prior(rates)                      -> BetaPrior
shrink(numerators, denominators, prior)    -> pd.Series   (shrunk rates)
shrink_dataframe(df, prior)               -> pd.DataFrame (adds columns)
stabilisation_point(prior)                -> float
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class BetaPrior(NamedTuple):
    """Fitted Beta distribution parameters."""
    alpha: float   # shape parameter α  (successes + prior "pseudo-successes")
    beta: float    # shape parameter β  (failures  + prior "pseudo-failures")

    @property
    def mean(self) -> float:
        """Prior mean = league-average rate."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def kappa(self) -> float:
        """Effective prior sample size (stabilisation point)."""
        return self.alpha + self.beta

    @property
    def variance(self) -> float:
        k = self.kappa
        m = self.mean
        return m * (1 - m) / (k + 1)


# ---------------------------------------------------------------------------
# Prior fitting via method of moments
# ---------------------------------------------------------------------------

def fit_beta_prior(
    rates: np.ndarray | pd.Series,
    min_var: float = 1e-9,
) -> BetaPrior:
    """
    Fit a Beta(α, β) prior to a population of observed rates via MoM.

    Parameters
    ----------
    rates   : 1-D array of per-player observed rates (x/n); should be
              filtered to players with at least a minimum sample (e.g. 50 PA)
              so extreme small-sample noise doesn't bias the prior.
    min_var : floor on sample variance to avoid division-by-zero when all
              players have the same observed rate.

    Returns
    -------
    BetaPrior(alpha, beta)

    Raises
    ------
    ValueError  if fewer than 2 valid rate observations are provided.
    """
    arr = np.asarray(rates, dtype=float)
    arr = arr[(arr >= 0) & (arr <= 1) & np.isfinite(arr)]
    if arr.size < 2:
        raise ValueError(
            f"Need at least 2 valid rates to fit a Beta prior; got {arr.size}."
        )

    mu = float(arr.mean())
    sigma2 = float(max(arr.var(ddof=1), min_var))

    # MoM estimates
    common = mu * (1.0 - mu) / sigma2 - 1.0

    if common <= 0:
        # Degenerate: observed variance ≥ theoretical max → near-flat prior
        # Fall back to a weakly informative α=β=1 (uniform) prior
        alpha = max(mu * 2.0, 0.5)
        beta = max((1.0 - mu) * 2.0, 0.5)
    else:
        alpha = mu * common
        beta = (1.0 - mu) * common

    if alpha <= 0 or beta <= 0:
        raise ValueError(
            f"Degenerate Beta prior: α={alpha:.4f}, β={beta:.4f}. "
            "Check that rates are in (0, 1) and have reasonable variance."
        )
    return BetaPrior(alpha=alpha, beta=beta)


# ---------------------------------------------------------------------------
# Shrinkage
# ---------------------------------------------------------------------------

def shrink(
    numerators: np.ndarray | pd.Series,
    denominators: np.ndarray | pd.Series,
    prior: BetaPrior,
) -> np.ndarray:
    """
    Return shrunk rate estimates for arrays of (numerator, denominator) pairs.

    Posterior mean:  θ̂ᵢ = (xᵢ + α) / (nᵢ + α + β)

    Players with nᵢ = 0 receive the prior mean.

    Parameters
    ----------
    numerators   : success counts (e.g. strikeouts)
    denominators : trial counts   (e.g. plate appearances)
    prior        : BetaPrior fitted by fit_beta_prior()

    Returns
    -------
    np.ndarray of float64 shrunk rates, same length as inputs.
    """
    x = np.asarray(numerators, dtype=float)
    n = np.asarray(denominators, dtype=float)
    return (x + prior.alpha) / (n + prior.kappa)


def shrink_dataframe(
    df: pd.DataFrame,
    prior: BetaPrior,
    numerator_col: str = "stat_numerator",
    denominator_col: str = "stat_denominator",
    player_col: str = "player_id",
) -> pd.DataFrame:
    """
    Add shrinkage columns to *df*.

    Expects columns: [player_id, stat_numerator, stat_denominator].

    Adds columns:
        observed_rate   — raw x/n  (NaN when n=0)
        shrunk_rate     — posterior mean
        eff_sample_size — nᵢ + κ   (total weight behind estimate)
        shrinkage_weight — κ / (nᵢ + κ)  — fraction pulled toward prior
    """
    out = df[[player_col, numerator_col, denominator_col]].copy()
    x = out[numerator_col].to_numpy(dtype=float)
    n = out[denominator_col].to_numpy(dtype=float)

    out["observed_rate"] = np.where(n > 0, x / n, np.nan)
    out["shrunk_rate"] = shrink(x, n, prior)
    out["eff_sample_size"] = n + prior.kappa
    out["shrinkage_weight"] = prior.kappa / (n + prior.kappa)
    return out


# ---------------------------------------------------------------------------
# Convenience: built-in stabilisation-point presets (Tango/MGL estimates)
# ---------------------------------------------------------------------------

# These are starting-point κ values for manual use without population data.
# Prefer fitting from data when a sufficient population sample is available.
TANGO_KAPPA: dict[str, float] = {
    "K_pct":      60.0,
    "BB_pct":    120.0,
    "HBP_pct":   250.0,
    "HR_per_FB": 300.0,
    "BABIP":     820.0,
    "AVG":       460.0,
    "OBP":       320.0,
    "SLG":       320.0,
    "wOBA":      460.0,
}


def make_prior_from_kappa(league_rate: float, kappa: float) -> BetaPrior:
    """
    Construct a BetaPrior from a known stabilisation point κ and league rate μ.

    Useful when you want to use Tango's published κ directly rather than
    fitting from a population.

        α = μ · κ
        β = (1 − μ) · κ
    """
    if not (0 < league_rate < 1):
        raise ValueError(f"league_rate must be in (0,1); got {league_rate}")
    if kappa <= 0:
        raise ValueError(f"kappa must be > 0; got {kappa}")
    return BetaPrior(alpha=league_rate * kappa, beta=(1.0 - league_rate) * kappa)


def stabilisation_point(prior: BetaPrior) -> float:
    """Return κ = α + β, the PA count at which observed and shrunk estimates carry equal weight."""
    return prior.kappa
