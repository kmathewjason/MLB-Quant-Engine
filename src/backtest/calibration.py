"""
backtest.calibration
====================
Probability calibration diagnostics:

1. Reliability diagrams  — predicted probability decile vs actual frequency,
   with bootstrap confidence intervals.

2. Murphy (1973) three-term Brier Score decomposition:

       BS = REL - RES + UNC

   where:
       UNC = o̅(1 - o̅)                          — base-rate uncertainty
       RES = (1/N) Σ_k n_k (ō_k - o̅)²          — resolution (how much the
                                                    model's bins differ from base rate)
       REL = (1/N) Σ_k n_k (f̄_k - ō_k)²        — reliability (calibration error;
                                                    how much bin mean forecast
                                                    differs from bin frequency)

   Interpretation:
       REL  → 0 is good (well-calibrated model)
       RES  → large is good (model makes confident and correct distinctions)
       UNC  → fixed by the data; cannot be improved

   Multi-class extension: compute per-class binary decomposition (one-vs-rest)
   and average across classes.

3. Brier skill score vs climatology (uninformed prior):
       BSS = 1 - BS / BS_climate
   BSS > 0 means the model beats a naive predictor.

Public API
----------
reliability_diagram(probs, y, n_bins, n_bootstrap)  -> ReliabilityResult
brier_decomposition(probs, y, n_bins)               -> BrierDecomp
brier_skill_score(probs, y)                         -> float
calibration_report(probs, y, class_names)           -> pd.DataFrame
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ReliabilityResult:
    """
    Output of reliability_diagram().

    Attributes
    ----------
    bins_df     : DataFrame with per-bin statistics (one row per bin)
                  columns: bin_lo, bin_hi, bin_centre, mean_pred, obs_freq,
                           count, ci_lo_95, ci_hi_95
    class_idx   : which class was analysed
    ece         : expected calibration error for this class
    n_bootstrap : number of bootstrap resamples used for CIs
    """
    bins_df: pd.DataFrame
    class_idx: int
    ece: float
    n_bootstrap: int


@dataclass
class BrierDecomp:
    """
    Murphy (1973) Brier Score decomposition: BS = REL - RES + UNC.

    Fields are averaged over all classes (one-vs-rest).
    """
    brier_score: float
    reliability: float   # REL — lower is better (calibration error)
    resolution:  float   # RES — higher is better (discrimination)
    uncertainty: float   # UNC — fixed by base rate; uncontrollable
    brier_skill_score: float   # BSS = 1 - BS / UNC;  > 0 beats climatology

    def to_dict(self) -> dict:
        return {
            "brier_score":       self.brier_score,
            "reliability":       self.reliability,
            "resolution":        self.resolution,
            "uncertainty":       self.uncertainty,
            "brier_skill_score": self.brier_skill_score,
        }


# ---------------------------------------------------------------------------
# Reliability diagram
# ---------------------------------------------------------------------------

def reliability_diagram(
    probs: np.ndarray,
    y: np.ndarray,
    class_idx: int = 0,
    n_bins: int = 10,
    n_bootstrap: int = 1000,
    rng: np.random.Generator | None = None,
) -> ReliabilityResult:
    """
    Build reliability diagram data for *class_idx* with bootstrap CIs.

    Parameters
    ----------
    probs       : (N, K) probability matrix (rows sum to 1)
    y           : (N,) integer true class labels
    class_idx   : which class to treat as the positive outcome
    n_bins      : number of equal-width probability bins (default 10)
    n_bootstrap : bootstrap resamples for 95% CI on empirical frequency
    rng         : numpy Generator; seeded internally if None

    Returns
    -------
    ReliabilityResult with bins_df and ECE.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    probs = np.asarray(probs, dtype=np.float64)
    y     = np.asarray(y, dtype=np.int32)
    p     = probs[:, class_idx]
    obs   = (y == class_idx).astype(float)
    n     = len(p)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    ece  = 0.0

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (p >= lo) & (p < hi)
        cnt  = int(mask.sum())
        if cnt == 0:
            continue

        mean_pred = float(p[mask].mean())
        obs_freq  = float(obs[mask].mean())
        ece      += (cnt / n) * abs(mean_pred - obs_freq)

        # Bootstrap CI on empirical frequency
        obs_bin = obs[mask]
        boot_freqs = np.array(
            [rng.choice(obs_bin, size=cnt, replace=True).mean() for _ in range(n_bootstrap)]
        )
        ci_lo = float(np.percentile(boot_freqs, 2.5))
        ci_hi = float(np.percentile(boot_freqs, 97.5))

        rows.append({
            "bin_lo":      lo,
            "bin_hi":      hi,
            "bin_centre":  (lo + hi) / 2.0,
            "mean_pred":   mean_pred,
            "obs_freq":    obs_freq,
            "count":       cnt,
            "ci_lo_95":    ci_lo,
            "ci_hi_95":    ci_hi,
        })

    bins_df = pd.DataFrame(rows)
    return ReliabilityResult(
        bins_df=bins_df,
        class_idx=class_idx,
        ece=ece,
        n_bootstrap=n_bootstrap,
    )


def reliability_diagram_all_classes(
    probs: np.ndarray,
    y: np.ndarray,
    class_names: Sequence[str] | None = None,
    n_bins: int = 10,
    n_bootstrap: int = 500,
) -> list[ReliabilityResult]:
    """
    Run reliability_diagram() for every class and return list of results.
    """
    n_classes = probs.shape[1]
    return [
        reliability_diagram(probs, y, class_idx=k, n_bins=n_bins, n_bootstrap=n_bootstrap)
        for k in range(n_classes)
    ]


# ---------------------------------------------------------------------------
# Murphy Brier decomposition
# ---------------------------------------------------------------------------

def _binary_brier_decomp(p: np.ndarray, obs: np.ndarray, n_bins: int) -> dict:
    """
    Murphy (1973) decomposition for a single binary problem.

    Returns dict with keys: bs, rel, res, unc
    """
    n = len(p)
    o_bar = float(obs.mean())
    bs    = float(np.mean((p - obs) ** 2))
    unc   = o_bar * (1.0 - o_bar)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    rel = 0.0
    res = 0.0

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (p >= lo) & (p < hi)
        cnt  = int(mask.sum())
        if cnt == 0:
            continue
        f_bar_k = float(p[mask].mean())
        o_bar_k = float(obs[mask].mean())
        rel += (cnt / n) * (f_bar_k - o_bar_k) ** 2
        res += (cnt / n) * (o_bar_k - o_bar) ** 2

    return {"bs": bs, "rel": rel, "res": res, "unc": unc}


def brier_decomposition(
    probs: np.ndarray,
    y: np.ndarray,
    n_bins: int = 10,
) -> BrierDecomp:
    """
    Murphy (1973) three-term Brier Score decomposition, averaged over
    all classes using the one-vs-rest scheme.

    Parameters
    ----------
    probs   : (N, K) probability matrix
    y       : (N,) integer true class labels
    n_bins  : bins for the decomposition (default 10)

    Returns
    -------
    BrierDecomp with BS = REL - RES + UNC relationship verified.
    """
    probs = np.asarray(probs, dtype=np.float64)
    y     = np.asarray(y, dtype=np.int32)
    n, k  = probs.shape

    # Full multi-class Brier score
    targets = np.zeros_like(probs)
    targets[np.arange(n), y] = 1.0
    bs_full = float(np.mean(((probs - targets) ** 2).sum(axis=1)))

    # Per-class binary decomposition
    rel_vals, res_vals, unc_vals = [], [], []
    for cls in range(k):
        d = _binary_brier_decomp(probs[:, cls], (y == cls).astype(float), n_bins)
        rel_vals.append(d["rel"])
        res_vals.append(d["res"])
        unc_vals.append(d["unc"])

    rel = float(np.mean(rel_vals))
    res = float(np.mean(res_vals))
    unc = float(np.mean(unc_vals))
    # BSS = 1 - BS / UNC_climate  (where UNC is the baseline Brier score)
    bss = float(1.0 - bs_full / unc) if unc > 1e-12 else 0.0

    return BrierDecomp(
        brier_score=bs_full,
        reliability=rel,
        resolution=res,
        uncertainty=unc,
        brier_skill_score=bss,
    )


def brier_skill_score(probs: np.ndarray, y: np.ndarray) -> float:
    """
    Brier Skill Score vs climatological baseline.

    BSS = 1 - BS / BS_climate
        where BS_climate = UNC = o̅(1-o̅) for each class, averaged.

    Returns float in (-∞, 1]; positive means better than climatology.
    """
    return brier_decomposition(probs, y).brier_skill_score


# ---------------------------------------------------------------------------
# Calibration report table
# ---------------------------------------------------------------------------

def calibration_report(
    probs: np.ndarray,
    y: np.ndarray,
    class_names: Sequence[str] | None = None,
    n_bins: int = 10,
    n_bootstrap: int = 500,
) -> pd.DataFrame:
    """
    Produce a tidy per-class calibration summary DataFrame.

    Columns: class_name, ece, brier_score, reliability, resolution,
             uncertainty, brier_skill_score, base_rate, n_positive

    Parameters
    ----------
    probs       : (N, K) probability matrix
    y           : (N,) true labels
    class_names : optional list of K class names
    n_bins      : bins for ECE and Murphy decomposition

    Returns
    -------
    pd.DataFrame with one row per class, sorted by ECE ascending.
    """
    probs = np.asarray(probs, dtype=np.float64)
    y     = np.asarray(y, dtype=np.int32)
    n, k  = probs.shape

    if class_names is None:
        class_names = [str(i) for i in range(k)]

    rows = []
    for cls in range(k):
        obs = (y == cls).astype(float)
        base_rate = float(obs.mean())

        # ECE
        rel_result = reliability_diagram(probs, y, class_idx=cls, n_bins=n_bins,
                                         n_bootstrap=n_bootstrap)

        # Per-class binary Brier
        d = _binary_brier_decomp(probs[:, cls], obs, n_bins)

        rows.append({
            "class_name":        class_names[cls],
            "ece":               round(rel_result.ece, 5),
            "brier_score":       round(d["bs"], 5),
            "reliability":       round(d["rel"], 5),
            "resolution":        round(d["res"], 5),
            "uncertainty":       round(d["unc"], 5),
            "brier_skill_score": round(1.0 - d["bs"] / d["unc"] if d["unc"] > 1e-12 else 0.0, 5),
            "base_rate":         round(base_rate, 4),
            "n_positive":        int(obs.sum()),
        })

    return pd.DataFrame(rows).sort_values("ece").reset_index(drop=True)
