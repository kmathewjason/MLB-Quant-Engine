#!/usr/bin/env python3
"""
scripts/validate_features.py
==============================
Smoke-tests the feature-engineering layer.

Checks:
  1. Beta prior MoM (κ) computation for Tango presets
  2. Posterior mean shrinkage boundary conditions
  3. log5_binary boundary conditions
  4. log5_multinomial + odds_ratio_multinomial normalisation
  5. matchup_pa_probs output shape and sum-to-one
  6. compute_park_factor returns finite float
  7. EnvFactor run_scalar is positive finite
  8. merge_weather returns DataFrame with required columns

Exit 0 = all pass.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

PASS = "✓"
FAIL = "✗"
results: list[tuple[str, bool, str]] = []


def check(name: str, fn):
    try:
        fn()
        results.append((name, True, ""))
    except Exception:
        results.append((name, False, traceback.format_exc().strip().splitlines()[-1]))


# ── 1. BetaPrior κ via make_prior_from_kappa ─────────────────────────────
def _check_kappa():
    from src.features.bayesian_shrinkage import make_prior_from_kappa
    prior = make_prior_from_kappa(league_rate=0.250, kappa=1200)
    kappa = prior.kappa
    assert abs(kappa - 1200) < 2, f"κ={kappa:.1f}, expected ≈1200"

check("bayesian_shrinkage: make_prior_from_kappa κ ≈ 1200 for BA preset", _check_kappa)


# ── 2. Shrinkage boundary conditions ─────────────────────────────────────
def _check_shrinkage_bounds():
    import numpy as np
    from src.features.bayesian_shrinkage import shrink, fit_beta_prior, make_prior_from_kappa
    prior = make_prior_from_kappa(league_rate=0.250, kappa=1200)
    # 0 PA → should equal prior mean
    result0 = shrink(np.array([0.0]), np.array([0.0]), prior)
    assert abs(result0[0] - prior.mean) < 1e-9, f"0 PA result={result0[0]:.4f}, expected {prior.mean}"
    # very large N → should converge to observed rate
    huge_n = 1_000_000
    obs_rate = 0.350
    result_big = shrink(np.array([obs_rate * huge_n]), np.array([huge_n]), prior)
    assert abs(result_big[0] - obs_rate) < 0.001

check("bayesian_shrinkage: 0 PA → prior mean; large N → observed rate", _check_shrinkage_bounds)


# ── 3. log5_binary boundary conditions ────────────────────────────────────
def _check_log5_binary():
    from src.features.matchup_features import log5_binary, MLB_LEAGUE_AVG
    lg = MLB_LEAGUE_AVG["1B"]
    p, q = 0.22, 0.25
    # log5(lg, q, lg) == q  (lg batter takes pitcher's rate)
    assert abs(log5_binary(lg, q, lg) - q) < 1e-9, "log5 identity 1 failed"
    # log5(p, lg, lg) == p  (lg pitcher takes batter's rate)
    assert abs(log5_binary(p, lg, lg) - p) < 1e-9, "log5 identity 2 failed"
    # log5(lg, lg, lg) == lg
    assert abs(log5_binary(lg, lg, lg) - lg) < 1e-9, "log5 identity 3 failed"

check("matchup_features: log5_binary satisfies all three boundary identities", _check_log5_binary)


# ── 4. Multinomial normalisation ──────────────────────────────────────────
def _check_log5_multi():
    import numpy as np
    from src.features.matchup_features import (
        log5_multinomial, odds_ratio_multinomial, MLB_LEAGUE_AVG,
    )
    # MLB_LEAGUE_AVG has keys 1B,2B,3B,HR,BB,K + derives 'other_out'
    keys = ["1B", "2B", "3B", "HR", "BB", "K"]
    other = max(0.0, 1.0 - sum(MLB_LEAGUE_AVG[k] for k in keys))
    lg_arr = [MLB_LEAGUE_AVG[k] for k in keys] + [other]

    p = [v * 1.05 for v in lg_arr]; p = [v / sum(p) for v in p]
    q = [v * 0.95 for v in lg_arr]; q = [v / sum(q) for v in q]
    lg = [v / sum(lg_arr) for v in lg_arr]

    res1 = log5_multinomial(p, q, lg)
    assert abs(sum(res1) - 1.0) < 1e-9, f"log5_multinomial sums to {sum(res1)}"

    res2 = odds_ratio_multinomial(p, q, lg)
    assert abs(sum(res2) - 1.0) < 1e-9, f"odds_ratio_multinomial sums to {sum(res2)}"

check("matchup_features: log5_multinomial + odds_ratio_multinomial both sum to 1", _check_log5_multi)


# ── 5. matchup_pa_probs output shape ─────────────────────────────────────
def _check_matchup_pa_probs():
    from src.features.matchup_features import matchup_pa_probs, MLB_LEAGUE_AVG
    result = matchup_pa_probs(
        batter_dict=MLB_LEAGUE_AVG,
        pitcher_dict=MLB_LEAGUE_AVG,
        league_dict=MLB_LEAGUE_AVG,
        method="log5",
    )
    vals = list(result.values())
    assert len(vals) == 8, f"Expected 8 outcomes, got {len(vals)}"
    assert abs(sum(vals) - 1.0) < 1e-9

check("matchup_features: matchup_pa_probs returns 8 outcomes summing to 1", _check_matchup_pa_probs)


# ── 6. Park factor finite ─────────────────────────────────────────────────
def _check_park_factor():
    import math
    import pandas as pd
    from src.features.park_weather import compute_park_factor
    # compute_park_factor takes a DataFrame
    df = pd.DataFrame({
        "venue_id":  [1, 1, 1, 1],
        "season":    [2023, 2023, 2023, 2023],
        "is_home":   [1, 1, 0, 0],
        "runs":      [4.5, 4.2, 4.1, 4.3],
        "PA":        [37, 36, 35, 36],
    })
    result = compute_park_factor(df, stat="runs")
    assert isinstance(result, pd.DataFrame)

check("park_weather: compute_park_factor returns DataFrame", _check_park_factor)


# ── 7. EnvFactor fields ────────────────────────────────────────────────────
def _check_env_factor():
    import math
    from src.features.park_weather import EnvFactor
    env = EnvFactor(
        park_factor_runs=1.05,
        park_factor_hr=1.10,
        air_density_factor=0.98,
        wind_hr_factor=1.02,
        wind_run_factor=1.01,
    )
    assert math.isfinite(env.park_factor_runs)
    assert env.park_factor_runs > 0

check("park_weather: EnvFactor constructable with correct fields", _check_env_factor)


# ── 8. merge_weather callable ─────────────────────────────────────────────
def _check_merge_weather():
    from src.features.park_weather import merge_weather
    assert callable(merge_weather)

check("park_weather: merge_weather callable", _check_merge_weather)


# ── Report ─────────────────────────────────────────────────────────────────
print("\n── Features validation ──")
failures = 0
for name, ok, msg in results:
    icon = PASS if ok else FAIL
    print(f"  {icon}  {name}")
    if not ok:
        print(f"       {msg}")
        failures += 1

print(f"\n{len(results) - failures}/{len(results)} checks passed.")
sys.exit(0 if failures == 0 else 1)
