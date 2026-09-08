#!/usr/bin/env python3
"""
scripts/validate_models.py
============================
Smoke-tests the models layer (Glicko-2, RE24 matrix, simulator, PA model).

Checks:
  1. Glicko-2 update is monotone in performance (better perf → higher μ)
  2. Glicko-2 RD increases when no games are played (ratings decay)
  3. RE24 matrix: 24 entries, all finite, state-0 RE > 0
  4. simulate_half_inning: returns non-negative integer runs
  5. _simulate_half_inning_variable: home wins more than away when home PA probs are stronger
  6. _simulate_half_inning_variable: over prob at total=0 ≈ 1; at total=99 ≈ 0
  7. PA outcome model: FEATURE_NAMES has 29 entries; OUTCOME_LABELS has 7
  8. Ensemble calibration: blend_weight property is float in [0, 1]

Exit 0 = all pass.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np

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


# ── 1. Glicko-2 monotone in performance ──────────────────────────────────
def _check_glicko_monotone():
    from src.models.pitcher_elo import Glicko2System, GameResult

    sys_high = Glicko2System()
    sys_mid  = Glicko2System()
    sys_low  = Glicko2System()

    # Register an opponent in each system with default rating
    for s in (sys_high, sys_mid, sys_low):
        s.get("opp")     # initialise with defaults

    sys_high.update([GameResult("pitcher", "opp", 0.95)])
    sys_mid.update( [GameResult("pitcher", "opp", 0.50)])
    sys_low.update( [GameResult("pitcher", "opp", 0.05)])

    mu_high = sys_high.get("pitcher").mu
    mu_mid  = sys_mid.get("pitcher").mu
    mu_low  = sys_low.get("pitcher").mu

    assert mu_high > mu_mid > mu_low, (
        f"μ not monotone: high={mu_high:.3f} mid={mu_mid:.3f} low={mu_low:.3f}"
    )

check("pitcher_elo: Glicko-2 μ is monotone in performance score", _check_glicko_monotone)


# ── 2. Glicko-2 RD increases with inactivity ─────────────────────────────
def _check_glicko_rd_decay():
    from src.models.pitcher_elo import Glicko2System, GlickoRating

    sys_ = Glicko2System()
    # Give the pitcher a well-established (low RD) rating by seeding directly
    r_before = GlickoRating(mu=0.0, phi=0.3, sigma=0.06, n_games=50)
    sys_.set("pitcher", r_before)

    # _update_one with no opponents → only φ grows (idle period RD expansion)
    r_after = sys_._update_one("pitcher", [], [])
    assert r_after.phi > r_before.phi, (
        f"RD did not increase: before={r_before.phi:.4f} after={r_after.phi:.4f}"
    )

check("pitcher_elo: Glicko-2 RD increases after period with no games", _check_glicko_rd_decay)


# ── 3. RE24 matrix shape and values ───────────────────────────────────────
def _check_re24():
    import math
    from src.models.markov_re_matrix import build_re_matrix_from_default_table

    re = build_re_matrix_from_default_table()
    assert re.re_values.shape == (24,), f"Expected shape (24,), got {re.re_values.shape}"
    assert all(math.isfinite(v) for v in re.re_values), "RE matrix contains non-finite values"
    # State 0 = 0 outs, bases empty — should have positive RE
    assert re.re_values[0] > 0, f"RE[0]={re.re_values[0]:.3f} expected > 0"
    # encode_state(0, 2) = bases-empty, 2 outs — should be lower than 0-out state
    from src.models.markov_re_matrix import encode_state
    s2 = encode_state(0, 2)
    assert re.re_values[s2] < re.re_values[0], (
        f"RE at 2-outs-bases-empty (state {s2}={re.re_values[s2]:.3f}) "
        f"should be < RE at 0-outs-bases-empty ({re.re_values[0]:.3f})"
    )

check("markov_re_matrix: build_re_matrix returns 24 finite values, RE[0]>0", _check_re24)


# ── 4. simulate_half_inning returns non-negative runs ────────────────────
def _check_half_inning():
    from src.models.markov_re_matrix import simulate_half_inning
    from src.features.matchup_features import MLB_LEAGUE_AVG as lg

    # simulate_half_inning expects a 7-element prob vector matching Markov outcomes
    # (K, BB, 1B, 2B, 3B, HR, OUT)
    pa_probs = np.array([
        lg["K"], lg["BB"] + lg.get("HBP", 0.01),
        lg["1B"], lg["2B"], lg["3B"], lg["HR"],
        lg["OUT"],
    ], dtype=np.float64)
    pa_probs /= pa_probs.sum()

    rng = np.random.default_rng(42)
    runs = simulate_half_inning(pa_probs, rng=rng, n_innings=1000)
    assert isinstance(runs, np.ndarray)
    assert runs.shape == (1000,)
    assert (runs >= 0).all(), "Negative runs detected"

check("markov_re_matrix: simulate_half_inning returns non-negative runs (1000 innings)", _check_half_inning)


# ── 5. _simulate_half_inning_variable: stronger lineup scores more runs ────
def _check_variable_half_inning_probs():
    from src.models.game_simulator import _simulate_half_inning_variable  # noqa: PLC2701
    from src.features.matchup_features import MLB_LEAGUE_AVG as lg

    rng = np.random.default_rng(0)
    n_sims = 5_000

    # Build a (9, 7) slot_probs array for an average lineup
    avg_row = np.array([
        lg["K"], lg["BB"] + lg.get("HBP", 0.01),
        lg["1B"], lg["2B"], lg["3B"], lg["HR"],
        lg["OUT"],
    ], dtype=np.float64)
    avg_row /= avg_row.sum()

    # Strong lineup: shift run-scoring outcomes up 30%
    strong_row = avg_row.copy()
    strong_row[2:6] *= 1.30   # 1B, 2B, 3B, HR
    strong_row /= strong_row.sum()

    avg_lineup    = np.tile(avg_row,    (9, 1))
    strong_lineup = np.tile(strong_row, (9, 1))

    runs_avg,    _ = _simulate_half_inning_variable(avg_lineup,    n_sims, 0, rng)
    runs_strong, _ = _simulate_half_inning_variable(strong_lineup, n_sims, 0, rng)

    assert runs_strong.mean() > runs_avg.mean(), (
        f"Strong lineup mean={runs_strong.mean():.3f} not > avg mean={runs_avg.mean():.3f}"
    )

check(
    "game_simulator: stronger lineup scores more runs on average (5000 half-innings)",
    _check_variable_half_inning_probs,
)


# ── 6. Over/under probability boundary conditions ─────────────────────────
def _check_over_probs():
    from src.models.game_simulator import _simulate_half_inning_variable, GameSimResult  # noqa: PLC2701
    from src.features.matchup_features import MLB_LEAGUE_AVG as lg

    rng = np.random.default_rng(1)
    n_sims = 5_000

    avg_row = np.array([
        lg["K"], lg["BB"] + lg.get("HBP", 0.01),
        lg["1B"], lg["2B"], lg["3B"], lg["HR"],
        lg["OUT"],
    ], dtype=np.float64)
    avg_row /= avg_row.sum()
    lineup = np.tile(avg_row, (9, 1))

    # Simulate 9 half-innings per side
    home_runs = np.zeros(n_sims, dtype=np.int32)
    away_runs = np.zeros(n_sims, dtype=np.int32)
    slot = 0
    for _ in range(9):
        h, _ = _simulate_half_inning_variable(lineup, n_sims, slot, rng)
        a, _ = _simulate_half_inning_variable(lineup, n_sims, slot, rng)
        home_runs += h
        away_runs += a

    result = GameSimResult(home_runs_dist=home_runs, away_runs_dist=away_runs)

    # Over 0 runs ≈ 1 (almost all games score at least 1 run)
    over0 = result.total_over_prob(0.0)
    assert over0 > 0.95, f"Over-0 prob={over0:.3f}, expected >0.95"

    # Over 99 runs ≈ 0
    over99 = result.total_over_prob(99.0)
    assert over99 < 0.01, f"Over-99 prob={over99:.3f}, expected <0.01"

check(
    "game_simulator: total_over_prob(0)>0.95 and total_over_prob(99)<0.01",
    _check_over_probs,
)


# ── 7. PA outcome model: feature schema ───────────────────────────────────
def _check_pa_model_features():
    from src.models.pa_outcome_model import FEATURE_NAMES, OUTCOME_LABELS
    assert len(FEATURE_NAMES) == 29, f"Expected 29 features, got {len(FEATURE_NAMES)}"
    assert len(OUTCOME_LABELS) == 7, f"Expected 7 outcome labels, got {len(OUTCOME_LABELS)}"

check("pa_outcome_model: FEATURE_NAMES has 29 entries; OUTCOME_LABELS has 7", _check_pa_model_features)


# ── 8. Stacked ensemble: blend_weight is float in [0, 1] ──────────────────
def _check_ensemble_blend_weight():
    from src.models.ensemble import StackedEnsemble
    ens = StackedEnsemble()
    bw = ens.blend_weight
    assert isinstance(bw, float), f"blend_weight should be float, got {type(bw)}"
    assert 0.0 <= bw <= 1.0, f"blend_weight={bw} not in [0, 1]"

check("ensemble: StackedEnsemble.blend_weight is float in [0, 1]", _check_ensemble_blend_weight)


# ── Report ─────────────────────────────────────────────────────────────────
print("\n── Models validation ──")
failures = 0
for name, ok, msg in results:
    icon = PASS if ok else FAIL
    print(f"  {icon}  {name}")
    if not ok:
        print(f"       {msg}")
        failures += 1

print(f"\n{len(results) - failures}/{len(results)} checks passed.")
sys.exit(0 if failures == 0 else 1)
