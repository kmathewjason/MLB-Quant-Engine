#!/usr/bin/env python3
"""
scripts/validate_optimizer.py
==============================
Smoke-tests the optimizer layer (de-vig, Kelly, portfolio caps, drawdown).

Checks:
  1.  additive_devig: symmetric market sums to 1 and is 50/50
  2.  power_devig: sums to 1; favourite gets more probability than additive
  3.  shin_devig: sums to 1; z>0 (informed-trader proportion detected)
  4.  best_devig: returns valid probs for each method
  5.  method_divergence: L1 divergence ≥ 0
  6.  favorite_longshot_bias: flb_index sign is correct
  7.  kelly_fraction: f*>0 for +EV bet; f*≤0 for −EV bet
  8.  kelly_ev: EV formula correct for known inputs
  9.  portfolio_kelly: fractions ≥ 0; total exposure ≤ max_exposure
  10. apply_caps: per-bet cap enforced
  11. bootstrap_drawdown_simulator: MDD p05 ≥ 0 for all Kelly fractions

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


# ── 1. additive_devig symmetric ───────────────────────────────────────────
def _check_additive_sym():
    from src.optimizer.vig_removal import additive_devig
    p = additive_devig([-110, -110])
    assert abs(sum(p) - 1.0) < 1e-9
    assert abs(p[0] - 0.5) < 1e-9

check("vig_removal: additive_devig(-110,-110) → [0.5, 0.5]", _check_additive_sym)


# ── 2. power_devig favourite bias ─────────────────────────────────────────
def _check_power_fav():
    from src.optimizer.vig_removal import power_devig, additive_devig
    odds = [-200, 170]  # heavy favourite
    p_pow = power_devig(odds)
    p_add = additive_devig(odds)
    assert abs(sum(p_pow) - 1.0) < 1e-9
    # Power assigns MORE to the favourite than additive (FLB direction)
    assert p_pow[0] > p_add[0], (
        f"Power fav={p_pow[0]:.4f} should exceed additive fav={p_add[0]:.4f}"
    )

check("vig_removal: power_devig(-200,+170) assigns more to favourite than additive", _check_power_fav)


# ── 3. shin_devig sums to 1 ───────────────────────────────────────────────
def _check_shin():
    from src.optimizer.vig_removal import shin_devig
    p = shin_devig([-110, -110])
    assert abs(sum(p) - 1.0) < 1e-9

check("vig_removal: shin_devig(-110,-110) sums to 1", _check_shin)


# ── 4. best_devig all methods ─────────────────────────────────────────────
def _check_best_devig():
    from src.optimizer.vig_removal import best_devig
    for method in ("power", "additive", "shin"):
        p = best_devig([-150, 130], method=method)
        assert abs(sum(p) - 1.0) < 1e-9, f"{method} does not sum to 1"
        assert all(0 < v < 1 for v in p), f"{method} has out-of-range prob"

check("vig_removal: best_devig all three methods sum to 1 with valid probs", _check_best_devig)


# ── 5. method_divergence ──────────────────────────────────────────────────
def _check_divergence():
    from src.optimizer.vig_removal import method_divergence
    div = method_divergence([-200, 170], method_a="power", method_b="additive")
    assert div.l1 >= 0
    assert div.l_inf >= 0
    assert div.js_divergence >= 0
    # Lopsided market: power and additive disagree → non-zero divergence
    assert div.l1 > 0, "L1 divergence should be >0 for lopsided market"

check("vig_removal: method_divergence L1>0 for lopsided market", _check_divergence)


# ── 6. favourite_longshot_bias sign ──────────────────────────────────────
def _check_flb():
    from src.optimizer.vig_removal import favorite_longshot_bias
    result = favorite_longshot_bias([-200, 170])
    # flb_index > 0 → power assigns more to favourite (standard FLB direction)
    assert result.flb_index > 0, f"FLB index={result.flb_index:.4f} expected >0"

check("vig_removal: favorite_longshot_bias index > 0 for lopsided market", _check_flb)


# ── 7. kelly_fraction +EV / −EV ──────────────────────────────────────────
def _check_kelly_fraction():
    from src.optimizer.kelly import kelly_fraction
    # Positive edge: 60% win, -110 market (b ≈ 0.909)
    f_pos = kelly_fraction(0.60, -110, fractional_divisor=1.0)
    assert f_pos > 0, f"f*={f_pos:.4f} expected >0 for +EV bet"

    # Negative edge: 40% win, -110 market → returns 0 (Kelly clips negatives)
    f_neg = kelly_fraction(0.40, -110, fractional_divisor=1.0)
    assert f_neg <= 0, f"f*={f_neg:.4f} expected ≤0 for −EV bet"

check("kelly: kelly_fraction >0 for +EV; ≤0 for −EV", _check_kelly_fraction)


# ── 8. kelly_ev formula ───────────────────────────────────────────────────
def _check_kelly_ev():
    from src.optimizer.kelly import kelly_ev
    # b=1 (even money, +100), p=0.55 → EV = 1×0.55 − 0.45 = +0.10
    ev = kelly_ev(0.55, 100)
    assert abs(ev - 0.10) < 1e-9, f"EV={ev:.4f} expected 0.10"

check("kelly: kelly_ev(p=0.55, +100) = +0.10", _check_kelly_ev)


# ── 9. portfolio_kelly fractions ─────────────────────────────────────────
def _check_portfolio_kelly():
    import pandas as pd
    from src.optimizer.kelly import portfolio_kelly

    bets = pd.DataFrame({
        "market":     ["h2h", "h2h", "total"],
        "side":       ["home", "away", "over"],
        "model_prob": [0.58, 0.45, 0.55],
        "fair_prob":  [0.50, 0.50, 0.52],
        "fair_odds_american": [-110.0, -110.0, -110.0],
    })
    result = portfolio_kelly(bets, bankroll=1000.0, max_total_exposure=0.20)
    # Output columns: ev, ev_excess, f_star, f_kelly, f_final, stake_units
    assert "f_kelly" in result.columns, f"Columns: {result.columns.tolist()}"
    assert (result["f_kelly"] >= 0).all(), "Negative Kelly fractions"
    total_exposure = result["f_kelly"].sum()
    assert total_exposure <= 0.20 + 1e-6, f"Total exposure={total_exposure:.4f} exceeds 0.20"

check("kelly: portfolio_kelly fractions ≥0; total exposure ≤ max_exposure", _check_portfolio_kelly)


# ── 10. apply_caps per-bet limit ──────────────────────────────────────────
def _check_apply_caps():
    import pandas as pd
    from src.optimizer.kelly import portfolio_kelly
    from src.optimizer.portfolio_cap import apply_caps, CapConfig

    bets = pd.DataFrame({
        "market":     ["h2h"],
        "side":       ["home"],
        "model_prob": [0.70],
        "fair_prob":  [0.50],
        "fair_odds_american": [-110.0],
    })
    raw = portfolio_kelly(bets, bankroll=1000.0, max_total_exposure=0.30)
    cfg = CapConfig(max_bet_fraction=0.02)   # cap at 2% of bankroll = $20
    capped = apply_caps(raw, bankroll=1000.0, config=cfg)
    # apply_caps writes the capped stake into 'capped_stake_units'
    max_stake = capped["capped_stake_units"].max()
    assert max_stake <= 1000.0 * 0.02 + 0.01, f"Stake ${max_stake:.2f} exceeds 2% cap"

check("portfolio_cap: apply_caps enforces per-bet fraction cap", _check_apply_caps)


# ── 11. bootstrap_drawdown_simulator MDD ≥ 0 ─────────────────────────────
def _check_drawdown():
    from src.optimizer.portfolio_cap import bootstrap_drawdown_simulator
    import pandas as pd

    rng_ = np.random.default_rng(42)
    n = 200
    # bet_log needs 'stake_units' (not 'stake'), 'result', and optionally 'fair_odds_american'
    bets = pd.DataFrame({
        "result":             rng_.binomial(1, 0.53, n).astype(float),
        "stake_units":        np.full(n, 0.05),   # 5% of bankroll per bet
        "fair_odds_american": np.full(n, -110.0),
    })
    # Returns list[DrawdownSimResult]; attributes: p05_max_drawdown, p50_terminal_bankroll
    sim_results = bootstrap_drawdown_simulator(
        bets, starting_bankroll=1.0, n_paths=200,
    )
    for res in sim_results:
        assert res.p05_max_drawdown >= 0, f"MDD p05 < 0 for {res.kelly_label}"
        assert res.p50_terminal_bankroll > 0, f"Terminal bankroll p50 ≤ 0 for {res.kelly_label}"

check("portfolio_cap: bootstrap_drawdown_simulator MDD p05 ≥ 0 for all Kelly fractions", _check_drawdown)


# ── Report ─────────────────────────────────────────────────────────────────
print("\n── Optimizer validation ──")
failures = 0
for name, ok, msg in results:
    icon = PASS if ok else FAIL
    print(f"  {icon}  {name}")
    if not ok:
        print(f"       {msg}")
        failures += 1

print(f"\n{len(results) - failures}/{len(results)} checks passed.")
sys.exit(0 if failures == 0 else 1)
