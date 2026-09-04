# MLB Quant Engine — Quantitative Methods Reference

> This document is built out incrementally as each phase is implemented.
> Every formula here corresponds directly to code in `src/`.

---

## Phase 2 — Feature Engineering

### Log5 Matchup Probability (Bill James)

Given a batter's observed rate `b`, pitcher's rate `p`, and league average `lg`,
the expected matchup rate is:

```
P(event | batter, pitcher) = (b · p / lg) / [ (b · p / lg) + ((1-b) · (1-p) / (1-lg)) ]
```

**Implementation:** [`src/features/matchup_features.py`](../src/features/matchup_features.py)

### Empirical Bayes Shrinkage

For a player with `PA` plate appearances and observed rate `x`:

```
shrunk = (PA · x + k · prior) / (PA + k)
```

`k` is the stabilisation point — the sample size at which observed rate and
true-talent estimate have equal weight. Key values (Tango):

| Stat | k (PA) |
|------|--------|
| K% | 60 |
| BB% | 120 |
| HR/FB | 300 |
| BABIP | 820 |

**Implementation:** [`src/features/bayesian_shrinkage.py`](../src/features/bayesian_shrinkage.py)

---

## Phase 3 — Markov Run Expectancy & Simulation

### 24-State Base-Out RE Matrix

Baseball inning state = (base occupancy, outs). There are 8 base
configurations × 3 out states = **24 states**.

The RE matrix `RE[s]` gives expected runs from state `s` to end of inning.
Solved via linear system from historical transition counts:

```
RE = (I - T_transient)^{-1} · r
```

where `T_transient` is the sub-stochastic transition matrix (terminal
states removed) and `r` is the immediate-run vector.

**RE24 delta** for a PA outcome:
```
ΔRUN_EXPECTANCY = RE[post_state] - RE[pre_state] + runs_scored
```

**Implementation:** [`src/models/markov_re_matrix.py`](../src/models/markov_re_matrix.py)

### Monte Carlo Game Simulation

1. For each half-inning, sample PA outcomes from the calibrated
   multinomial model (`pa_outcome_model.predict_proba`).
2. Advance base-out state deterministically via lookup table.
3. Accumulate runs; terminate on out #3.
4. Repeat N = 10,000 iterations.
5. `P(home win) = count(home_runs > away_runs) / N`

**Implementation:** [`src/models/game_simulator.py`](../src/models/game_simulator.py)

### Glicko-2 Pitcher Rating

Extends Elo with rating deviation `RD` and volatility `σ`:

```
μ  = (R - 1500) / 173.7178
φ  = RD / 173.7178

E(s | μ, μ_j, φ_j) = 1 / (1 + exp(-g(φ_j) · (μ - μ_j)))
g(φ) = 1 / sqrt(1 + 3φ² / π²)
```

New `μ'` and `φ'` updated per Glicko-2 algorithm (Glickman 2012).

**Implementation:** [`src/models/pitcher_elo.py`](../src/models/pitcher_elo.py)

---

## Phase 4 — Bet Sizing & Vig Removal

### Power-Method De-vig

Given raw implied probabilities `p_1, p_2` (from American odds):

Find `k` such that `p_1^k + p_2^k = 1`, then `p_fair_i = p_i^k`.

**Implementation:** [`src/optimizer/vig_removal.py`](../src/optimizer/vig_removal.py)

### Kelly Criterion

```
f* = (b · p - q) / b
```

- `b` = decimal odds − 1
- `p` = model win probability
- `q` = 1 − p

**Fractional Kelly:** `f = f* / D` where D ≥ 2 (default 4) to account for
model uncertainty.

**Portfolio Kelly** (multi-bet):
Maximise `E[log(W)]` subject to correlation constraints via QP solver.

**Implementation:** [`src/optimizer/kelly.py`](../src/optimizer/kelly.py)

---

## Phase 5 — Backtesting

### Closing-Line Value (CLV)

```
CLV_bet = log(closing_odds / odds_taken)
```

Positive mean CLV indicates the model identifies edges before the market
closes. This is the primary evidence of a predictive edge, independent of
short-term results.

**Implementation:** [`src/backtest/clv_tracker.py`](../src/backtest/clv_tracker.py)

---

*Document last updated: Phase 0 scaffolding*
