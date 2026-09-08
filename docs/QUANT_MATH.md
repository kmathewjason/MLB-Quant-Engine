# QUANT_MATH.md — MLB Quant Engine Mathematical Reference

This document consolidates every significant mathematical model used in the
engine.  It is intentionally self-contained: a reader who understands
probability and basic statistics should be able to derive every formula here
without looking at the code.

---

## Table of Contents

1. [Empirical Bayes / Bayesian Shrinkage](#1-empirical-bayes--bayesian-shrinkage)
2. [Log5 Matchup Formula](#2-log5-matchup-formula)
3. [24-State Run Expectancy Matrix (RE24)](#3-24-state-run-expectancy-matrix-re24)
4. [Glicko-2 Pitcher Rating](#4-glicko-2-pitcher-rating)
5. [Park & Weather Adjustments](#5-park--weather-adjustments)
6. [PA Outcome Model (Multinomial Classifier)](#6-pa-outcome-model-multinomial-classifier)
7. [Monte Carlo Game Simulation](#7-monte-carlo-game-simulation)
8. [De-Vig: Power, Additive, and Shin Methods](#8-de-vig-power-additive-and-shin-methods)
9. [Murphy Brier Score Decomposition](#9-murphy-brier-score-decomposition)
10. [Expected Calibration Error (ECE)](#10-expected-calibration-error-ece)
11. [Closing-Line Value (CLV)](#11-closing-line-value-clv)
12. [Kelly Criterion & Covariance-Adjusted Portfolio Kelly](#12-kelly-criterion--covariance-adjusted-portfolio-kelly)
13. [Walk-Forward Temporal Validation](#13-walk-forward-temporal-validation)
14. [Negative Binomial Dispersion Check](#14-negative-binomial-dispersion-check)

---

## 1. Empirical Bayes / Bayesian Shrinkage

**File:** `src/features/bayesian_shrinkage.py`

### Motivation

Small samples produce extreme observed rates.  A batter who goes 5-for-10 on
HRs has an "observed" HR rate of 50%, which is absurd.  Bayesian shrinkage
pulls extreme estimates back toward the population mean.

### Beta Prior via Method of Moments

For a binomial event with population mean `μ` and variance `σ²`:

```
α = μ · κ
β = (1 − μ) · κ

where  κ = μ(1−μ)/σ² − 1  (concentration parameter)
```

`κ` encodes the amount of prior belief.  Tango's empirically derived values:

| Statistic | κ (PAs equivalent) |
|-----------|-------------------|
| BA        | 1,200             |
| OBP       | 1,500             |
| SLG       | 1,100             |
| HR/PA     | 1,200             |
| K%        | 60                |
| BB%       | 120               |

### Posterior Mean (Shrinkage)

Given `x` successes in `n` trials:

```
posterior_mean = (α + x) / (α + β + n)
               = (μ·κ + x) / (κ + n)
```

This is the weighted average of the prior mean and the observed rate,
weighted by `κ` vs `n`.  As `n → ∞`, the posterior approaches the observed
rate.  As `n → 0`, the posterior approaches the prior mean `μ`.

---

## 2. Log5 Matchup Formula

**File:** `src/features/matchup_features.py`

### Binary Log5 (Bill James 1983)

Given batter event probability `p`, pitcher event probability `q`, and
league-average event probability `lg`:

```
log5(p, q, lg) = (p · q / lg) / (p·q/lg + (1−p)·(1−q)/(1−lg))
```

Derivation: model outcomes as draws from independent Bernoulli variables on a
common log-odds scale.  The formula satisfies the boundary conditions:
- `log5(lg, q, lg) = q`   (league-average batter takes on pitcher's rate)
- `log5(p, lg, lg) = p`   (league-average pitcher takes on batter's rate)
- `log5(lg, lg, lg) = lg` (both league-average → league average)

### Multinomial Odds-Ratio (multi-outcome)

For an outcome set `k ∈ {1B, 2B, 3B, HR, BB, K, out}`, we extend log5 via
the odds-ratio model:

```
OR_k = (p_k / p_other) × (q_k / q_other) × (lg_other / lg_k)

p_matchup_k ∝ OR_k
```

where `p_other = 1 − p_k` (binary other) and the results are normalised to
sum to 1.

---

## 3. 24-State Run Expectancy Matrix (RE24)

**File:** `src/models/markov_re_matrix.py`

### State Space

A base-out state is fully described by:
- **3 base occupancy bits**: (1B occupied?, 2B occupied?, 3B occupied?) → 8 combinations
- **Outs**: 0, 1, or 2 → 3 combinations

Total states: `8 × 3 = 24`.  End-of-inning (3 outs) is absorbing.

State encoding: `state = outs × 8 + base_bits` where `base_bits` is the
3-bit integer (bit 0 = 1B, bit 1 = 2B, bit 2 = 3B).

### Run Expectancy via Linear System

Let `RE[s]` be the expected runs scored from state `s` to the end of the
inning.  Define the transition matrix `T[s, s']` = probability of moving from
state `s` to state `s'` on one PA, and the immediate runs vector `r[s, s']`.

The linear system:

```
RE = r̄ + T · RE

(I − T) · RE = r̄

RE = (I − T)⁻¹ · r̄
```

where `r̄[s] = Σ_{s'} T[s,s'] · r[s,s']` is the expected immediate runs
scored on the next PA.

### Advance Table

Each PA outcome (single, double, triple, HR, BB, K, out) maps deterministically
(with fixed base-running assumptions) to a new base-out state and immediate runs.
The `_ADVANCE_TABLE` in the code encodes these transitions as a flat array of
shape `(24_states × 7_outcomes × 2)` for `(next_state, immediate_runs)`.

### RE24 Contribution

The value of a plate appearance outcome is:

```
RE24 = RE[new_state] + runs_scored − RE[old_state]
```

Positive RE24 means the PA helped the offense more than the average outcome
from that state.

---

## 4. Glicko-2 Pitcher Rating

**File:** `src/models/pitcher_elo.py`

### Why Glicko-2 over Elo

Elo assigns equal uncertainty to all players.  Glicko-2 adds a *ratings
deviation* (RD) representing uncertainty about the true rating, and a
*volatility* σ representing how much the RD changes over time (accounts for
pitchers being genuinely inconsistent vs. merely unlucky).

### State Variables

Each pitcher carries `(μ, φ, σ)`:
- `μ` — rating on the Glicko-2 internal scale (≈ Elo / 173.7478)
- `φ` — ratings deviation (uncertainty); 0 = perfect knowledge
- `σ` — volatility; reflects the pitcher's inconsistency

### Update Equations

For a pitcher with rating `μ`, RD `φ`, and volatility `σ`, facing opponents
with ratings `{μⱼ}` and RDs `{φⱼ}`:

**Step 1** — Compute `g(φⱼ)` and `E(μ, μⱼ, φⱼ)`:

```
g(φ) = 1 / sqrt(1 + 3φ² / π²)

E(μ, μⱼ, φⱼ) = 1 / (1 + exp(−g(φⱼ) · (μ − μⱼ)))
```

**Step 2** — Estimated variance `v` of the ratings based on game outcomes:

```
v = [Σⱼ g(φⱼ)² · E(μ,μⱼ,φⱼ) · (1 − E(μ,μⱼ,φⱼ))]⁻¹
```

**Step 3** — Score deviation `Δ`:

```
Δ = v · Σⱼ g(φⱼ) · (sⱼ − E(μ,μⱼ,φⱼ))
```

**Step 4** — New volatility σ′ via Illinois algorithm (bisection on the cubic):

```
f(x) = [exp(x)(Δ²−φ²−v−exp(x))] / [2(φ²+v+exp(x))²]  −  (x−ln σ²)/τ²
```

Solve `f(x) = 0` for `x`; then `σ′ = exp(x/2)`.

**Step 5** — New RD and rating:

```
φ★ = sqrt(φ² + σ′²)
φ′ = 1 / sqrt(1/φ★² + 1/v)
μ′ = μ + φ′² · Σⱼ g(φⱼ) · (sⱼ − E(μ,μⱼ,φⱼ))
```

### Performance Score

The `pitcher_performance_score()` function computes a per-start `sⱼ ∈ [0,1]`
from FIP-components (K, BB, HR, IP):

```
performance = (K_rate × w_K − BB_rate × w_BB − HR_rate × w_HR + IP_bonus) / scale
```

Clipped to `[0, 1]` to stay within the Glicko-2 outcome range.

---

## 5. Park & Weather Adjustments

**File:** `src/features/park_weather.py`

### Air Density

Air density ρ affects ball carry.  Using the ideal gas law with Magnus
vapour-pressure correction for humidity:

```
p_vapour = 0.6112 × exp(17.67 × T / (T + 243.5))   [kPa; T in °C]

ρ = (p_station − humidity × p_vapour) / (R_d × T_K)
  + (humidity × p_vapour) / (R_v × T_K)
```

where `R_d = 287.058 J/(kg·K)`, `R_v = 461.495 J/(kg·K)`.

### Wind Decomposition

Wind speed `W` at bearing `θ_wind` relative to CF orientation `θ_CF`:

```
θ_rel = θ_wind − θ_CF

W_out  = W · cos(θ_rel)   [positive = blowing toward CF → helps HRs]
W_cross = W · sin(θ_rel)  [crosswind]
```

### Park Factor

Run park factor from a symmetric home/away split:

```
PF = (home_RS + home_RA) / (home_G)
   ÷ (road_RS + road_RA) / (road_G)
```

EWMA update (decay `d = 0.5`):

```
PF_t = d × PF_observed_t + (1 − d) × PF_{t-1}
```

### Environmental Scalar

The `EnvFactor` dataclass combines park factor, air density (relative to
standard 1.2041 kg/m³), and wind-out component into a single `run_scalar`:

```
run_scalar = park_factor × (1.2041 / air_density) × (1 + 0.01 × wind_out_mph)
```

---

## 6. PA Outcome Model (Multinomial Classifier)

**File:** `src/models/pa_outcome_model.py`

### Outcome Classes

Seven discrete PA outcomes are modelled:

| Index | Outcome       |
|-------|---------------|
| 0     | Single (1B)   |
| 1     | Double (2B)   |
| 2     | Triple (3B)   |
| 3     | Home run (HR) |
| 4     | Walk (BB)     |
| 5     | Strikeout (K) |
| 6     | Other out     |

### Feature Vector (29 features)

```
batter_features     (12): BA_shr, OBP_shr, SLG_shr, HR_rate_shr, K_rate_shr,
                           BB_rate_shr, wOBA_shr, ISO_shr, BABIP_shr,
                           pull_rate, oppo_rate, line_drive_rate

pitcher_features    (11): ERA_shr, FIP_shr, K9_shr, BB9_shr, HR9_shr,
                           WHIP_shr, GB_rate_shr, glicko_mu, glicko_phi,
                           glicko_sigma, days_rest

matchup_features    (4):  log5_xBA, log5_xOBP, log5_xSLG, log5_xHR

context             (2):  is_home (batter), inning

TTO flag            (1):  times_through_order (1, 2, or 3)
```

All rate features are Bayesian-shrunk before entry.

### Ensemble

Two base learners:

1. **XGBoost** (multinomial softmax, `n_estimators=400`, `max_depth=5`,
   `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`)
2. **MLP** (3 hidden layers: 256→128→64, ReLU, dropout=0.3, batch norm)

Blend weight `α ∈ [0,1]` is optimised on the calibration split to minimise
cross-entropy:

```
p_blend = α × p_xgb + (1 − α) × p_mlp
```

Isotonic calibration is applied per class (one-vs-rest) on the calibration
set using monotone regression.

### Chronological Split

```
|←────── 70% train ──────→|← 15% val →|← 8% cal →|← 7% test →|
                                                               time →
```

No shuffling.  The val split is used for early stopping; cal split for blend
weight and isotonic calibration; test split for final evaluation.

---

## 7. Monte Carlo Game Simulation

**File:** `src/models/game_simulator.py`

### Vectorised Architecture

For `N` simulations, all 9 × 9 innings × batters are processed as NumPy
array operations — no Python loop per at-bat.

Pre-computed PA probability matrix shape: `(n_innings, 9_batters, 7_outcomes)`.

### Half-Inning Simulation

For one half-inning across all `N` simulations simultaneously:

1. **Batter index** rotates via modulo from lineup position (0–8)
2. For each batter slot, draw PA outcomes via `np.searchsorted` on the
   cumulative probability array (inverse-CDF sampling)
3. Look up `(next_base_state, immediate_runs)` from the 24-state flat table
4. Accumulate runs; increment outs
5. Terminate when outs ≥ 3 (active simulations mask)

### Bottom-of-9th Logic

The bottom-of-9th is simulated only for games where the home team is trailing
or tied after 8.5 innings.  If home takes the lead mid-inning, the simulation
marks those games as complete (walk-off).

### Extra Innings (Rule 2023+)

After 9 full innings, games tied go to extras with the automatic runner (ghost
runner on 2B).  The starting base state is `0b010` (runner on second, 0 outs)
= state index `8`.

### Output: `GameSimResult`

```python
@dataclass
class GameSimResult:
    home_runs:     np.ndarray  # (N,) final home run totals
    away_runs:     np.ndarray  # (N,) final away run totals
    home_wins:     np.ndarray  # (N,) bool

    win_prob_home: float       # mean(home_wins)
    win_prob_away: float       # 1 - win_prob_home
    mean_home_runs: float
    mean_away_runs: float
```

Derived quantities:
```python
spread_cover_prob(line)  = mean(home_runs - away_runs > line)
total_over_prob(ou_line) = mean(home_runs + away_runs > ou_line)
```

---

## 8. De-Vig: Power, Additive, and Shin Methods

**File:** `src/optimizer/vig_removal.py`

### From American Odds to Raw Implied Probability

```
american ≥ 0:  p_raw = 100 / (american + 100)
american < 0:  p_raw = |american| / (|american| + 100)
```

The raw probs sum to `overround = 1 + vig > 1`.

### Additive (Proportional) De-Vig

```
p_fair_i = p_raw_i / Σⱼ p_raw_j
```

Assumes the vig is distributed in proportion to the raw implied probabilities.
Biased toward favourites in lopsided markets.

### Power Method (Smoczynski & Tomczyk 2010)

Find exponent `k > 1` such that `Σᵢ p_raw_i^k = 1`:

```
f(k) = Σᵢ p_raw_i^k − 1 = 0

p_fair_i = p_raw_i^k   (then re-normalised for floating-point)
```

Solved via Brent's method.  The power method preserves the log-odds ratio
between outcomes and is empirically more accurate than additive in lopsided
markets.

### Shin Model (Shin 1992; Jullien & Salanié 1994)

Accounts for the presence of informed traders (proportion `z`):

```
p_fair_i = [sqrt(z² + 4(1−z)·qᵢ·p_raw_i) − z] / [2(1−z)]
```

where `qᵢ = p_raw_i / Σⱼ p_raw_j` (additive normalisation).

`z` is found by solving `Σᵢ p_fair_i = 1` numerically (Brent on `[0, 0.999]`).
At `z = 0` the Shin formula reduces to the additive method.

---

## 9. Murphy Brier Score Decomposition

**File:** `src/backtest/calibration.py`

The **Brier Score** measures probability forecast accuracy:

```
BS = (1/N) Σᵢ Σₖ (pᵢₖ − 1{yᵢ=k})²
```

Murphy (1973) decomposes it into three meaningful terms:

```
BS = REL − RES + UNC
```

**Uncertainty (UNC)** — fixed by the data; cannot be improved:
```
UNC = o̅(1 − o̅)      where o̅ = base rate of the positive class
```

**Reliability (REL)** — calibration error; lower is better:
```
REL = (1/N) Σₖ nₖ (f̄ₖ − ōₖ)²
```

where `f̄ₖ` is the mean forecast probability in bin `k`, `ōₖ` is the mean
observed frequency in bin `k`, and `nₖ` is the bin count.

**Resolution (RES)** — discrimination; higher is better:
```
RES = (1/N) Σₖ nₖ (ōₖ − o̅)²
```

**Brier Skill Score (BSS)**:
```
BSS = 1 − BS / UNC
```

`BSS > 0` means the model beats the naive climatological predictor.

For multi-class, each term is computed per-class (one-vs-rest) and averaged.

---

## 10. Expected Calibration Error (ECE)

**File:** `src/backtest/calibration.py`, `src/models/ensemble.py`

ECE measures how much predicted probabilities deviate from empirical frequencies:

```
ECE = Σₖ (nₖ/N) |f̄ₖ − ōₖ|
```

over equal-width probability bins `[0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]`.

A perfectly calibrated model has `ECE = 0`.
A model that always predicts 0.9 when the true rate is 0.5 has `ECE = 0.4`.

Bootstrap 95% CI on `ōₖ` is computed by resampling within each bin 1,000 times.

---

## 11. Closing-Line Value (CLV)

**File:** `src/backtest/clv_tracker.py`

### Motivation

Short-run P&L is dominated by variance.  CLV measures whether the model
found lines that were more accurate than the market's final consensus,
independent of results.

### Per-Bet CLV (Probability Space)

Both the opening line (what we bet) and the closing line (final market price)
are de-vigged using the power method:

```
clv_prob = p_fair_close − p_fair_open
```

Positive CLV → we were smarter than the market at bet-placement time.

### CLV in Log-Odds Space (Scale-Invariant)

```
clv_log_odds = logit(p_fair_close) − logit(p_fair_open)
             = log(p_close/(1−p_close)) − log(p_open/(1−p_open))
```

A +0.1 log-odds CLV means the same thing whether the price is near 50% or 90%.
This is the primary aggregation metric.

### Statistical Tests

**t-test** (H0: mean CLV = 0):
```
t = mean(clv) / (std(clv) / sqrt(N))
```

**CLV-result correlation** (Pearson r between CLV and binary win):
Positive correlation distinguishes genuine edge from variance.
A bettor with real edge will have their CLV predict their wins better than chance.

---

## 12. Kelly Criterion & Covariance-Adjusted Portfolio Kelly

**File:** `src/optimizer/kelly.py`

### Single-Bet Kelly

For a bet with win probability `p`, net payout `b` (decimal odds − 1),
`q = 1 − p`:

```
f* = (b·p − q) / b
```

This maximises the long-run geometric growth rate `E[log W]` (Kelly 1956).

**Fractional Kelly** reduces variance at the cost of slightly lower growth:
```
f = f* / divisor   (default divisor = 4)
```

Half-Kelly cuts variance by ~75% and growth by ~25% (relative to full Kelly).

### No-Edge Case

`f* ≤ 0` when `b·p < q`, i.e., when the expected value `EV = b·p − q ≤ 0`.
Never bet when EV ≤ 0.

### Covariance-Adjusted Portfolio Kelly

For `n` simultaneous bets with edge vector `μ` and bet-outcome covariance matrix `Σ`:

```
E[log W] ≈ f·μ − ½ f·Σ·fᵀ
```

The unconstrained optimum is `f* = Σ⁻¹·μ`.  We solve the constrained version:

```
maximise   f·μ − ½ f·Σ·fᵀ
subject to f ≥ 0,  Σᵢ fᵢ ≤ max_exposure
```

via SLSQP (scipy.optimize.minimize).  The `max_exposure` constraint prevents
over-betting when many positive-EV correlated opportunities exist simultaneously.

---

## 13. Walk-Forward Temporal Validation

**File:** `src/backtest/walk_forward.py`

### Design Invariant

> For every prediction row `r`, `date[r]` > `max(date[training rows])` for
> that fold.  No information from the future leaks into any training fold.

This is strictly stronger than a random train/test split and is mandatory
for time-series data.

### Expanding Window

```
|───── initial window ────|─ fold 1 ─|─ fold 2 ─|─ fold 3 ─|...
                          ↑ T₀      T₀+step    T₀+2·step
```

Each fold trains on **all** data before the fold's start date.  This allows
the model to learn from an ever-growing sample.

### Metrics

Per-fold and aggregate:
- **Log-loss**: `−(1/N) Σᵢ log P(yᵢ | xᵢ)`
- **Accuracy**: `(1/N) Σᵢ 1{ŷᵢ = yᵢ}`
- **Brier score**: `(1/N) Σᵢ Σₖ (pᵢₖ − 1{yᵢ=k})²`

---

## 14. Negative Binomial Dispersion Check

**File:** `src/backtest/run_dispersion_check.py`

### Motivation

Runs-per-game is *overdispersed* count data (`Var[Y] > E[Y]`).  The Poisson
model (`Var = μ`) is mis-specified.  We use the **NB2 parameterisation**:

```
Var[Y] = μ + α·μ²    where α ≥ 0 is the dispersion parameter
```

A linear regression for `log μ`:

```
log μ = β₀ + β₁·is_home + β₂·park_factor + ...
```

### Dispersion Ratio

For each game, compare Monte Carlo simulator variance to the NB-predicted
variance:

```
dispersion_ratio = Var_MC / Var_NB
```

**Flag the game** if `dispersion_ratio < 0.60` — the simulator captures less
than 60% of the expected real-world variance.

Likely causes of underdispersion:
- Missing within-game momentum structure
- Constant lineup (no platoon splits, no bullpen transitions)
- PA outcomes treated as i.i.d. within an inning (ignores clustering)

---

## 15. Favourite-Longshot Bias (FLB) Detection

**File:** `src/optimizer/vig_removal.py` — `favorite_longshot_bias()`

### What FLB Is

In betting markets, implied probabilities for large favourites tend to be
*overstated* relative to their true frequency, while longshots are
*understated*.  This means naive additive de-vig over-prices favourites.

### Detection: Power vs Additive Divergence

Define the power-method probability `p_pow_fav` and additive probability
`p_add_fav` for the favourite side of a two-way market:

```
flb_index = (p_pow_fav − p_add_fav) / vig_pct
```

where `vig_pct = overround − 1` (e.g., 0.045 for a 4.5% book).

**Interpretation:**
- `flb_index > 0` → power assigns more probability to the favourite than
  additive does, consistent with standard FLB.  The two methods disagree most
  when the market is lopsided.
- `flb_index ≈ 0` → near-50/50 market; both methods agree.

This index is dimensionless and market-size-neutral: a 3-point and a 30-point
favourite with the same vig percentage produce comparable indices.

### Why This Matters

If you use additive de-vig on a heavily-juiced favourite (e.g., −250 / +200)
you will systematically underestimate the favourite's true probability and
overestimate the longshot's — the opposite of real-world FLB.  The power
method is preferred because it preserves the log-odds ratio structure and
empirically matches actual win frequencies better on lopsided markets.

---

## 16. Same-Game Parlay: Empirical Correlation Model

**File:** `src/optimizer/kelly.py` — `simulate_corr_matrix()`
**File:** `src/api.py` — `POST /api/predictions/sgp`

### Why Independence Fails for SGPs

A same-game parlay (SGP) bundles multiple legs from the same game.  The legs
are **not** independent: home moneyline and over total are positively correlated
(high-scoring games help the home team in baseball because they bat last), and
home ML and away ML are almost perfectly negatively correlated (one wins iff
the other loses).

Using the naive product `Π pᵢ` as the joint probability systematically
misprices SGPs.

### Simulation-Based Joint Probability

Run `N` Monte Carlo game simulations using the standard half-inning engine.
For each leg `i` and simulation `j`, record the binary outcome `Xᵢⱼ ∈ {0,1}`:

```
moneyline leg: Xᵢⱼ = 1 if home wins sim j  (or away wins, depending on side)
over/under leg: Xᵢⱼ = 1 if total > line in sim j
spread leg:    Xᵢⱼ = 1 if home − away > run_line in sim j
```

The **empirical correlation matrix** is:

```
ρᵢₖ = Cov(Xᵢ, Xₖ) / sqrt(Var(Xᵢ) · Var(Xₖ))
     = (mean(Xᵢ · Xₖ) − mean(Xᵢ)·mean(Xₖ)) / ...
```

computed directly from the simulation outcomes (`numpy.corrcoef`).

The **correlation-adjusted joint probability** is the empirical frequency
of all legs winning simultaneously:

```
p_joint_corr = mean(min_i(Xᵢⱼ) = 1)  for j = 1…N
             = fraction of sims where every leg fires
```

This is strictly more accurate than `Π pᵢ` because it captures the full
dependence structure, not just pairwise correlations.

### Same-Game Deduplication

Legs from the same game share the same simulation draws (`_game_key`
deduplication in the code ensures this).  A two-leg SGP on game `G` uses
the same `N` simulation paths for both legs — only one simulation is run
per unique game, no matter how many legs reference it.

### Kelly Sizing for the Parlay

Treat the SGP as a single binary bet with:

```
p  = p_joint_corr       (correlation-adjusted)
b  = Π dᵢ − 1          (net payout; dᵢ = decimal odds per leg)
```

Standard single-bet Kelly applies:

```
f* = (b·p − q) / b
f  = f* / 4            (quarter Kelly)
```

The naive comparison uses `p_naive = Π pᵢ` in place of `p_joint_corr`.
The **percentage difference** `(f_corr − f_naive) / f_naive` quantifies
the dollar value of the correlation model — visible in the SGP Builder tab.

### Interpretation

| Leg combination | ρ sign | Corr adj vs naive |
|---|---|---|
| Home ML + Over total | positive | corr stake > naive |
| Home ML + Away ML | ≈ −1 | joint prob ≈ 0 (can't both win) |
| Two totals (e.g., both O8.5 and O9.5) | positive | corr stake > naive |
| Home ML + Under total | negative | corr stake < naive |

---

## 17. Bootstrap Drawdown Simulator

**File:** `src/optimizer/portfolio_cap.py` — `bootstrap_drawdown_simulator()`

### Motivation

Maximum drawdown (MDD) cannot be computed analytically for correlated,
non-normal bet outcomes.  We use parametric bootstrap to build a distribution
over MDD and terminal bankroll under different Kelly fractions.

### Algorithm

Given `n_bets` bet records with outcomes `yᵢ ∈ {0,1}` and stakes `sᵢ`:

1. Draw `n_paths × n_bets` bootstrap samples (with replacement) from
   the bet population.
2. For each Kelly fraction `f_scale ∈ {1, 0.5, 0.25, 0.125}` (full, half,
   quarter, eighth):
   - Scale all stakes: `sⱼ = f_scale × s_original`
   - Compute running P&L per path:  `W_t = W_0 + Σⱼ sⱼ · (yⱼ · bⱼ − (1−yⱼ))`
   - Maximum drawdown per path: `MDD_path = max_t (max_{t'≤t} W_{t'} − W_t)`

3. Report `p05`, `p50`, `p95` of the MDD distribution and `p05`, `p50`,
   `mean` of the terminal bankroll distribution.

### Vectorised Implementation

The bootstrap is fully NumPy-vectorised:

```python
# shape: (n_paths, n_bets)
idx = rng.integers(0, n_bets, size=(n_paths, n_bets))
outcomes = y[idx]          # (n_paths, n_bets)
payouts  = payout[idx]     # (n_paths, n_bets)

# P&L per bet: +payout if win, −stake if loss
pnl = np.where(outcomes == 1, payouts, -stakes[idx])

# Running sum along bet axis
running = np.cumsum(pnl, axis=1)

# MDD from cummax
cummax = np.maximum.accumulate(running, axis=1)
mdd    = (cummax - running).max(axis=1)
```

No Python loop over individual paths — all `n_paths` are computed in a
single vectorised pass.

### Interpretation

The MDD `p05` at full Kelly is the worst-case drawdown you should plan for.
If `p05 MDD at full Kelly > max_drawdown_cap` (default 20%), the portfolio
cap kicks in and scales stakes down until the expected MDD is acceptable.

---

## Summary: Data Flow

```
Raw data (Statcast, MLB API, Odds API, Weather)
    ↓  ingestion/
Parquet files (raw/)
    ↓  features/
Feature matrices (processed/)    ← Bayesian shrinkage + log5 + park/weather
    ↓  models/
PA outcome probabilities          ← XGBoost + MLP ensemble + isotonic calibration
    ↓  models/game_simulator.py
Game win probabilities            ← Monte Carlo (50k sims, vectorised NumPy)
    ↓  optimizer/
Fair odds (de-vigged)             ← Power / Shin method
Kelly fractions                   ← Covariance-adjusted portfolio Kelly
Capped stakes                     ← Per-bet + team + daily + drawdown guardrails
SGP joint probability             ← Empirical correlation (simulation-based)
    ↓  backtest/
Walk-forward OOS metrics          ← Temporal split, no data leakage
CLV tracking                      ← Closing-line value vs. placed-line value
FLB detection                     ← Power vs additive divergence index
HTML report                       ← run_report.py
    ↓  api.py / dashboard
REST API (127.0.0.1:8000)
React dashboard (127.0.0.1:5173)
```

---

*Last updated: Phase 9 (SGP correlation model, FLB detection, drawdown bootstrap)*
