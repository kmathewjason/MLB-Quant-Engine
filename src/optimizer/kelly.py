"""
optimizer.kelly
===============
Kelly criterion bet sizing — single-bet fractional Kelly and
covariance-adjusted portfolio Kelly.

Single-bet Kelly
----------------
For a bet with fair win-probability p and decimal odds b+1 (payout = b per unit
staked on a win):

    f* = (b·p - q) / b      where q = 1 - p

This maximises the long-run geometric growth rate E[log W].
We apply a fractional divisor (default 4 — "quarter-Kelly") to reduce variance
at the cost of slightly lower growth:

    f = f* / divisor

Convert American odds → decimal:
    american ≥ 0  →  decimal = american/100 + 1
    american < 0  →  decimal = 100/|american| + 1

Edge filter
-----------
Only size bets where model_prob > fair_prob (positive expected value).
kelly_fraction() returns 0.0 when there is no edge.

Covariance-adjusted portfolio Kelly
------------------------------------
When sizing multiple simultaneous bets, the correlated-Kelly solution
maximises:

    E[log W] ≈ f·μ - ½ f·Σ·fᵀ

where:
    μ_i = b_i·p_i - q_i      (per-unit edge for bet i)
    Σ    = covariance matrix of bet outcomes

The quadratic-programme solution (unconstrained):

    f* = Σ⁻¹ · μ

We project f* onto the feasible simplex {f ≥ 0, Σf_i ≤ max_total_exposure}
using scipy.optimize.minimize (SLSQP), which handles the inequality constraint.

Reference
---------
Kelly, J.L. (1956). "A New Interpretation of Information Rate."
MacLean, Thorp & Ziemba (2010). "The Kelly Capital Growth Investment Criterion."

Covariance matrix from simulator
----------------------------------
simulate_corr_matrix() runs the game simulator jointly across all games in
a same-day slate and extracts the empirical correlation / covariance matrix
from the joint win-indicator distributions:

    outcomes[i, sim] = 1{team_i wins simulation sim}

    Σ_ij = Cov(outcome_i, outcome_j)

Games on the same slate share weather, park, and lineup dependencies, so
treating them as independent (diagonal Σ) slightly over-bets correlated
combinations.  The simulator-derived Σ captures all structural correlations.

Public API
----------
kelly_fraction(model_prob, fair_odds_american, fractional_divisor)  -> float
portfolio_kelly(bets_df, cov_matrix, bankroll, max_total_exposure,
                fractional_divisor, risk_free_rate)                  -> pd.DataFrame
simulate_corr_matrix(games, n_sims, rng)                            -> SimCorrResult
american_to_decimal(american)                                        -> float
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Odds converters
# ---------------------------------------------------------------------------

def american_to_decimal(american: float) -> float:
    """
    Convert American odds to decimal odds.

    +150  →  2.50   (profit of 1.50 per unit staked)
    -110  →  1.909…
    """
    a = float(american)
    if a >= 100.0:
        return a / 100.0 + 1.0
    return 100.0 / abs(a) + 1.0


def decimal_to_american(decimal: float) -> float:
    """
    Convert decimal odds to American odds.

    2.50  →  +150
    1.909 →  -110
    """
    if decimal <= 1.0:
        raise ValueError(f"Decimal odds must be > 1.0; got {decimal}")
    if decimal >= 2.0:
        return (decimal - 1.0) * 100.0
    return -100.0 / (decimal - 1.0)


# ---------------------------------------------------------------------------
# Single-bet fractional Kelly
# ---------------------------------------------------------------------------

def kelly_fraction(
    model_prob: float,
    fair_odds_american: float,
    fractional_divisor: float = 4.0,
) -> float:
    """
    Compute the fractional Kelly stake as a fraction of bankroll.

    Parameters
    ----------
    model_prob          : model's estimated win probability for this bet
    fair_odds_american  : American odds at which we place the bet
                          (should be the line we get, already de-vigged
                           externally by the caller using vig_removal)
    fractional_divisor  : divisor applied to full-Kelly stake (default 4).
                          Use 1.0 for full Kelly (maximum growth, high variance).

    Returns
    -------
    float — fraction of bankroll to wager (0.0 if no edge or divisor ≤ 0).

    Notes
    -----
    Returns 0.0 when:
    - model_prob ≤ 0 or ≥ 1 (degenerate probability)
    - computed f* ≤ 0 (model has no edge at this price)
    - divisor ≤ 0
    """
    if fractional_divisor <= 0.0:
        return 0.0
    if not (0.0 < model_prob < 1.0):
        return 0.0

    b = american_to_decimal(fair_odds_american) - 1.0   # net profit per unit
    if b <= 0.0:
        return 0.0

    p = model_prob
    q = 1.0 - p
    f_star = (b * p - q) / b

    if f_star <= 0.0:
        return 0.0

    return float(f_star / fractional_divisor)


def kelly_ev(model_prob: float, fair_odds_american: float) -> float:
    """
    Expected value per unit staked.

    EV = b·p - q   where b = decimal - 1, p = model prob, q = 1-p.

    Positive EV is required before sizing; this function exposes the raw EV
    for logging and filtering.

    Returns float (may be negative — do NOT bet if EV ≤ 0).
    """
    if not (0.0 < model_prob < 1.0):
        return float("-inf")
    b = american_to_decimal(fair_odds_american) - 1.0
    return float(b * model_prob - (1.0 - model_prob))


# ---------------------------------------------------------------------------
# Portfolio covariance-adjusted Kelly
# ---------------------------------------------------------------------------

def portfolio_kelly(
    bets_df: pd.DataFrame,
    cov_matrix: np.ndarray | None = None,
    bankroll: float = 1.0,
    max_total_exposure: float = 0.25,
    fractional_divisor: float = 4.0,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """
    Covariance-adjusted Kelly sizing for a portfolio of simultaneous bets.

    Solves the constrained quadratic programme:

        maximise   f · μ_excess  −  ½ f · Σ · fᵀ
        subject to f ≥ 0,  sum(f) ≤ max_total_exposure

    where:
        μ_excess_i = b_i · p_i − q_i − risk_free_rate · b_i
                   = EV_i − risk_free_rate · b_i

    The risk_free_rate term follows the multivariate Kelly derivation
    (Thorp 1971; MacLean, Thorp & Ziemba 2010):

        f* = Σ⁻¹ · (μ − r · 1)

    where r is the risk-free rate (e.g. daily T-bill yield) and 1 is the
    all-ones vector.  At r = 0 (default) this reduces to the standard
    Kelly solution f* = Σ⁻¹ · μ.

    Practical note: for sports bets r is typically 0.0 or very small
    (overnight T-bill yield ≈ 0.013% per day).  Including it slightly
    reduces optimal stakes when the edge is small.

    Parameters
    ----------
    bets_df : DataFrame with (at minimum) columns:
              - model_prob         : float, model win probability
              - fair_odds_american : float, line we expect to take
              Optional: bet_id (preserved in output)

    cov_matrix : (n, n) covariance matrix of bet-outcome indicators.
                 If None → diagonal (independent bets; single-Kelly per bet).
                 Use simulate_corr_matrix() to derive Σ from the game simulator.

    bankroll           : total bankroll in currency units (default 1.0 = fractions)
    max_total_exposure : hard cap on total fraction of bankroll across all bets
    fractional_divisor : divisor applied to f* (default 4 = quarter-Kelly)
    risk_free_rate     : risk-free rate in the same units as EV (default 0.0).
                         For daily sports bets this is typically 0 or ~1e-4.

    Returns
    -------
    DataFrame — input bets_df plus extra columns:
        ev            : expected value per unit  (b·p − q)
        ev_excess     : EV minus risk-free component  (b·p − q − r·b)
        f_star        : unconstrained per-bet Kelly fraction
        f_kelly       : fractional Kelly after divisor
        f_final       : final stake fraction (after portfolio cap)
        stake_units   : stake in bankroll units (f_final * bankroll)
    """
    from scipy.optimize import minimize  # noqa: PLC0415

    n = len(bets_df)
    if n == 0:
        return bets_df.copy().assign(
            ev=[], ev_excess=[], f_star=[], f_kelly=[], f_final=[], stake_units=[]
        )

    probs = bets_df["model_prob"].to_numpy(dtype=np.float64)
    odds  = bets_df["fair_odds_american"].to_numpy(dtype=np.float64)

    # Net payout per unit (b = decimal - 1)
    b = np.array([american_to_decimal(o) - 1.0 for o in odds])
    q = 1.0 - probs

    # EV vector: μ_i = b_i * p_i − q_i
    mu = b * probs - q  # (n,)

    # Excess return: subtract risk-free component r * b_i (opportunity cost)
    mu_excess = mu - risk_free_rate * b  # (n,)

    # Single-Kelly vector: f* = μ_i / b_i  (unconstrained, per-bet)
    f_star_single = np.where(b > 0, mu / b, 0.0)
    f_star_single = np.maximum(f_star_single, 0.0)   # no short-selling

    if cov_matrix is None:
        # Diagonal case: independent bets
        # f*_i = max(μ_excess_i / b_i, 0)
        f_star_excess = np.where(b > 0, mu_excess / b, 0.0)
        f_opt = np.maximum(f_star_excess, 0.0)
    else:
        # Quadratic programme: maximise f·μ_excess − ½ f·Σ·fᵀ
        # subject to f ≥ 0 and sum(f) ≤ max_total_exposure
        Sigma = np.asarray(cov_matrix, dtype=np.float64)
        if Sigma.shape != (n, n):
            raise ValueError(
                f"cov_matrix shape {Sigma.shape} does not match n_bets={n}."
            )

        def neg_obj(f: np.ndarray) -> float:
            return float(-(f @ mu_excess) + 0.5 * (f @ Sigma @ f))

        def neg_grad(f: np.ndarray) -> np.ndarray:
            return -mu_excess + Sigma @ f

        bounds     = [(0.0, max_total_exposure)] * n
        constraint = {"type": "ineq", "fun": lambda f: max_total_exposure - f.sum()}
        x0 = np.full(n, max_total_exposure / n)

        res = minimize(
            neg_obj,
            x0,
            jac=neg_grad,
            method="SLSQP",
            bounds=bounds,
            constraints=[constraint],
            options={"ftol": 1e-10, "maxiter": 500},
        )
        f_opt = np.maximum(res.x, 0.0)

    # Apply fractional divisor
    divisor = max(fractional_divisor, 1e-6)
    f_kelly = f_opt / divisor

    # Enforce per-bet positivity and total exposure cap
    f_kelly = np.maximum(f_kelly, 0.0)
    total   = f_kelly.sum()
    if total > max_total_exposure:
        f_kelly = f_kelly * (max_total_exposure / total)

    result = bets_df.copy()
    result["ev"]          = mu.round(6)
    result["ev_excess"]   = mu_excess.round(6)
    result["f_star"]      = f_star_single.round(6)
    result["f_kelly"]     = f_kelly.round(6)
    result["f_final"]     = f_kelly.round(6)
    result["stake_units"] = (f_kelly * bankroll).round(4)

    return result


# ---------------------------------------------------------------------------
# Empirical covariance from game simulator
# ---------------------------------------------------------------------------

@dataclass
class SimCorrResult:
    """
    Empirical covariance and correlation matrices extracted from the game
    simulator across a joint slate of games.

    Attributes
    ----------
    cov_matrix   : (n_bets, n_bets) empirical covariance matrix of win indicators.
                   Use this as the Sigma argument to portfolio_kelly().
    corr_matrix  : (n_bets, n_bets) Pearson correlation matrix.  Diagonal = 1.0.
    win_probs    : (n_bets,) empirical win probability per bet from simulations.
    outcome_matrix : (n_bets, n_sims) raw win-indicator matrix used to compute Σ.
                     Preserved for downstream analysis.
    bet_labels   : list of str labels for each row/column of the matrix.
    n_sims       : number of simulations used.
    """
    cov_matrix:     np.ndarray
    corr_matrix:    np.ndarray
    win_probs:      np.ndarray
    outcome_matrix: np.ndarray
    bet_labels:     list[str]
    n_sims:         int

    def to_dict(self) -> dict:
        return {
            "bet_labels":  self.bet_labels,
            "n_sims":      self.n_sims,
            "win_probs":   self.win_probs.tolist(),
            "cov_matrix":  self.cov_matrix.tolist(),
            "corr_matrix": self.corr_matrix.tolist(),
        }


def simulate_corr_matrix(
    games: list[dict],
    n_sims: int = 20_000,
    rng: "np.random.Generator | None" = None,
) -> SimCorrResult:
    """
    Run the game simulator jointly across all games in a slate and extract
    the empirical covariance / correlation matrix from win-indicator outcomes.

    Why this matters for Kelly sizing
    ----------------------------------
    Two bets on different games in the same slate are NOT truly independent if:
    - They share correlated environmental factors (dome games both affected by
      weather forecasts; double-headers where lineup fatigue is shared)
    - They are same-game legs (totals + moneyline) for the same contest

    The portfolio Kelly formula f* = Σ⁻¹(μ − r·1) requires Σ to be the
    covariance matrix of bet outcomes, NOT assumed-diagonal.  Using a diagonal
    Σ over-bets correlated legs.  This function derives Σ empirically from
    the simulator's joint outcome distributions, capturing all structural
    correlations without model assumptions.

    For independent games the off-diagonal elements will be near zero;
    for same-game legs (moneyline + total) they will be meaningfully non-zero.

    Input format
    ------------
    Each game in *games* is a dict with keys:
        pa_probs_home : np.ndarray (9, 7) — PA outcome probs for home lineup
        pa_probs_away : np.ndarray (9, 7) — PA outcome probs for away lineup
        side          : "home" | "away"   — which side this bet is on
        label         : str               — bet identifier (game_pk + side)

    For same-game parlay legs from the *same* game, include multiple entries
    with the same pa_probs but different 'side' or 'outcome' keys (e.g. home
    moneyline AND over/under) — the simulator runs that game once per bundle
    and correlates outcomes across legs automatically.

    Parameters
    ----------
    games    : list of bet-leg dicts (see format above)
    n_sims   : simulations per game bundle (default 20,000)
    rng      : numpy Generator; seeded internally if None

    Returns
    -------
    SimCorrResult with cov_matrix, corr_matrix, win_probs, outcome_matrix.

    Notes
    -----
    - The simulator is imported lazily so this function can be called without
      a full model pipeline (only pa_probs arrays are needed).
    - Each unique (pa_probs_home, pa_probs_away) pair is simulated once; bets
      on the same game share the same simulation draws.
    - This makes the covariance computation O(n_unique_games * n_sims), not
      O(n_legs * n_sims).
    """
    # Import the half-inning simulator directly so we can pass raw pa_probs
    # arrays without needing full lineup/model objects.
    from src.models.game_simulator import (  # noqa: PLC0415
        _simulate_half_inning_variable, GameSimResult,
    )

    if rng is None:
        rng = np.random.default_rng()

    if not games:
        empty = np.empty((0, 0))
        return SimCorrResult(
            cov_matrix=empty, corr_matrix=empty,
            win_probs=np.empty(0), outcome_matrix=np.empty((0, 0)),
            bet_labels=[], n_sims=n_sims,
        )

    n_legs = len(games)

    # ── Step 1: simulate each unique game once ──────────────────────────────
    # Key a game by the content of its pa_probs arrays (first-row sum as proxy).
    # Callers who want to correlate same-game legs must pass the SAME
    # numpy array objects; the content-based key ensures structural equality.

    def _game_key(g: dict) -> bytes:
        """Deterministic content-based key for a (home, away) PA-probs pair."""
        h = np.asarray(g["pa_probs_home"], dtype=np.float32).tobytes()
        a = np.asarray(g["pa_probs_away"], dtype=np.float32).tobytes()
        return h + b"|" + a

    def _run_game(pa_h: np.ndarray, pa_a: np.ndarray) -> "GameSimResult":
        """Simulate one game using only the half-inning engine + raw PA probs."""
        # Normalise every row to sum exactly to 1.0 (float64 arithmetic can
        # produce rows that sum to 1 ± 1e-15; rng.choice is sensitive to this)
        pa_h = pa_h / pa_h.sum(axis=1, keepdims=True)
        pa_a = pa_a / pa_a.sum(axis=1, keepdims=True)

        n_innings = 9
        max_extra = 6
        home_runs = np.zeros(n_sims, dtype=np.int32)
        away_runs = np.zeros(n_sims, dtype=np.int32)

        for inn in range(n_innings):
            slot_probs_a = pa_a   # (9, 7) — all slots for this game
            slot_probs_h = pa_h

            inn_runs_a, _ = _simulate_half_inning_variable(
                slot_probs_a, n_sims, start_slot=0, rng=rng,
            )
            away_runs += inn_runs_a

            if inn < n_innings - 1:
                inn_runs_h, _ = _simulate_half_inning_variable(
                    slot_probs_h, n_sims, start_slot=0, rng=rng,
                )
                home_runs += inn_runs_h
            else:
                # Bottom of 9th: only teams that are tied or trailing bat
                needs = away_runs >= home_runs
                n_need = int(needs.sum())
                if n_need > 0:
                    inn_runs_h_part, _ = _simulate_half_inning_variable(
                        slot_probs_h, n_need, start_slot=0, rng=rng,
                    )
                    home_runs[needs] += inn_runs_h_part

        # Extra innings (automatic runner)
        tied = home_runs == away_runs
        for ex in range(max_extra):
            n_tied = int(tied.sum())
            if n_tied == 0:
                break
            slot_a = pa_a   # (9, 7)
            slot_h = pa_h
            ex_a, _ = _simulate_half_inning_variable(
                slot_a, n_tied, start_slot=0, rng=rng, start_base_config=2,
            )
            away_runs[tied] += ex_a
            ex_h, _ = _simulate_half_inning_variable(
                slot_h, n_tied, start_slot=0, rng=rng, start_base_config=2,
            )
            home_runs[tied] += ex_h
            tied_sub = home_runs[tied] == away_runs[tied]
            new_tied = np.zeros(n_sims, dtype=bool)
            new_tied[np.where(tied)[0][tied_sub]] = True
            tied = new_tied

        return GameSimResult(home_runs_dist=home_runs, away_runs_dist=away_runs)

    # Cache results: simulate each unique game exactly once
    game_results: dict[bytes, "GameSimResult"] = {}
    for g in games:
        key = _game_key(g)
        if key not in game_results:
            game_results[key] = _run_game(
                np.asarray(g["pa_probs_home"], dtype=np.float64),
                np.asarray(g["pa_probs_away"], dtype=np.float64),
            )

    # ── Step 2: build outcome matrix (n_legs × n_sims) ─────────────────────
    # outcome[i, sim] = 1 if bet i wins in simulation sim, 0 otherwise
    outcome_matrix = np.zeros((n_legs, n_sims), dtype=np.float64)

    for i, g in enumerate(games):
        key    = _game_key(g)
        result = game_results[key]
        side   = str(g.get("side", "home")).lower()
        outcome_type = str(g.get("outcome", "moneyline")).lower()

        home_runs = result.home_runs_dist
        away_runs = result.away_runs_dist

        if outcome_type == "moneyline":
            if side == "home":
                outcome_matrix[i] = (home_runs > away_runs).astype(float)
            else:
                outcome_matrix[i] = (away_runs > home_runs).astype(float)
        elif outcome_type == "over":
            line = float(g.get("total_line", 8.5))
            outcome_matrix[i] = ((home_runs + away_runs) > line).astype(float)
        elif outcome_type == "under":
            line = float(g.get("total_line", 8.5))
            outcome_matrix[i] = ((home_runs + away_runs) < line).astype(float)
        elif outcome_type == "spread":
            run_line = float(g.get("run_line", -1.5))
            if side == "home":
                outcome_matrix[i] = ((home_runs - away_runs) > run_line).astype(float)
            else:
                outcome_matrix[i] = ((away_runs - home_runs) > run_line).astype(float)
        else:
            # Default: treat as moneyline home
            outcome_matrix[i] = (home_runs > away_runs).astype(float)

    # ── Step 3: empirical covariance and correlation ────────────────────────
    # cov[i,j] = E[X_i * X_j] - E[X_i] * E[X_j]  (sample covariance)
    win_probs = outcome_matrix.mean(axis=1)            # (n_legs,)

    # np.cov returns a scalar for a single-row input; force (n_legs, n_legs)
    raw_cov    = np.cov(outcome_matrix, ddof=1)
    cov_matrix = np.atleast_2d(raw_cov).reshape(n_legs, n_legs)

    # Correlation matrix from covariance
    std_diag = np.sqrt(np.maximum(np.diag(cov_matrix), 0.0))
    # Guard against zero-variance legs (degenerate)
    outer_std = np.outer(std_diag, std_diag)
    outer_std = np.where(outer_std > 1e-12, outer_std, 1.0)
    corr_matrix = cov_matrix / outer_std
    np.fill_diagonal(corr_matrix, 1.0)

    labels = [str(g.get("label", f"bet_{i}")) for i, g in enumerate(games)]

    return SimCorrResult(
        cov_matrix=cov_matrix,
        corr_matrix=corr_matrix,
        win_probs=win_probs,
        outcome_matrix=outcome_matrix,
        bet_labels=labels,
        n_sims=n_sims,
    )
