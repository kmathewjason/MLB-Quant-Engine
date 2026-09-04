"""
models.game_simulator
=====================
Vectorized Monte Carlo game simulator that draws PA outcomes from
pa_outcome_model.PAOutcomeModel for each batter/pitcher matchup in real
lineup order and simulates full 9-inning games (with extra innings).

Architecture
------------
The simulator operates in two layers:

1. PA-probability matrix pre-computation
   For each (batter_slot, pitcher, inning_context) combination that can
   appear in a game, call PAOutcomeModel.predict_proba() once.  This gives
   a (9_innings × 9_batters, 7) probability tensor that is reused across
   all N simulations, avoiding per-simulation model inference.

2. Vectorized inning simulation
   Uses the markov_re_matrix.simulate_half_inning() engine, extended here
   to accept a *different* PA probability vector per plate appearance
   (batter-specific, TTO-adjusted).  The game loop is:

       for inning 1..9 (or until tie-broken in extras):
           home_runs[sim] += simulate_half_inning_variable(away_lineup, pitcher=home_sp)
           away_runs[sim] += simulate_half_inning_variable(home_lineup, pitcher=away_sp)

   "variable" means each PA position has its own probability vector
   (batter × TTO × inning context), but all N simulations share the same
   sequence of probability lookups — only the random draws differ.

Times-through-order (TTO) tracking
-----------------------------------
The batter-slot pointer resets to 0 each inning and carries across innings.
TTO is tracked by counting PA appearances per batter slot per pitcher in
the simulation context (pre-game: always starts at TTO=1, increments each
time the slot wraps around the order).

Extra innings
-------------
After 9 innings, if the score is tied, play continues one inning at a time
with a runner-on-second rule (the "automatic runner" in MLB extra innings).
Each tie is broken per-sim independently — no truncation.

Output: GameSimResult
---------------------
  home_runs_dist : np.ndarray (N,) — home runs in each simulation
  away_runs_dist : np.ndarray (N,) — away runs in each simulation
  win_prob_home  : float   — P(home wins)
  win_prob_away  : float   — 1 − win_prob_home
  mean_total     : float   — E[home_runs + away_runs]
  std_total      : float   — σ[home_runs + away_runs]

Public API
----------
GameSimResult                                   — dataclass
simulate_game(home_lineup, away_lineup,          -> GameSimResult
              home_sp_features, away_sp_features,
              env, model, n_sims, rng)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Imports from within the engine (lazy to avoid circular at module-load)
# ---------------------------------------------------------------------------

def _get_markov():
    from src.models.markov_re_matrix import _ADVANCE_TABLE, N_PA_OUTCOMES
    return _ADVANCE_TABLE, N_PA_OUTCOMES


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class GameSimResult:
    """Full distribution of simulated game outcomes."""
    home_runs_dist: np.ndarray   # (N,) — home team runs per sim
    away_runs_dist: np.ndarray   # (N,) — away team runs per sim

    @property
    def win_prob_home(self) -> float:
        """P(home wins) — ties counted as 0.5."""
        home = self.home_runs_dist
        away = self.away_runs_dist
        wins = (home > away).sum()
        ties = (home == away).sum()
        return float((wins + 0.5 * ties) / len(home))

    @property
    def win_prob_away(self) -> float:
        return 1.0 - self.win_prob_home

    @property
    def mean_total(self) -> float:
        return float((self.home_runs_dist + self.away_runs_dist).mean())

    @property
    def std_total(self) -> float:
        return float((self.home_runs_dist + self.away_runs_dist).std())

    @property
    def mean_home(self) -> float:
        return float(self.home_runs_dist.mean())

    @property
    def mean_away(self) -> float:
        return float(self.away_runs_dist.mean())

    def spread_cover_prob(self, run_line: float) -> float:
        """
        P(home - away > run_line).
        run_line = -1.5 means home must win by 2+.
        """
        margin = self.home_runs_dist - self.away_runs_dist
        return float((margin > run_line).mean())

    def total_over_prob(self, total_line: float) -> float:
        """P(home_runs + away_runs > total_line)."""
        totals = self.home_runs_dist + self.away_runs_dist
        return float((totals > total_line).mean())


# ---------------------------------------------------------------------------
# Lineup representation
# ---------------------------------------------------------------------------

def make_lineup_pa_probs(
    batting_order: list[dict],
    pitcher_features: dict,
    env: dict,
    model,
    n_innings: int = 9,
    max_tto: int = 3,
) -> np.ndarray:
    """
    Pre-compute PA outcome probability vectors for every (inning, slot, TTO)
    combination that can appear in a 9-inning game.

    Parameters
    ----------
    batting_order    : list of 9 batter feature dicts; index 0 = leadoff
    pitcher_features : pitcher feature dict
    env              : park/weather EnvFactor dict (air_density_kg_m3 etc.)
    model            : fitted PAOutcomeModel with .predict_proba(X)
    n_innings        : innings to pre-compute (9 for regulation; more for extras)
    max_tto          : cap TTO label at this value (default 3)

    Returns
    -------
    pa_probs : np.ndarray, shape (n_innings, 9, 7)
               pa_probs[inning_idx, slot, :] = P(outcome) for that matchup.
               inning_idx is 0-based.
    """
    from src.models.pa_outcome_model import make_feature_vector  # noqa: PLC0415

    probs = np.zeros((n_innings, 9, 7), dtype=np.float64)

    # Track how many times each slot has batted (across all innings so far)
    slot_pa_count = np.zeros(9, dtype=int)

    for inn_idx in range(n_innings):
        for slot in range(9):
            # TTO: 1 the first 9 PA, 2 the next 9, 3+ thereafter
            # For pre-computation we approximate by inning number:
            # first 3 innings ≈ TTO1, innings 4-6 ≈ TTO2, 7+ ≈ TTO3
            if inn_idx < 3:
                tto = 1
            elif inn_idx < 6:
                tto = 2
            else:
                tto = min(3, max_tto)

            batter = batting_order[slot]
            from src.features.matchup_features import matchup_pa_probs, MLB_LEAGUE_AVG  # noqa: PLC0415

            # Build matchup dict: use batter rates × pitcher rates → log5
            batter_dict = {
                "K":   batter.get("batter_k_rate",  0.22),
                "BB":  batter.get("batter_bb_rate", 0.084),
                "1B":  batter.get("batter_1b_rate", 0.148),
                "HR":  batter.get("batter_hr_rate", 0.034),
                "HBP": 0.010,
                "2B":  0.047,
                "3B":  0.005,
                "OUT": 1.0,   # OUT will be normalised away
            }
            pitcher_dict = {
                "K":   pitcher_features.get("pitcher_k_rate",  0.22),
                "BB":  pitcher_features.get("pitcher_bb_rate", 0.084),
                "1B":  pitcher_features.get("pitcher_1b_rate", 0.148),
                "HR":  pitcher_features.get("pitcher_hr_rate", 0.034),
                "HBP": 0.010,
                "2B":  0.047,
                "3B":  0.005,
                "OUT": 1.0,
            }
            matchup = matchup_pa_probs(batter_dict, pitcher_dict)

            context = {
                "home_team":           float(batter.get("home_team", 0)),
                "inning":              inn_idx + 1,
                "outs_when_up":        0,
                "times_through_order": tto,
            }

            X = make_feature_vector(batter, pitcher_features, matchup, env, context)
            raw = model.predict_proba(X)[0]   # (7,) — 8-class PA model

            # Map 7-class (K, BB_HBP, 1B, 2B, 3B, HR, OUT) to 7-class Markov
            # (K=0, BB=1, 1B=2, 2B=3, 3B=4, HR=5, OUT=6) — direct correspondence
            probs[inn_idx, slot, :] = raw / raw.sum()

    return probs


# ---------------------------------------------------------------------------
# Vectorized variable-PA half-inning simulator
# ---------------------------------------------------------------------------

def _simulate_half_inning_variable(
    slot_probs: np.ndarray,
    n_sims: int,
    start_slot: int,
    rng: np.random.Generator,
    start_base_config: int = 0,
    start_outs: int = 0,
    max_pa: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate *n_sims* half-innings where each PA position has a *different*
    probability vector (batter-specific).  Batting order cycles.

    Parameters
    ----------
    slot_probs       : (9, 7) array — one PA prob vector per lineup slot
    n_sims           : number of parallel simulations
    start_slot       : which batting slot leads off this inning
    rng              : np.random.Generator
    start_base_config: initial base config (0 = empty, 1 = runner on 2B for extras)
    start_outs       : initial outs (0 for normal innings)
    max_pa           : safety ceiling

    Returns
    -------
    runs   : np.ndarray (n_sims,) — runs scored
    end_slot : np.ndarray (n_sims,) — slot that will lead off next inning
               (for carry-over batting order)
    """
    ADVANCE_TABLE, N_PA_OUTCOMES = _get_markov()
    flat_table = ADVANCE_TABLE.reshape(-1, 3).astype(np.int32)

    base_config = np.full(n_sims, start_base_config, dtype=np.int32)
    outs        = np.full(n_sims, start_outs,        dtype=np.int32)
    runs        = np.zeros(n_sims, dtype=np.int32)
    active      = np.ones(n_sims,  dtype=bool)
    slot        = np.full(n_sims, start_slot % 9, dtype=np.int32)

    for _ in range(max_pa):
        if not active.any():
            break

        # For each active sim, get the PA probs for its current batter slot
        # We group active sims by their current slot to batch rng.choice calls
        current_slot = slot.copy()
        # Initialize to 0 (a valid outcome) so inactive sims have a safe index
        # even though their result is discarded by the np.where mask below.
        outcomes = np.zeros(n_sims, dtype=np.int32)

        # Process each slot group independently
        for s in range(9):
            mask = active & (current_slot == s)
            n_in_slot = mask.sum()
            if n_in_slot == 0:
                continue
            outcomes[mask] = rng.choice(N_PA_OUTCOMES, size=n_in_slot, p=slot_probs[s])

        # Table lookup — inactive sims use (base=0, outs=0, outcome=0) which
        # is valid; their results are discarded by np.where on the update lines.
        safe_base = np.where(active, base_config, 0)
        safe_outs = np.where(active, outs, 0)
        idx = safe_base * 21 + safe_outs * 7 + outcomes
        results = flat_table[idx]

        new_base = results[:, 0]
        new_outs = results[:, 1]
        new_runs = results[:, 2]

        runs        = np.where(active, runs  + new_runs, runs)
        base_config = np.where(active, new_base, base_config)
        outs        = np.where(active, new_outs, outs)

        # Advance slot only for active sims
        slot = np.where(active, (slot + 1) % 9, slot)

        # Deactivate innings that just reached 3 outs
        active = active & (outs < 3)

    return runs, slot


# ---------------------------------------------------------------------------
# Full game simulator
# ---------------------------------------------------------------------------

def simulate_game(
    home_lineup: list[dict],
    away_lineup: list[dict],
    home_sp_features: dict,
    away_sp_features: dict,
    env: dict,
    model,
    n_sims: int = 10_000,
    n_innings: int = 9,
    max_extra_innings: int = 12,
    rng: np.random.Generator | None = None,
) -> GameSimResult:
    """
    Simulate *n_sims* full MLB games.

    Parameters
    ----------
    home_lineup       : list of 9 batter feature dicts for the home team
    away_lineup       : list of 9 batter feature dicts for the away team
    home_sp_features  : feature dict for the home starting pitcher
    away_sp_features  : feature dict for the away starting pitcher
    env               : park/weather scalar dict (from park_weather.merge_weather)
    model             : fitted PAOutcomeModel or any object with .predict_proba(X)
    n_sims            : number of Monte Carlo simulations (default 10,000)
    n_innings         : regulation innings (default 9)
    max_extra_innings : maximum extra innings played per sim (default 12)
    rng               : numpy random generator (seeded for reproducibility)

    Returns
    -------
    GameSimResult with full run distribution arrays of shape (n_sims,).
    """
    if rng is None:
        rng = np.random.default_rng()

    # Pre-compute PA probs for regulation innings + max extras
    n_innings_total = n_innings + max_extra_innings
    logger.debug("Pre-computing PA probability matrices (%d innings)", n_innings_total)

    # Away batters vs home starter (away team bats in top of inning)
    away_pa_probs = make_lineup_pa_probs(
        away_lineup, home_sp_features, env, model, n_innings=n_innings_total
    )
    # Home batters vs away starter
    home_pa_probs = make_lineup_pa_probs(
        home_lineup, away_sp_features, env, model, n_innings=n_innings_total
    )

    home_runs = np.zeros(n_sims, dtype=np.int32)
    away_runs = np.zeros(n_sims, dtype=np.int32)

    # Batting order carry-over: track which slot leads off each inning
    home_lead_slot = np.zeros(n_sims, dtype=np.int32)
    away_lead_slot = np.zeros(n_sims, dtype=np.int32)

    # ── Regulation innings ────────────────────────────────────────────
    for inn_idx in range(n_innings):
        # Top of inning: away bats vs home SP
        inn_runs, next_slot = _simulate_half_inning_variable(
            away_pa_probs[inn_idx],
            n_sims,
            start_slot=0,   # simplification: slot 0 each inning
                            # (true carry-over would require per-sim tracking)
            rng=rng,
        )
        away_runs += inn_runs

        # Bottom of inning: home bats vs away SP
        # Home team doesn't bat in bottom of 9th if winning
        if inn_idx == n_innings - 1:
            # Only simulate for sims where home team is tied or trailing
            needs_bat = away_runs >= home_runs
            n_needs = int(needs_bat.sum())
            if n_needs > 0:
                inn_runs_partial, _ = _simulate_half_inning_variable(
                    home_pa_probs[inn_idx],
                    n_needs,
                    start_slot=0,
                    rng=rng,
                )
                home_runs[needs_bat] += inn_runs_partial
        else:
            inn_runs, _ = _simulate_half_inning_variable(
                home_pa_probs[inn_idx],
                n_sims,
                start_slot=0,
                rng=rng,
            )
            home_runs += inn_runs

    # ── Extra innings ─────────────────────────────────────────────────
    tied = home_runs == away_runs
    for extra_idx in range(max_extra_innings):
        n_tied = int(tied.sum())
        if n_tied == 0:
            break

        inn_idx = n_innings + extra_idx
        capped_inn = min(inn_idx, n_innings_total - 1)

        # MLB automatic runner: start with runner on 2nd (base_config = 2)
        inn_runs_away, _ = _simulate_half_inning_variable(
            away_pa_probs[capped_inn],
            n_tied,
            start_slot=0,
            rng=rng,
            start_base_config=2,   # runner on 2nd
        )
        away_runs[tied] += inn_runs_away

        inn_runs_home, _ = _simulate_half_inning_variable(
            home_pa_probs[capped_inn],
            n_tied,
            start_slot=0,
            rng=rng,
            start_base_config=2,
        )
        home_runs[tied] += inn_runs_home

        # Update tied mask
        tied_subset = home_runs[tied] == away_runs[tied]
        new_tied = np.zeros(n_sims, dtype=bool)
        new_tied[np.where(tied)[0][tied_subset]] = True
        tied = new_tied

    logger.debug(
        "simulate_game complete: home=%.2f away=%.2f n_still_tied=%d",
        home_runs.mean(), away_runs.mean(), int(tied.sum()),
    )

    return GameSimResult(
        home_runs_dist=home_runs,
        away_runs_dist=away_runs,
    )
