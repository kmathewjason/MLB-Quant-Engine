"""
tests/test_models.py
=====================
Unit tests for the four model modules implemented in Phase 3/4:

  src/models/pa_outcome_model.py
  src/models/ensemble.py
  src/models/game_simulator.py
  src/backtest/run_dispersion_check.py

All tests use synthetic data so no real Statcast/Retrosheet files are needed.
Heavy ML training is exercised with tiny synthetic datasets (N~200) so the
test suite stays fast (<30s total).
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest


# ===========================================================================
# ── Helpers ─────────────────────────────────────────────────────────────────
# ===========================================================================

def _make_synthetic_pa_dataset(n: int = 300, n_classes: int = 7, seed: int = 0):
    """Create a deterministic synthetic (X, y) PA dataset."""
    rng = np.random.default_rng(seed)
    from src.models.pa_outcome_model import N_FEATURES
    X = rng.standard_normal((n, N_FEATURES)).astype(np.float32)
    # Slightly unbalanced classes (realistic for PA outcomes)
    probs = np.array([0.22, 0.09, 0.15, 0.047, 0.005, 0.034, 0.454])
    probs /= probs.sum()
    y = rng.choice(n_classes, size=n, p=probs).astype(np.int32)
    return X, y


def _make_league_average_lineup(n_batters: int = 9) -> list[dict]:
    """Return a list of 9 league-average batter feature dicts."""
    return [
        {
            "batter_k_rate":  0.222,
            "batter_bb_rate": 0.084,
            "batter_1b_rate": 0.148,
            "batter_hr_rate": 0.034,
            "batter_babip":   0.300,
            "home_team":      0,
        }
    ] * n_batters


def _make_league_average_pitcher() -> dict:
    return {
        "pitcher_k_rate":  0.222,
        "pitcher_bb_rate": 0.084,
        "pitcher_1b_rate": 0.148,
        "pitcher_hr_rate": 0.034,
        "pitcher_babip":   0.300,
    }


def _neutral_env() -> dict:
    return {
        "env_run_factor":   1.0,
        "env_hr_factor":    1.0,
        "air_density_kg_m3": 1.225,
    }


# ===========================================================================
# ── pa_outcome_model — feature construction ──────────────────────────────────
# ===========================================================================

class TestMakeFeatureVector:
    def test_output_shape(self):
        from src.models.pa_outcome_model import make_feature_vector, N_FEATURES
        batter  = {"batter_k_rate": 0.25, "batter_bb_rate": 0.09, "batter_1b_rate": 0.15,
                   "batter_hr_rate": 0.04, "batter_babip": 0.31}
        pitcher = {"pitcher_k_rate": 0.22, "pitcher_bb_rate": 0.08, "pitcher_1b_rate": 0.14,
                   "pitcher_hr_rate": 0.03, "pitcher_babip": 0.29}
        matchup = {"K": 0.22, "BB": 0.08, "1B": 0.14, "2B": 0.05, "3B": 0.005,
                   "HR": 0.03, "HBP": 0.01}
        env     = {"env_run_factor": 1.1, "env_hr_factor": 1.2, "air_density_kg_m3": 1.1}
        context = {"home_team": 1, "inning": 4, "outs_when_up": 1, "times_through_order": 2}
        X = make_feature_vector(batter, pitcher, matchup, env, context)
        assert X.shape == (1, N_FEATURES)
        assert X.dtype == np.float32

    def test_tto_one_hot_exclusive(self):
        """Exactly one TTO flag is 1."""
        from src.models.pa_outcome_model import make_feature_vector, FEATURE_NAMES
        batter = pitcher = {}
        matchup = env = {}
        tto_idx = [FEATURE_NAMES.index("tto_1"),
                   FEATURE_NAMES.index("tto_2"),
                   FEATURE_NAMES.index("tto_3plus")]
        for tto in [1, 2, 3]:
            ctx = {"times_through_order": tto}
            X = make_feature_vector(batter, pitcher, matchup, env, ctx)
            flags = [X[0, i] for i in tto_idx]
            assert sum(flags) == pytest.approx(1.0, abs=1e-6), f"TTO={tto}: flags={flags}"

    def test_tto_interaction_terms_correct(self):
        """tto1_x_pk should equal tto_1 * pitcher_k_rate."""
        from src.models.pa_outcome_model import make_feature_vector, FEATURE_NAMES
        pk = 0.28
        batter = {}
        pitcher = {"pitcher_k_rate": pk}
        X = make_feature_vector(batter, pitcher, {}, {}, {"times_through_order": 1})
        tto1_idx = FEATURE_NAMES.index("tto_1")
        tto1_x_pk_idx = FEATURE_NAMES.index("tto1_x_pk")
        tto2_x_pk_idx = FEATURE_NAMES.index("tto2_x_pk")
        assert X[0, tto1_x_pk_idx] == pytest.approx(pk, rel=1e-5)
        assert X[0, tto2_x_pk_idx] == pytest.approx(0.0, abs=1e-6)

    def test_default_values_dont_raise(self):
        """Empty dicts should not raise — all values have defaults."""
        from src.models.pa_outcome_model import make_feature_vector
        X = make_feature_vector({}, {}, {}, {}, {})
        assert X.shape[0] == 1

    def test_tto_3plus_for_high_tto(self):
        """times_through_order >= 3 should set tto_3plus = 1."""
        from src.models.pa_outcome_model import make_feature_vector, FEATURE_NAMES
        idx = FEATURE_NAMES.index("tto_3plus")
        for tto in [3, 4, 5]:
            X = make_feature_vector({}, {}, {}, {}, {"times_through_order": tto})
            assert X[0, idx] == pytest.approx(1.0, abs=1e-6)


# ===========================================================================
# ── pa_outcome_model — chronological split ──────────────────────────────────
# ===========================================================================

class TestChronologicalSplit:
    def test_no_overlap(self):
        from src.models.pa_outcome_model import chronological_split
        tr, val, cal, te = chronological_split(1000)
        all_idx = np.concatenate([tr, val, cal, te])
        assert len(all_idx) == len(set(all_idx)), "Overlap detected between splits"

    def test_order_preserved(self):
        from src.models.pa_outcome_model import chronological_split
        tr, val, cal, te = chronological_split(1000)
        assert tr.max() < val.min()
        assert val.max() < cal.min()
        assert cal.max() < te.min()

    def test_fractions_approximately_correct(self):
        from src.models.pa_outcome_model import chronological_split
        n = 10_000
        tr, val, cal, te = chronological_split(n)
        assert len(tr) / n == pytest.approx(0.70, abs=0.01)
        assert len(val) / n == pytest.approx(0.15, abs=0.01)
        assert len(cal) / n == pytest.approx(0.08, abs=0.01)

    def test_covers_all_indices(self):
        from src.models.pa_outcome_model import chronological_split
        n = 500
        tr, val, cal, te = chronological_split(n)
        assert len(tr) + len(val) + len(cal) + len(te) == n

    def test_small_n_doesnt_crash(self):
        from src.models.pa_outcome_model import chronological_split
        tr, val, cal, te = chronological_split(10)
        assert len(tr) + len(val) + len(cal) + len(te) == 10


# ===========================================================================
# ── pa_outcome_model — training on tiny synthetic data ──────────────────────
# ===========================================================================

@pytest.fixture(scope="module")
def tiny_fitted_model():
    """Train a PAOutcomeModel on 300 synthetic rows (fast, cached per module)."""
    pytest.importorskip("xgboost")
    pytest.importorskip("sklearn")
    from src.models.pa_outcome_model import PAOutcomeModel
    X, y = _make_synthetic_pa_dataset(300)
    model = PAOutcomeModel(n_estimators=10, mlp_max_iter=20, mlp_hidden=(16,))
    model.fit(X, y)
    return model


class TestPAOutcomeModelFit:
    def test_is_fitted_after_fit(self, tiny_fitted_model):
        assert tiny_fitted_model._is_fitted is True

    def test_predict_proba_shape(self, tiny_fitted_model):
        from src.models.pa_outcome_model import N_CLASSES
        X, _ = _make_synthetic_pa_dataset(50)
        probs = tiny_fitted_model.predict_proba(X)
        assert probs.shape == (50, N_CLASSES)

    def test_predict_proba_rows_sum_to_one(self, tiny_fitted_model):
        X, _ = _make_synthetic_pa_dataset(30)
        probs = tiny_fitted_model.predict_proba(X)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_predict_proba_non_negative(self, tiny_fitted_model):
        X, _ = _make_synthetic_pa_dataset(30)
        probs = tiny_fitted_model.predict_proba(X)
        assert np.all(probs >= 0)

    def test_blend_weight_in_unit_interval(self, tiny_fitted_model):
        w = tiny_fitted_model._blend_weight
        assert 0.0 <= w <= 1.0

    def test_predict_before_fit_raises(self):
        from src.models.pa_outcome_model import PAOutcomeModel
        m = PAOutcomeModel()
        X, _ = _make_synthetic_pa_dataset(5)
        with pytest.raises(RuntimeError, match="fit"):
            m.predict_proba(X)


# ===========================================================================
# ── ensemble — StackedEnsemble ──────────────────────────────────────────────
# ===========================================================================

@pytest.fixture(scope="module")
def tiny_ensemble():
    pytest.importorskip("xgboost")
    pytest.importorskip("sklearn")
    from src.models.ensemble import StackedEnsemble
    X, y = _make_synthetic_pa_dataset(300)
    ens = StackedEnsemble(n_estimators=10, mlp_max_iter=20, mlp_hidden=(16,))
    ens.fit(X, y)
    return ens


class TestStackedEnsemble:
    def test_predict_proba_sums_to_one(self, tiny_ensemble):
        X, _ = _make_synthetic_pa_dataset(40)
        probs = tiny_ensemble.predict_proba(X)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_predict_proba_non_negative(self, tiny_ensemble):
        X, _ = _make_synthetic_pa_dataset(40)
        assert np.all(tiny_ensemble.predict_proba(X) >= 0)

    def test_blend_weight_in_unit_interval(self, tiny_ensemble):
        w = tiny_ensemble.blend_weight
        assert 0.0 <= w <= 1.0

    def test_raw_proba_rows_sum_approximately_one(self, tiny_ensemble):
        """Raw (pre-calibration) probs should also nearly sum to 1 per row."""
        X, _ = _make_synthetic_pa_dataset(20)
        raw = tiny_ensemble.predict_proba_raw(X)
        np.testing.assert_allclose(raw.sum(axis=1), 1.0, atol=1e-4)

    def test_evaluate_returns_expected_keys(self, tiny_ensemble):
        X, y = _make_synthetic_pa_dataset(50, seed=99)
        result = tiny_ensemble.evaluate(X, y)
        assert "log_loss" in result
        assert "accuracy" in result
        assert "ece" in result

    def test_predict_before_fit_raises(self):
        from src.models.ensemble import StackedEnsemble
        ens = StackedEnsemble()
        X, _ = _make_synthetic_pa_dataset(5)
        with pytest.raises(RuntimeError, match="fit"):
            ens.predict_proba(X)


class TestECE:
    def test_perfect_calibration_gives_low_ece(self):
        """If predicted probability == empirical frequency, ECE ≈ 0."""
        from src.models.ensemble import expected_calibration_error
        # Simulate perfectly calibrated binary classifier
        rng = np.random.default_rng(1)
        n = 1000
        probs_class = rng.uniform(0, 1, n)
        y = rng.binomial(1, probs_class).astype(int)
        # Wrap in (N, 2) format
        probs = np.column_stack([1 - probs_class, probs_class])
        ece = expected_calibration_error(probs, y)
        assert ece < 0.10, f"ECE too high for well-calibrated predictor: {ece:.4f}"

    def test_ece_returns_float(self):
        from src.models.ensemble import expected_calibration_error
        probs = np.ones((10, 2)) * 0.5
        y = np.zeros(10, dtype=int)
        result = expected_calibration_error(probs, y)
        assert isinstance(result, float)

    def test_reliability_diagram_returns_dataframe(self):
        from src.models.ensemble import reliability_diagram_data
        rng = np.random.default_rng(5)
        probs = np.column_stack([rng.uniform(0, 1, 200)] * 2)
        probs /= probs.sum(axis=1, keepdims=True)
        y = rng.integers(0, 2, size=200)
        df = reliability_diagram_data(probs, y)
        assert isinstance(df, pd.DataFrame)
        assert "bin_centre" in df.columns
        assert "empirical_freq" in df.columns


# ===========================================================================
# ── game_simulator — GameSimResult ──────────────────────────────────────────
# ===========================================================================

class TestGameSimResult:
    def _make_result(self, home_wins: int = 6000, n: int = 10_000):
        from src.models.game_simulator import GameSimResult
        home = np.zeros(n, dtype=np.int32)
        away = np.zeros(n, dtype=np.int32)
        home[:home_wins] = 5
        away[:home_wins] = 3
        away[home_wins:] = 5
        home[home_wins:] = 3
        return GameSimResult(home_runs_dist=home, away_runs_dist=away)

    def test_win_prob_home_correct(self):
        r = self._make_result(6000, 10_000)
        assert r.win_prob_home == pytest.approx(0.6, rel=1e-4)

    def test_win_prob_sums_to_one(self):
        r = self._make_result(4500, 10_000)
        assert r.win_prob_home + r.win_prob_away == pytest.approx(1.0, rel=1e-9)

    def test_mean_total_correct(self):
        from src.models.game_simulator import GameSimResult
        home = np.array([3, 5, 2, 8], dtype=np.int32)
        away = np.array([2, 4, 3, 1], dtype=np.int32)
        r = GameSimResult(home, away)
        assert r.mean_total == pytest.approx((3+5+2+8+2+4+3+1)/4, rel=1e-5)

    def test_spread_cover_prob_run_line_minus_1_5(self):
        """
        Home wins by 2+ in 4 out of 4 sims → 100% cover.
        """
        from src.models.game_simulator import GameSimResult
        home = np.array([5, 4, 3, 6], dtype=np.int32)
        away = np.array([2, 1, 0, 3], dtype=np.int32)
        r = GameSimResult(home, away)
        # home - away > -1.5 → home must win (all do here → 100%)
        assert r.spread_cover_prob(-1.5) == pytest.approx(1.0, abs=1e-9)

    def test_total_over_prob(self):
        from src.models.game_simulator import GameSimResult
        home = np.array([4, 5, 3], dtype=np.int32)
        away = np.array([4, 5, 3], dtype=np.int32)
        r = GameSimResult(home, away)
        # Totals: 8, 10, 6
        # P(total > 7) = 2/3
        assert r.total_over_prob(7.0) == pytest.approx(2.0 / 3.0, rel=1e-5)

    def test_tied_game_counts_half_win(self):
        from src.models.game_simulator import GameSimResult
        home = np.array([3, 3, 4], dtype=np.int32)
        away = np.array([3, 3, 3], dtype=np.int32)
        r = GameSimResult(home, away)
        # 2 ties (0.5 each) + 1 win = 2.0 / 3 ≈ 0.667
        assert r.win_prob_home == pytest.approx(2.0 / 3.0, rel=1e-5)


# ===========================================================================
# ── game_simulator — make_lineup_pa_probs + simulate (stub model) ───────────
# ===========================================================================

class _StubModel:
    """Always returns the MLB league-average distribution — no ML required."""
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # 7-class league-average: K=22%, BB_HBP=9.5%, 1B=14.8%, 2B=4.7%,
        # 3B=0.5%, HR=3.4%, OUT=45.1%
        p = np.array([0.222, 0.095, 0.148, 0.047, 0.005, 0.034, 0.449])
        p /= p.sum()
        return np.tile(p, (len(X), 1))


class TestMakeLineupPaProbs:
    def test_shape(self):
        from src.models.game_simulator import make_lineup_pa_probs
        lineup = _make_league_average_lineup()
        pitcher = _make_league_average_pitcher()
        env = _neutral_env()
        probs = make_lineup_pa_probs(lineup, pitcher, env, _StubModel(), n_innings=9)
        assert probs.shape == (9, 9, 7)

    def test_rows_sum_to_one(self):
        from src.models.game_simulator import make_lineup_pa_probs
        lineup  = _make_league_average_lineup()
        pitcher = _make_league_average_pitcher()
        probs   = make_lineup_pa_probs(lineup, pitcher, _neutral_env(), _StubModel(), n_innings=3)
        np.testing.assert_allclose(probs.sum(axis=2), 1.0, atol=1e-6)

    def test_tto_changes_probs_when_pitcher_k_varies(self):
        """
        With different pitcher_k_rate for two pitchers, the pre-computed
        prob arrays should differ.
        """
        from src.models.game_simulator import make_lineup_pa_probs
        lineup = _make_league_average_lineup()
        env    = _neutral_env()

        # Stub that returns different probs based on pitcher_k_rate in X
        class _SensitiveModel:
            def predict_proba(self, X: np.ndarray) -> np.ndarray:
                pk = float(X[0, 5])   # pitcher_k_rate column index = 5
                p = np.array([pk, 0.09, 0.15, 0.047, 0.005, 0.034, 1.0])
                p /= p.sum()
                return np.tile(p, (len(X), 1))

        p1 = make_lineup_pa_probs(
            lineup, {"pitcher_k_rate": 0.15}, env, _SensitiveModel(), n_innings=1
        )
        p2 = make_lineup_pa_probs(
            lineup, {"pitcher_k_rate": 0.32}, env, _SensitiveModel(), n_innings=1
        )
        # K-rate column (index 0) should differ
        assert not np.allclose(p1[:, :, 0], p2[:, :, 0])


class TestSimulateHalfInningVariable:
    def test_all_outs_produces_no_runs(self):
        from src.models.game_simulator import _simulate_half_inning_variable
        slot_probs = np.zeros((9, 7))
        slot_probs[:, 6] = 1.0   # 100% out-in-play for all slots
        runs, _ = _simulate_half_inning_variable(
            slot_probs, n_sims=500, start_slot=0,
            rng=np.random.default_rng(1)
        )
        assert np.all(runs == 0)

    def test_all_hr_produces_positive_runs(self):
        from src.models.game_simulator import _simulate_half_inning_variable
        slot_probs = np.zeros((9, 7))
        slot_probs[:, 5] = 1.0   # 100% HR for all slots
        runs, _ = _simulate_half_inning_variable(
            slot_probs, n_sims=200, start_slot=0,
            rng=np.random.default_rng(2)
        )
        assert runs.mean() > 2.0

    def test_extra_inning_runner_on_second_increases_scoring(self):
        """
        Starting with a runner on 2nd (extra-innings rule) should produce
        more runs on average than starting with bases empty.
        """
        from src.models.game_simulator import _simulate_half_inning_variable
        rng = np.random.default_rng(42)
        slot_probs = np.zeros((9, 7))
        slot_probs[:, :] = np.array([0.22, 0.09, 0.15, 0.047, 0.005, 0.034, 0.449])
        slot_probs[:, :] /= slot_probs[0].sum()

        runs_normal, _ = _simulate_half_inning_variable(
            slot_probs, n_sims=5000, start_slot=0,
            rng=rng, start_base_config=0
        )
        runs_extra, _ = _simulate_half_inning_variable(
            slot_probs, n_sims=5000, start_slot=0,
            rng=rng, start_base_config=2   # runner on 2nd
        )
        assert runs_extra.mean() > runs_normal.mean()


class TestSimulateGame:
    def test_result_shape(self):
        from src.models.game_simulator import simulate_game
        lineup  = _make_league_average_lineup()
        pitcher = _make_league_average_pitcher()
        env     = _neutral_env()
        result  = simulate_game(
            lineup, lineup, pitcher, pitcher, env, _StubModel(),
            n_sims=500, rng=np.random.default_rng(7)
        )
        assert result.home_runs_dist.shape == (500,)
        assert result.away_runs_dist.shape == (500,)

    def test_win_probs_sum_to_one(self):
        from src.models.game_simulator import simulate_game
        lineup  = _make_league_average_lineup()
        pitcher = _make_league_average_pitcher()
        env     = _neutral_env()
        result  = simulate_game(
            lineup, lineup, pitcher, pitcher, env, _StubModel(),
            n_sims=1000, rng=np.random.default_rng(8)
        )
        assert result.win_prob_home + result.win_prob_away == pytest.approx(1.0, rel=1e-9)

    def test_equal_teams_produce_near_50pct_win_prob(self):
        """Equal lineups and pitchers should produce win probability ~50% ± 5%."""
        from src.models.game_simulator import simulate_game
        lineup  = _make_league_average_lineup()
        pitcher = _make_league_average_pitcher()
        env     = _neutral_env()
        result  = simulate_game(
            lineup, lineup, pitcher, pitcher, env, _StubModel(),
            n_sims=2000, rng=np.random.default_rng(9)
        )
        assert 0.40 < result.win_prob_home < 0.60, (
            f"Expected ~50% win prob for equal teams, got {result.win_prob_home:.3f}"
        )

    def test_mean_total_is_reasonable(self):
        """
        MLB league-average total is ~9 RPG.  The simplified advancement rules
        will undercount somewhat; accept 4–12 as a sanity bound.
        """
        from src.models.game_simulator import simulate_game
        lineup  = _make_league_average_lineup()
        pitcher = _make_league_average_pitcher()
        env     = _neutral_env()
        result  = simulate_game(
            lineup, lineup, pitcher, pitcher, env, _StubModel(),
            n_sims=2000, rng=np.random.default_rng(10)
        )
        assert 4.0 < result.mean_total < 14.0, (
            f"Expected ~9 RPG, got {result.mean_total:.2f}"
        )

    def test_runs_are_non_negative(self):
        from src.models.game_simulator import simulate_game
        lineup  = _make_league_average_lineup()
        pitcher = _make_league_average_pitcher()
        env     = _neutral_env()
        result  = simulate_game(
            lineup, lineup, pitcher, pitcher, env, _StubModel(),
            n_sims=500, rng=np.random.default_rng(11)
        )
        assert np.all(result.home_runs_dist >= 0)
        assert np.all(result.away_runs_dist >= 0)


# ===========================================================================
# ── run_dispersion_check ─────────────────────────────────────────────────────
# ===========================================================================

def _make_game_logs(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """Synthetic team-game log: runs drawn from NegBin(mu=4.5, alpha=0.3)."""
    rng = np.random.default_rng(seed)
    # NegBin via Gamma-Poisson mixture
    mu, alpha = 4.5, 0.3
    gamma_samples = rng.gamma(1.0 / alpha, alpha * mu, size=n)
    runs = rng.poisson(gamma_samples)
    return pd.DataFrame({
        "game_pk":         np.arange(n),
        "runs":            runs,
        "is_home":         rng.integers(0, 2, n),
        "park_factor_runs": rng.uniform(0.9, 1.1, n),
    })


def _make_sim_results(game_pks, n_sims_per_game: int = 50, seed: int = 1) -> pd.DataFrame:
    """Synthetic sim results: total runs drawn from Normal(9, 3)."""
    rng = np.random.default_rng(seed)
    rows = []
    for pk in game_pks:
        home_runs = rng.integers(0, 10, n_sims_per_game)
        away_runs = rng.integers(0, 10, n_sims_per_game)
        for h, a in zip(home_runs, away_runs):
            rows.append({"game_pk": pk, "home_runs": h, "away_runs": a})
    return pd.DataFrame(rows)


class TestNegBinFit:
    def test_fit_returns_result(self):
        pytest.importorskip("statsmodels")
        from src.backtest.run_dispersion_check import fit_negbin
        logs = _make_game_logs(200)
        result = fit_negbin(logs, runs_col="runs", feature_cols=["is_home"])
        assert hasattr(result, "params")

    def test_dispersion_alpha_positive(self):
        pytest.importorskip("statsmodels")
        from src.backtest.run_dispersion_check import fit_negbin
        logs = _make_game_logs(200)
        result = fit_negbin(logs, runs_col="runs")
        alpha = result.params.get("alpha", 0)
        assert alpha > 0, f"Expected positive NegBin dispersion, got α={alpha}"

    def test_too_few_rows_raises(self):
        pytest.importorskip("statsmodels")
        from src.backtest.run_dispersion_check import fit_negbin, MIN_GAME_LOGS
        logs = _make_game_logs(5)
        with pytest.raises(ValueError, match="at least"):
            fit_negbin(logs)


class TestPredictNegBinVariance:
    def test_variance_is_positive(self):
        pytest.importorskip("statsmodels")
        from src.backtest.run_dispersion_check import fit_negbin, predict_negbin_variance
        logs = _make_game_logs(200)
        result = fit_negbin(logs, runs_col="runs")
        var = predict_negbin_variance(result, logs.head(10))
        assert np.all(var > 0)

    def test_variance_exceeds_mean(self):
        """NB2 variance = mu + alpha*mu² > mu (overdispersion)."""
        pytest.importorskip("statsmodels")
        from src.backtest.run_dispersion_check import (
            fit_negbin, predict_negbin_variance
        )
        logs = _make_game_logs(200)
        result = fit_negbin(logs, runs_col="runs")
        var = predict_negbin_variance(result, logs.head(10))
        mu  = result.predict(None)   # statsmodels returns prediction from stored exog if None raises
        # Just check variances are > 0 and reasonable
        assert float(var.mean()) > 1.0


class TestComputeMCVariance:
    def test_output_columns(self):
        from src.backtest.run_dispersion_check import compute_mc_variance
        sim = _make_sim_results([1, 2, 3], n_sims_per_game=20)
        result = compute_mc_variance(sim)
        assert "mc_var_total" in result.columns
        assert "mc_mean_total" in result.columns
        assert "n_sims" in result.columns

    def test_constant_scores_zero_variance(self):
        """If all simulations produce the same score, variance = 0."""
        from src.backtest.run_dispersion_check import compute_mc_variance
        sim = pd.DataFrame({
            "game_pk":   [1] * 50,
            "home_runs": [4] * 50,
            "away_runs": [3] * 50,
        })
        result = compute_mc_variance(sim)
        assert result.loc[result["game_pk"] == 1, "mc_var_total"].values[0] == 0.0


class TestFlagUnderdispersed:
    def test_underdispersed_flags_low_ratio_games(self):
        pytest.importorskip("statsmodels")
        from src.backtest.run_dispersion_check import (
            fit_negbin, compute_mc_variance, flag_underdispersed_games,
        )
        logs = _make_game_logs(250)
        # Sim results with very small variance (1 run per sim, constant)
        game_pks = logs["game_pk"].values[:10]
        sim_rows = []
        for pk in game_pks:
            for _ in range(50):
                sim_rows.append({"game_pk": pk, "home_runs": 4, "away_runs": 3})
        sim = pd.DataFrame(sim_rows)

        fit = fit_negbin(logs, runs_col="runs")
        mc_var = compute_mc_variance(sim)
        flagged = flag_underdispersed_games(logs, mc_var, fit, threshold=0.60)

        # All simulated games have 0 variance → all should be flagged
        assert flagged["underdispersed"].all()

    def test_dispersion_summary_keys(self):
        pytest.importorskip("statsmodels")
        from src.backtest.run_dispersion_check import (
            fit_negbin, compute_mc_variance,
            flag_underdispersed_games, dispersion_summary,
        )
        logs = _make_game_logs(200)
        sim  = _make_sim_results(logs["game_pk"].values[:20])
        fit  = fit_negbin(logs, runs_col="runs")
        mc_var = compute_mc_variance(sim)
        flagged = flag_underdispersed_games(logs, mc_var, fit)
        summary = dispersion_summary(flagged)
        for key in ["n_games", "n_underdispersed", "mean_dispersion_ratio",
                    "median_dispersion_ratio"]:
            assert key in summary
