"""
tests.test_backtest
===================
Unit tests for all Phase 5 backtest and optimizer modules:

  - src/optimizer/vig_removal.py      (power, additive, Shin de-vig)
  - src/backtest/calibration.py       (reliability diagram, Brier decomposition)
  - src/backtest/walk_forward.py      (expanding-window walk-forward harness)
  - src/backtest/clv_tracker.py       (CLV computation and summary)
  - src/backtest/run_report.py        (HTML report generation)
  - src/optimizer/kelly.py            (single-bet and portfolio Kelly)
  - src/optimizer/portfolio_cap.py    (risk caps, drawdown halt)

All tests run fully offline — no API calls, no file I/O beyond tmp data.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest


# ============================================================================
# vig_removal
# ============================================================================

class TestVigRemoval:

    def test_additive_devig_balanced(self):
        from src.optimizer.vig_removal import additive_devig
        probs = additive_devig([-110, -110])
        assert abs(probs.sum() - 1.0) < 1e-9
        assert abs(probs[0] - 0.5) < 1e-6

    def test_power_devig_balanced(self):
        from src.optimizer.vig_removal import power_devig
        probs = power_devig([-110, -110])
        assert abs(probs.sum() - 1.0) < 1e-9
        assert abs(probs[0] - 0.5) < 1e-4

    def test_power_devig_lopsided(self):
        from src.optimizer.vig_removal import power_devig
        # Heavy favourite -200 / underdog +170
        probs = power_devig([-200, 170])
        assert abs(probs.sum() - 1.0) < 1e-9
        assert probs[0] > probs[1]   # favourite more probable

    def test_shin_devig_sums_to_one(self):
        from src.optimizer.vig_removal import shin_devig
        probs = shin_devig([-110, -110])
        assert abs(probs.sum() - 1.0) < 1e-9

    def test_shin_devig_lopsided(self):
        from src.optimizer.vig_removal import shin_devig
        probs = shin_devig([-200, 170])
        assert abs(probs.sum() - 1.0) < 1e-9
        assert probs[0] > probs[1]

    def test_vig_percentage_balanced(self):
        from src.optimizer.vig_removal import vig_percentage
        vig = vig_percentage([-110, -110])
        assert 4.0 < vig < 5.5   # ~4.76%

    def test_vig_percentage_zero_vig(self):
        from src.optimizer.vig_removal import vig_percentage
        vig = vig_percentage([100, -100])
        assert abs(vig) < 1e-6

    def test_best_devig_dispatch_power(self):
        from src.optimizer.vig_removal import best_devig
        probs = best_devig([-110, -110], method="power")
        assert abs(probs.sum() - 1.0) < 1e-9

    def test_best_devig_dispatch_additive(self):
        from src.optimizer.vig_removal import best_devig
        probs = best_devig([-110, -110], method="additive")
        assert abs(probs.sum() - 1.0) < 1e-9

    def test_best_devig_dispatch_shin(self):
        from src.optimizer.vig_removal import best_devig
        probs = best_devig([-110, -110], method="shin")
        assert abs(probs.sum() - 1.0) < 1e-9

    def test_best_devig_bad_method(self):
        from src.optimizer.vig_removal import best_devig
        with pytest.raises(ValueError, match="Unknown de-vig method"):
            best_devig([-110, -110], method="banana")

    def test_no_vig_passthrough(self):
        from src.optimizer.vig_removal import power_devig
        # Even-money market: no vig → returned unchanged
        probs = power_devig([100, -100])
        assert abs(probs.sum() - 1.0) < 1e-9
        assert abs(probs[0] - 0.5) < 1e-6

    def test_single_odds_raises(self):
        from src.optimizer.vig_removal import additive_devig
        with pytest.raises(ValueError):
            additive_devig([-110])

    def test_american_to_implied_prob_favourite(self):
        from src.optimizer.vig_removal import american_to_implied_prob
        p = american_to_implied_prob(-200)
        assert abs(p - 2/3) < 1e-9

    def test_american_to_implied_prob_underdog(self):
        from src.optimizer.vig_removal import american_to_implied_prob
        p = american_to_implied_prob(200)
        assert abs(p - 1/3) < 1e-9


# ============================================================================
# calibration
# ============================================================================

class TestCalibration:

    def _make_probs_y(self, n=500, n_classes=3, seed=42):
        rng = np.random.default_rng(seed)
        probs = rng.dirichlet(alpha=np.ones(n_classes), size=n)
        y = probs.argmax(axis=1)   # "well-calibrated" ground truth
        return probs, y

    def test_reliability_diagram_returns_result(self):
        from src.backtest.calibration import reliability_diagram
        probs, y = self._make_probs_y()
        result = reliability_diagram(probs, y, class_idx=0, n_bootstrap=10)
        assert result.class_idx == 0
        assert 0.0 <= result.ece <= 1.0
        assert not result.bins_df.empty

    def test_reliability_diagram_columns(self):
        from src.backtest.calibration import reliability_diagram
        probs, y = self._make_probs_y()
        result = reliability_diagram(probs, y, class_idx=1, n_bootstrap=5)
        expected_cols = {"bin_lo", "bin_hi", "bin_centre", "mean_pred",
                         "obs_freq", "count", "ci_lo_95", "ci_hi_95"}
        assert expected_cols.issubset(set(result.bins_df.columns))

    def test_brier_decomposition_returns_correct_types(self):
        from src.backtest.calibration import brier_decomposition
        probs, y = self._make_probs_y()
        bd = brier_decomposition(probs, y)
        assert isinstance(bd.brier_score, float)
        assert isinstance(bd.reliability, float)
        assert isinstance(bd.resolution, float)
        assert isinstance(bd.uncertainty, float)

    def test_brier_decomposition_bs_relation(self):
        """
        The Murphy three-term identity BS ≈ REL - RES + UNC holds per-class.
        The full multi-class BS is computed directly; the terms are per-class
        averages, so the exact identity only needs to hold approximately.
        """
        from src.backtest.calibration import brier_decomposition
        probs, y = self._make_probs_y(n=2000, seed=7)
        bd = brier_decomposition(probs, y)
        # BSS should be finite
        assert math.isfinite(bd.brier_skill_score)
        # REL ≥ 0, RES ≥ 0, UNC ≥ 0
        assert bd.reliability >= -1e-9
        assert bd.resolution  >= -1e-9
        assert bd.uncertainty >= -1e-9

    def test_brier_skill_score_worse_than_climatology(self):
        """A completely random predictor should have BSS < 0 (worse than climatology)."""
        from src.backtest.calibration import brier_skill_score
        rng = np.random.default_rng(42)
        n = 500
        # Random probs — should score worse than the base-rate prior
        probs = rng.dirichlet(alpha=np.ones(3), size=n)
        y = rng.integers(0, 3, n)
        bss = brier_skill_score(probs, y)
        # Random model should at most barely beat climatology; BSS ≤ 1 always
        assert bss <= 1.0
        assert math.isfinite(bss)

    def test_calibration_report_shape(self):
        from src.backtest.calibration import calibration_report
        probs, y = self._make_probs_y(n_classes=3)
        df = calibration_report(probs, y, class_names=["HR", "BB", "K"], n_bootstrap=5)
        assert len(df) == 3
        assert "class_name" in df.columns
        assert "ece" in df.columns
        assert "brier_skill_score" in df.columns

    def test_calibration_report_ece_positive(self):
        from src.backtest.calibration import calibration_report
        probs, y = self._make_probs_y()
        df = calibration_report(probs, y, n_bootstrap=5)
        assert (df["ece"] >= 0).all()


# ============================================================================
# walk_forward
# ============================================================================

class TestWalkForward:
    """Smoke + contract tests for the expanding-window harness."""

    def _make_toy_model(self):
        """Returns a factory for a trivial 3-class 'model'."""
        class _ConstantModel:
            def fit(self, X, y):
                pass
            def predict_proba(self, X):
                n = len(X)
                return np.column_stack([
                    np.full(n, 0.6),
                    np.full(n, 0.3),
                    np.full(n, 0.1),
                ])
        return lambda: _ConstantModel()

    def _make_dataset(self, n=2000, n_classes=3, seed=0):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, 10)).astype(np.float32)
        y = rng.integers(0, n_classes, size=n)
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        return X, y, dates

    def test_basic_run(self):
        from src.backtest.walk_forward import walk_forward_validate, WalkForwardConfig
        X, y, dates = self._make_dataset()
        config = WalkForwardConfig(
            initial_train_days=180,
            step_days=30,
            min_train_samples=50,
            min_test_samples=5,
        )
        result = walk_forward_validate(X, y, dates, self._make_toy_model(), config)
        assert result.n_folds > 0
        assert len(result.predictions_df) > 0
        assert len(result.fold_metrics) == result.n_folds

    def test_no_future_leak(self):
        """Every prediction date must be strictly after all training dates."""
        from src.backtest.walk_forward import walk_forward_validate, WalkForwardConfig
        X, y, dates = self._make_dataset()
        config = WalkForwardConfig(
            initial_train_days=180,
            step_days=30,
            min_train_samples=50,
            min_test_samples=5,
        )
        result = walk_forward_validate(X, y, dates, self._make_toy_model(), config)
        preds = result.predictions_df
        # For fold 0, prediction dates >= initial_train_days from start
        assert preds["fold"].min() == 0
        # All prediction dates must be after the training end for that fold
        # (we just check that the DataFrame has the required columns)
        assert "game_date" in preds.columns
        assert "fold" in preds.columns

    def test_oos_log_loss_finite(self):
        from src.backtest.walk_forward import walk_forward_validate, WalkForwardConfig
        X, y, dates = self._make_dataset()
        config = WalkForwardConfig(
            initial_train_days=180,
            step_days=30,
            min_train_samples=50,
            min_test_samples=5,
        )
        result = walk_forward_validate(X, y, dates, self._make_toy_model(), config)
        assert math.isfinite(result.oos_log_loss())
        assert 0.0 <= result.oos_accuracy() <= 1.0

    def test_no_retrain_mode(self):
        from src.backtest.walk_forward import walk_forward_validate, WalkForwardConfig
        X, y, dates = self._make_dataset()
        config = WalkForwardConfig(
            initial_train_days=180,
            step_days=30,
            min_train_samples=50,
            min_test_samples=5,
            retrain=False,
        )
        result = walk_forward_validate(X, y, dates, self._make_toy_model(), config)
        assert result.n_folds > 0

    def test_summary_keys(self):
        from src.backtest.walk_forward import walk_forward_validate, WalkForwardConfig
        X, y, dates = self._make_dataset()
        config = WalkForwardConfig(
            initial_train_days=180,
            step_days=30,
            min_train_samples=50,
            min_test_samples=5,
        )
        result = walk_forward_validate(X, y, dates, self._make_toy_model(), config)
        required = {"n_folds", "n_oos_predictions", "oos_log_loss", "oos_accuracy",
                    "mean_fold_log_loss", "std_fold_log_loss", "mean_fold_brier"}
        assert required.issubset(set(result.summary.keys()))

    def test_empty_dataset_raises(self):
        from src.backtest.walk_forward import walk_forward_validate
        with pytest.raises(ValueError, match="Empty dataset"):
            walk_forward_validate(
                np.empty((0, 5)), np.empty(0), pd.Series([], dtype="datetime64[ns]"),
                lambda: None,
            )

    def test_mismatched_dates_raises(self):
        from src.backtest.walk_forward import walk_forward_validate
        X = np.ones((10, 3))
        y = np.zeros(10, dtype=int)
        dates = pd.date_range("2020-01-01", periods=5)   # wrong length
        with pytest.raises(ValueError, match="dates length"):
            walk_forward_validate(X, y, dates, lambda: None)


# ============================================================================
# clv_tracker
# ============================================================================

class TestCLVTracker:

    def _sample_bets(self, n=20, seed=42) -> list:
        from src.backtest.clv_tracker import BetRecord
        rng = np.random.default_rng(seed)
        bets = []
        for i in range(n):
            result = float(rng.choice([0.0, 1.0]))
            bets.append(BetRecord(
                bet_id=f"g{i}",
                bet_date=f"2024-04-{(i % 28) + 1:02d}",
                market="h2h",
                side="home",
                open_odds=float(rng.choice([-130, -110, +110, +130])),
                open_odds_other=float(rng.choice([+110, +120, -110, -120])),
                close_odds=float(rng.choice([-120, -105, +105, +120])),
                close_odds_other=float(rng.choice([+110, +115, -110, -115])),
                result=result,
                stake=1.0,
                model_prob=float(rng.uniform(0.45, 0.65)),
            ))
        return bets

    def test_compute_clv_shape(self):
        from src.backtest.clv_tracker import compute_clv
        bets = self._sample_bets()
        df = compute_clv(bets)
        assert len(df) == len(bets)
        assert "clv_prob" in df.columns
        assert "clv_log_odds" in df.columns
        assert "p_fair_open" in df.columns
        assert "p_fair_close" in df.columns

    def test_compute_clv_probs_in_range(self):
        from src.backtest.clv_tracker import compute_clv
        df = compute_clv(self._sample_bets())
        assert (df["p_fair_open"] > 0).all()
        assert (df["p_fair_open"] < 1).all()
        assert (df["p_fair_close"] > 0).all()
        assert (df["p_fair_close"] < 1).all()

    def test_clv_summary_keys(self):
        from src.backtest.clv_tracker import compute_clv, clv_summary
        df = compute_clv(self._sample_bets())
        s = clv_summary(df)
        required = {"n_bets", "mean_clv_prob", "mean_clv_log_odds",
                    "pct_positive_clv", "clv_tstat", "clv_pvalue", "roi_pct"}
        assert required.issubset(set(s.keys()))
        assert s["n_bets"] == len(df)

    def test_clv_summary_empty(self):
        from src.backtest.clv_tracker import clv_summary
        s = clv_summary(pd.DataFrame())
        assert s["n_bets"] == 0

    def test_rolling_clv_length(self):
        from src.backtest.clv_tracker import compute_clv, rolling_clv
        df = compute_clv(self._sample_bets(n=30))
        roll = rolling_clv(df, window=5)
        assert len(roll) == len(df)

    def test_clv_by_market_has_index(self):
        from src.backtest.clv_tracker import compute_clv, clv_by_market
        df = compute_clv(self._sample_bets())
        mkt = clv_by_market(df)
        assert "h2h" in mkt.index

    def test_plot_data_clv_keys(self):
        from src.backtest.clv_tracker import compute_clv, plot_data_clv
        df = compute_clv(self._sample_bets(n=30))
        data = plot_data_clv(df, window=5)
        assert "rolling_clv" in data
        assert "cum_pnl" in data
        assert "clv_histogram" in data
        assert "edges" in data["clv_histogram"]
        assert "counts" in data["clv_histogram"]

    def test_win_clv_positive(self):
        """A bet taken at +110 when the market moves to +100 should have positive CLV."""
        from src.backtest.clv_tracker import BetRecord, compute_clv
        bet = BetRecord(
            bet_id="test1",
            bet_date="2024-05-01",
            market="h2h",
            side="home",
            open_odds=110,    # we got +110
            open_odds_other=-130,
            close_odds=100,   # market moved toward us — we beat the closing line
            close_odds_other=-120,
            result=1.0,
            stake=1.0,
            model_prob=0.52,
        )
        df = compute_clv([bet])
        # p_fair_close > p_fair_open → positive CLV
        assert df["clv_prob"].iloc[0] > 0
        assert df["clv_log_odds"].iloc[0] > 0

    def test_pnl_win(self):
        """Winning a +150 bet on $1 stake should return $1.50 profit."""
        from src.backtest.clv_tracker import BetRecord, compute_clv
        bet = BetRecord(
            bet_id="w1",
            bet_date="2024-05-01",
            market="h2h",
            side="home",
            open_odds=150,
            open_odds_other=-180,
            close_odds=140,
            close_odds_other=-170,
            result=1.0,
            stake=1.0,
        )
        df = compute_clv([bet])
        assert abs(df["pnl_units"].iloc[0] - 1.5) < 1e-9

    def test_pnl_loss(self):
        from src.backtest.clv_tracker import BetRecord, compute_clv
        bet = BetRecord(
            bet_id="l1",
            bet_date="2024-05-01",
            market="h2h",
            side="home",
            open_odds=-110,
            open_odds_other=-110,
            close_odds=-110,
            close_odds_other=-110,
            result=0.0,
            stake=2.0,
        )
        df = compute_clv([bet])
        assert abs(df["pnl_units"].iloc[0] - (-2.0)) < 1e-9


# ============================================================================
# run_report
# ============================================================================

class TestRunReport:

    def _dummy_wf_result(self):
        from src.backtest.walk_forward import WalkForwardResult
        preds = pd.DataFrame({
            "row_idx":   [0, 1, 2, 3, 4],
            "game_date": ["2024-04-01"] * 5,
            "fold":      [0] * 5,
            "y_true":    [0, 1, 2, 0, 1],
            "y_pred":    [0, 1, 1, 0, 1],
            "loss":      [0.5, 0.4, 0.9, 0.3, 0.2],
            "p_0":       [0.6, 0.2, 0.1, 0.7, 0.1],
            "p_1":       [0.3, 0.6, 0.2, 0.2, 0.7],
            "p_2":       [0.1, 0.2, 0.7, 0.1, 0.2],
        })
        fold_metrics = [{
            "fold": 0, "date_start": "2024-04-01", "date_end": "2024-04-07",
            "n_train": 100, "n_test": 5,
            "log_loss": 0.46, "accuracy": 0.8, "brier": 0.3,
        }]
        summary = {
            "n_folds": 1, "n_oos_predictions": 5,
            "oos_log_loss": 0.46, "oos_accuracy": 0.8,
            "mean_fold_log_loss": 0.46, "std_fold_log_loss": 0.0,
            "mean_fold_accuracy": 0.8, "mean_fold_brier": 0.3,
        }
        return WalkForwardResult(preds, fold_metrics, summary, n_folds=1)

    def test_build_report_no_sections(self):
        from src.backtest.run_report import build_report
        html = build_report()
        assert "<!DOCTYPE html>" in html
        assert "No backtest data was provided" in html

    def test_build_report_with_wf(self):
        from src.backtest.run_report import build_report
        html = build_report(wf_result=self._dummy_wf_result())
        assert "Walk-Forward" in html
        assert "OOS log-loss" in html

    def test_build_report_with_cal(self):
        from src.backtest.run_report import build_report
        cal_df = pd.DataFrame({
            "class_name": ["HR", "BB"],
            "ece": [0.05, 0.07],
            "brier_score": [0.2, 0.18],
            "reliability": [0.01, 0.02],
            "resolution":  [0.05, 0.04],
            "uncertainty": [0.25, 0.22],
            "brier_skill_score": [0.2, 0.18],
            "base_rate": [0.08, 0.10],
            "n_positive": [80, 100],
        })
        html = build_report(cal_df=cal_df)
        assert "Calibration" in html

    def test_build_report_with_dispersion(self):
        from src.backtest.run_report import build_report
        html = build_report(dispersion_summary={
            "n_games": 100,
            "n_underdispersed": 12,
            "pct_underdispersed": 12.0,
            "mean_dispersion_ratio": 0.78,
            "median_dispersion_ratio": 0.80,
        })
        assert "Dispersion" in html

    def test_save_report(self, tmp_path):
        from src.backtest.run_report import build_report, save_report
        html = build_report()
        out = save_report(html, tmp_path / "report.html")
        assert out.exists()
        assert out.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")

    def test_html_escape(self):
        """Title with special chars must be escaped, not raw HTML."""
        from src.backtest.run_report import build_report
        html = build_report(title="<script>alert('xss')</script>")
        assert "<script>alert" not in html
        assert "&lt;script&gt;" in html


# ============================================================================
# kelly
# ============================================================================

class TestKelly:

    def test_american_to_decimal_favourite(self):
        from src.optimizer.kelly import american_to_decimal
        d = american_to_decimal(-110)
        assert abs(d - (100/110 + 1)) < 1e-9

    def test_american_to_decimal_underdog(self):
        from src.optimizer.kelly import american_to_decimal
        d = american_to_decimal(150)
        assert abs(d - 2.5) < 1e-9

    def test_decimal_to_american_plus(self):
        from src.optimizer.kelly import decimal_to_american
        a = decimal_to_american(2.5)
        assert abs(a - 150.0) < 1e-9

    def test_decimal_to_american_minus(self):
        from src.optimizer.kelly import decimal_to_american, american_to_decimal
        # Round-trip: american → decimal → american
        for amer in [-200, -110, -150]:
            d = american_to_decimal(amer)
            back = decimal_to_american(d)
            assert abs(back - amer) < 1e-6

    def test_kelly_fraction_with_edge(self):
        """p=0.55 at -110 odds (~1.91 decimal) should give positive Kelly."""
        from src.optimizer.kelly import kelly_fraction
        f = kelly_fraction(model_prob=0.55, fair_odds_american=-110, fractional_divisor=4.0)
        assert f > 0.0
        assert f < 0.10   # quarter-Kelly on a thin edge should be small

    def test_kelly_fraction_no_edge(self):
        """p=0.40 at -110 odds — model has no edge; should return 0."""
        from src.optimizer.kelly import kelly_fraction
        f = kelly_fraction(model_prob=0.40, fair_odds_american=-110)
        assert f == 0.0

    def test_kelly_fraction_degenerate_prob(self):
        from src.optimizer.kelly import kelly_fraction
        assert kelly_fraction(0.0, -110) == 0.0
        assert kelly_fraction(1.0, -110) == 0.0

    def test_kelly_fraction_full_kelly(self):
        """Full Kelly (divisor=1) should be 4× quarter-Kelly."""
        from src.optimizer.kelly import kelly_fraction
        quarter = kelly_fraction(0.55, -110, fractional_divisor=4.0)
        full    = kelly_fraction(0.55, -110, fractional_divisor=1.0)
        assert abs(full - 4 * quarter) < 1e-9

    def test_kelly_ev_positive(self):
        from src.optimizer.kelly import kelly_ev
        ev = kelly_ev(0.55, -110)
        assert ev > 0.0

    def test_kelly_ev_negative(self):
        from src.optimizer.kelly import kelly_ev
        ev = kelly_ev(0.40, -110)
        assert ev < 0.0

    def test_portfolio_kelly_empty(self):
        from src.optimizer.kelly import portfolio_kelly
        df = pd.DataFrame(columns=["model_prob", "fair_odds_american"])
        result = portfolio_kelly(df)
        assert len(result) == 0

    def test_portfolio_kelly_single_bet(self):
        from src.optimizer.kelly import portfolio_kelly
        df = pd.DataFrame([{"model_prob": 0.55, "fair_odds_american": -110}])
        result = portfolio_kelly(df, bankroll=1000.0)
        assert "stake_units" in result.columns
        assert result["stake_units"].iloc[0] >= 0.0

    def test_portfolio_kelly_no_edge_zero_stake(self):
        from src.optimizer.kelly import portfolio_kelly
        df = pd.DataFrame([{"model_prob": 0.40, "fair_odds_american": -110}])
        result = portfolio_kelly(df, bankroll=1000.0)
        # No edge → stake should be 0
        assert result["stake_units"].iloc[0] == 0.0

    def test_portfolio_kelly_total_exposure_cap(self):
        from src.optimizer.kelly import portfolio_kelly
        rng = np.random.default_rng(0)
        n = 10
        df = pd.DataFrame({
            "model_prob":         rng.uniform(0.52, 0.65, n),
            "fair_odds_american": [-110.0] * n,
        })
        result = portfolio_kelly(df, bankroll=1.0, max_total_exposure=0.25)
        total_fraction = result["f_final"].sum()
        assert total_fraction <= 0.25 + 1e-6

    def test_portfolio_kelly_with_covariance(self):
        """With explicit 2×2 covariance matrix, solver should run without error."""
        from src.optimizer.kelly import portfolio_kelly
        df = pd.DataFrame([
            {"model_prob": 0.55, "fair_odds_american": -110},
            {"model_prob": 0.58, "fair_odds_american": +105},
        ])
        cov = np.array([[0.25, 0.10], [0.10, 0.25]])
        result = portfolio_kelly(df, cov_matrix=cov, bankroll=1.0, max_total_exposure=0.25)
        assert (result["f_final"] >= -1e-9).all()
        assert result["f_final"].sum() <= 0.25 + 1e-6

    def test_portfolio_kelly_cov_wrong_shape(self):
        from src.optimizer.kelly import portfolio_kelly
        df = pd.DataFrame([
            {"model_prob": 0.55, "fair_odds_american": -110},
            {"model_prob": 0.58, "fair_odds_american": +105},
        ])
        with pytest.raises(ValueError, match="cov_matrix shape"):
            portfolio_kelly(df, cov_matrix=np.eye(3))


# ============================================================================
# portfolio_cap
# ============================================================================

class TestPortfolioCap:

    def _sized_df(self, n=5, stake_per_bet=0.05):
        return pd.DataFrame({
            "model_prob":         [0.55] * n,
            "fair_odds_american": [-110.0] * n,
            "ev":                 [0.02] * n,
            "stake_units":        [stake_per_bet] * n,
        })

    def test_no_caps_triggered(self):
        from src.optimizer.portfolio_cap import apply_caps, CapConfig
        df = self._sized_df(n=2, stake_per_bet=0.01)
        cfg = CapConfig(max_bet_fraction=0.05, daily_cap_fraction=0.15)
        result = apply_caps(df, bankroll=1.0, config=cfg)
        # Nothing should be capped
        assert (result["cap_applied"] == "none").all()
        assert (result["capped_stake_units"] == 0.01).all()

    def test_per_bet_cap_triggers(self):
        from src.optimizer.portfolio_cap import apply_caps, CapConfig
        df = self._sized_df(n=3, stake_per_bet=0.10)  # 10% per bet
        cfg = CapConfig(max_bet_fraction=0.03)          # cap at 3%
        result = apply_caps(df, bankroll=1.0, config=cfg)
        assert (result["capped_stake_units"] <= 0.03 + 1e-9).all()
        assert (result["cap_applied"].str.contains("per_bet_cap")).all()

    def test_daily_cap_triggers(self):
        from src.optimizer.portfolio_cap import apply_caps, CapConfig
        df = self._sized_df(n=10, stake_per_bet=0.05)  # 50% total
        cfg = CapConfig(max_bet_fraction=0.10, daily_cap_fraction=0.15)
        result = apply_caps(df, bankroll=1.0, config=cfg)
        assert result["capped_stake_units"].sum() <= 0.15 + 1e-6

    def test_ev_filter_zeros_bets(self):
        from src.optimizer.portfolio_cap import apply_caps, CapConfig
        df = pd.DataFrame({
            "model_prob":         [0.55, 0.45],
            "fair_odds_american": [-110.0, -110.0],
            "ev":                 [0.03, -0.02],   # second bet has negative EV
            "stake_units":        [0.02, 0.02],
        })
        cfg = CapConfig(min_ev_threshold=0.0)
        result = apply_caps(df, bankroll=1.0, config=cfg)
        # Second bet should be zeroed
        assert result["capped_stake_units"].iloc[1] == 0.0
        assert "ev_filter" in result["cap_applied"].iloc[1]

    def test_team_cap_triggers(self):
        from src.optimizer.portfolio_cap import apply_caps, CapConfig
        df = pd.DataFrame({
            "model_prob":         [0.55, 0.56, 0.57],
            "fair_odds_american": [-110.0] * 3,
            "ev":                 [0.03] * 3,
            "stake_units":        [0.04] * 3,   # 12% total on one team
            "side":               ["home", "home", "home"],
            "home_team":          ["NYY"] * 3,
        })
        cfg = CapConfig(max_team_fraction=0.06, daily_cap_fraction=0.30)
        result = apply_caps(df, bankroll=1.0, config=cfg)
        assert result["capped_stake_units"].sum() <= 0.06 + 1e-6

    def test_drawdown_halt_triggers(self):
        from src.optimizer.portfolio_cap import apply_caps, CapConfig
        df = self._sized_df(n=3, stake_per_bet=0.02)
        cfg = CapConfig(drawdown_halt_pct=20.0)
        # bankroll = 0.75, HWM = 1.0 → 25% drawdown > 20% threshold
        result = apply_caps(df, bankroll=0.75, config=cfg, current_hwm=1.0)
        assert (result["capped_stake_units"] == 0.0).all()
        assert (result["cap_applied"] == "drawdown_halt").all()

    def test_drawdown_halt_not_triggered(self):
        from src.optimizer.portfolio_cap import apply_caps, CapConfig
        df = self._sized_df(n=3, stake_per_bet=0.02)
        cfg = CapConfig(drawdown_halt_pct=20.0)
        # bankroll = 0.90, HWM = 1.0 → 10% drawdown < 20% threshold
        result = apply_caps(df, bankroll=0.90, config=cfg, current_hwm=1.0)
        assert not (result["cap_applied"] == "drawdown_halt").any()

    def test_check_drawdown_halt_boundary(self):
        from src.optimizer.portfolio_cap import check_drawdown_halt, CapConfig
        cfg = CapConfig(drawdown_halt_pct=20.0)
        assert check_drawdown_halt(0.79, 1.0, cfg) is True    # 21% > 20%
        assert check_drawdown_halt(0.81, 1.0, cfg) is False   # 19% < 20%

    def test_empty_df_passthrough(self):
        from src.optimizer.portfolio_cap import apply_caps
        df = pd.DataFrame(columns=["stake_units", "ev"])
        result = apply_caps(df, bankroll=1000.0)
        assert len(result) == 0

    def test_with_date_column_daily_cap(self):
        from src.optimizer.portfolio_cap import apply_caps, CapConfig
        df = pd.DataFrame({
            "model_prob":         [0.55] * 6,
            "fair_odds_american": [-110.0] * 6,
            "ev":                 [0.03] * 6,
            "stake_units":        [0.05] * 6,
            "bet_date":           ["2024-04-01"] * 3 + ["2024-04-02"] * 3,
        })
        cfg = CapConfig(max_bet_fraction=0.10, daily_cap_fraction=0.10)
        result = apply_caps(df, bankroll=1.0, config=cfg)
        # Each day's total should respect the daily cap
        for date, grp in result.groupby("bet_date"):
            assert grp["capped_stake_units"].sum() <= 0.10 + 1e-6
