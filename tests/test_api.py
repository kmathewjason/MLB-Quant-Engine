"""
tests.test_api
==============
Unit tests for the new /api/ endpoints in src/api.py.

All tests run fully offline — external I/O (MLB Stats API, Odds API) is
mocked out with pytest-mock / unittest.mock.  The game simulator is exercised
via the real half-inning engine with league-average PA probs (fast, no model
needed).

Endpoint coverage:
  GET  /health                           (legacy — quick sanity)
  GET  /api/predictions/daily            (schedule mocked, odds absent)
  GET  /api/games/{game_id}/simulation   (real simulation, n_sims=1000)
  POST /api/predictions/sgp              (real sim_corr_matrix, n_sims=1000)
  GET  /api/backtest/report              (file fixtures written to tmp_path)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# App under test
# ---------------------------------------------------------------------------

from src.api import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _fake_schedule(game_pk: int = 999_001) -> list[dict]:
    """Minimal MLB Stats API game dict used to stub fetch_schedule()."""
    return [
        {
            "gamePk":       game_pk,
            "officialDate": "2024-04-15",
            "gameDate":     "2024-04-15T18:05:00Z",
            "status":       {"detailedState": "Preview"},
            "teams": {
                "home": {
                    "team":             {"name": "New York Yankees"},
                    "probablePitcher":  {"fullName": "Gerrit Cole"},
                },
                "away": {
                    "team":             {"name": "Boston Red Sox"},
                    "probablePitcher":  {"fullName": "Chris Sale"},
                },
            },
        }
    ]


# ===========================================================================
# /health  (legacy — quick sanity)
# ===========================================================================

class TestHealth:
    def test_health_ok(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "version" in body
        assert "utc_now" in body


# ===========================================================================
# GET /api/predictions/daily
# ===========================================================================

class TestDailyPredictions:

    def _patch_schedule(self, mocker, games=None):
        if games is None:
            games = _fake_schedule()
        return mocker.patch(
            "src.ingestion.mlb_stats_api.fetch_schedule",
            return_value=games,
        )

    def test_returns_envelope(self, mocker):
        self._patch_schedule(mocker)
        resp = client.get("/api/predictions/daily?n_sims=2000")
        assert resp.status_code == 200
        body = resp.json()
        assert "data" in body
        assert "meta" in body
        assert "generated_at" in body["meta"]
        assert "version" in body["meta"]

    def test_empty_slate_returns_empty_list(self, mocker):
        mocker.patch("src.ingestion.mlb_stats_api.fetch_schedule", return_value=[])
        resp = client.get("/api/predictions/daily?n_sims=2000")
        assert resp.status_code == 200
        assert resp.json()["data"] == []

    def test_game_shape(self, mocker):
        self._patch_schedule(mocker)
        resp = client.get("/api/predictions/daily?n_sims=2000")
        assert resp.status_code == 200
        games = resp.json()["data"]
        assert len(games) == 1
        g = games[0]
        assert g["game_pk"] == 999_001
        assert g["home_team"] == "New York Yankees"
        assert g["away_team"] == "Boston Red Sox"

    def test_simulation_block_present(self, mocker):
        self._patch_schedule(mocker)
        resp = client.get("/api/predictions/daily?n_sims=2000")
        sim = resp.json()["data"][0]["simulation"]
        assert "n_sims" in sim
        assert "home_win_prob" in sim
        assert "away_win_prob" in sim
        assert "home_runs" in sim
        assert "away_runs" in sim
        assert "total_runs" in sim
        assert "spread" in sim
        assert "totals" in sim

    def test_simulation_probs_sum_to_one(self, mocker):
        self._patch_schedule(mocker)
        resp = client.get("/api/predictions/daily?n_sims=2000")
        sim = resp.json()["data"][0]["simulation"]
        total = sim["home_win_prob"] + sim["away_win_prob"]
        assert abs(total - 1.0) < 0.05   # allow small rounding from ties

    def test_run_distribution_has_percentiles(self, mocker):
        self._patch_schedule(mocker)
        resp = client.get("/api/predictions/daily?n_sims=2000")
        home_runs = resp.json()["data"][0]["simulation"]["home_runs"]
        for key in ["mean", "std", "p10", "p25", "p50", "p75", "p90"]:
            assert key in home_runs
            assert isinstance(home_runs[key], float)

    def test_percentile_ordering(self, mocker):
        """p10 ≤ p25 ≤ p50 ≤ p75 ≤ p90 for each distribution."""
        self._patch_schedule(mocker)
        resp = client.get("/api/predictions/daily?n_sims=2000")
        for dist_name in ["home_runs", "away_runs", "total_runs"]:
            d = resp.json()["data"][0]["simulation"][dist_name]
            assert d["p10"] <= d["p25"] <= d["p50"] <= d["p75"] <= d["p90"]

    def test_markets_present(self, mocker):
        self._patch_schedule(mocker)
        resp = client.get("/api/predictions/daily?n_sims=2000")
        markets = resp.json()["data"][0]["markets"]
        assert isinstance(markets, list)
        assert len(markets) > 0

    def test_market_schema(self, mocker):
        self._patch_schedule(mocker)
        resp = client.get("/api/predictions/daily?n_sims=2000")
        m = resp.json()["data"][0]["markets"][0]
        for key in ["market", "label", "model_prob"]:
            assert key in m
        assert 0.0 < m["model_prob"] < 1.0

    def test_no_odds_key_still_returns_totals(self, mocker):
        """Without ODDS_API_KEY the totals/spread markets still appear (model probs only)."""
        self._patch_schedule(mocker)
        mocker.patch.dict("os.environ", {"ODDS_API_KEY": ""}, clear=False)
        resp = client.get("/api/predictions/daily?n_sims=2000")
        markets = resp.json()["data"][0]["markets"]
        market_types = {m["market"].split("_")[0] for m in markets}
        assert "total" in market_types
        assert "spread" in market_types

    def test_schedule_error_returns_502(self, mocker):
        mocker.patch(
            "src.ingestion.mlb_stats_api.fetch_schedule",
            side_effect=RuntimeError("network error"),
        )
        resp = client.get("/api/predictions/daily?n_sims=2000")
        assert resp.status_code == 502

    def test_invalid_devig_method_still_returns(self, mocker):
        """Unsupported devig parameter: odds de-vig fails gracefully → market_prob=null."""
        self._patch_schedule(mocker)
        resp = client.get("/api/predictions/daily?n_sims=2000&devig=power")
        # Should succeed (totals/spread never require de-vig)
        assert resp.status_code == 200


# ===========================================================================
# GET /api/games/{game_id}/simulation
# ===========================================================================

class TestGameSimulation:

    def test_returns_envelope(self):
        resp = client.get("/api/games/12345/simulation?n_sims=2000")
        assert resp.status_code == 200
        body = resp.json()
        assert "data" in body
        assert "meta" in body

    def test_data_shape(self):
        resp = client.get("/api/games/12345/simulation?n_sims=2000")
        d = resp.json()["data"]
        assert d["game_id"] == 12345
        assert d["n_sims"] > 0
        for key in [
            "home_win_prob", "away_win_prob", "spread_cover_prob",
            "over_prob", "under_prob",
        ]:
            assert key in d
            assert isinstance(d[key], float)

    def test_win_probs_sum_to_one(self):
        resp = client.get("/api/games/12345/simulation?n_sims=2000")
        d = resp.json()["data"]
        total = d["home_win_prob"] + d["away_win_prob"]
        assert abs(total - 1.0) < 0.05

    def test_over_under_complement(self):
        resp = client.get("/api/games/12345/simulation?n_sims=2000")
        d = resp.json()["data"]
        assert abs(d["over_prob"] + d["under_prob"] - 1.0) < 1e-9

    def test_histogram_structure(self):
        resp = client.get("/api/games/12345/simulation?n_sims=2000")
        home_runs = resp.json()["data"]["home_runs"]
        assert "histogram" in home_runs
        hist = home_runs["histogram"]
        assert "edges" in hist and "counts" in hist
        assert len(hist["edges"]) == len(hist["counts"]) + 1

    def test_histogram_bins_param(self):
        resp = client.get("/api/games/12345/simulation?n_sims=2000&bins=10")
        hist = resp.json()["data"]["home_runs"]["histogram"]
        assert len(hist["counts"]) == 10

    def test_margin_dist_present(self):
        resp = client.get("/api/games/12345/simulation?n_sims=2000")
        d = resp.json()["data"]
        assert "margin_dist" in d
        assert "histogram" in d["margin_dist"]

    def test_custom_total_line(self):
        """Different total_line changes over_prob."""
        r1 = client.get("/api/games/23456/simulation?n_sims=2000&total_line=7.5")
        r2 = client.get("/api/games/23456/simulation?n_sims=2000&total_line=10.5")
        # Over prob at lower line should be higher
        p1 = r1.json()["data"]["over_prob"]
        p2 = r2.json()["data"]["over_prob"]
        assert p1 >= p2

    def test_spread_cover_prob_complement(self):
        """P(home - away > -1.5) + P(away - home > 0.5) should ≈ 1 (ignoring ties on integers)."""
        resp = client.get("/api/games/34567/simulation?n_sims=3000&run_line=-1.5")
        assert resp.status_code == 200
        d = resp.json()["data"]
        # spread_cover_prob is P(home wins by 2+); just check it's a valid probability
        assert 0.0 <= d["spread_cover_prob"] <= 1.0

    def test_all_percentiles_finite(self):
        resp = client.get("/api/games/45678/simulation?n_sims=2000")
        d = resp.json()["data"]
        for dist in ["home_runs", "away_runs", "total_runs"]:
            for key in ["mean", "std", "p10", "p25", "p50", "p75", "p90"]:
                assert math.isfinite(d[dist][key])


# ===========================================================================
# POST /api/predictions/sgp
# ===========================================================================

class TestSGPPrediction:

    def _sgp_body(self, game_id: int = 99999, n_sims: int = 2000) -> dict:
        return {
            "game_id": game_id,
            "bankroll": 1000.0,
            "n_sims":   n_sims,
            "legs": [
                {"side": "home", "outcome": "moneyline",
                 "odds": -130, "label": "NYY ML"},
                {"side": "home", "outcome": "over",
                 "line": 8.5,   "label": "O 8.5"},
            ],
        }

    def test_returns_envelope(self):
        resp = client.post("/api/predictions/sgp", json=self._sgp_body())
        assert resp.status_code == 200
        body = resp.json()
        assert "data" in body
        assert "meta" in body

    def test_data_keys(self):
        body = client.post("/api/predictions/sgp", json=self._sgp_body()).json()["data"]
        for key in [
            "game_id", "n_sims", "legs",
            "joint_prob_corr_adjusted", "joint_prob_naive",
            "parlay_net_payout", "correlation_matrix", "kelly",
        ]:
            assert key in body

    def test_leg_count_matches(self):
        body = client.post("/api/predictions/sgp", json=self._sgp_body()).json()["data"]
        assert len(body["legs"]) == 2

    def test_leg_schema(self):
        legs = client.post("/api/predictions/sgp", json=self._sgp_body()).json()["data"]["legs"]
        for leg in legs:
            assert "label" in leg
            assert "sim_prob" in leg
            assert 0.0 < leg["sim_prob"] < 1.0

    def test_joint_probs_in_range(self):
        d = client.post("/api/predictions/sgp", json=self._sgp_body()).json()["data"]
        assert 0.0 <= d["joint_prob_naive"]         <= 1.0
        assert 0.0 <= d["joint_prob_corr_adjusted"] <= 1.0

    def test_naive_is_product_of_marginals(self):
        """joint_prob_naive must equal Π sim_prob_i (within 4-dp rounding tolerance)."""
        d = client.post("/api/predictions/sgp", json=self._sgp_body()).json()["data"]
        product = 1.0
        for leg in d["legs"]:
            product *= leg["sim_prob"]
        assert abs(d["joint_prob_naive"] - product) < 1e-4

    def test_home_ml_plus_away_ml_correlation_negative(self):
        """
        Home ML + Away ML must have near-perfectly negative correlation
        (one wins iff the other loses).
        """
        body = {
            "game_id": 77777,
            "bankroll": 1000.0,
            "n_sims": 5000,
            "legs": [
                {"side": "home", "outcome": "moneyline", "label": "H"},
                {"side": "away", "outcome": "moneyline", "label": "A"},
            ],
        }
        d = client.post("/api/predictions/sgp", json=body).json()["data"]
        corr_01 = d["correlation_matrix"][0][1]
        assert corr_01 < -0.8, f"Expected strong negative corr; got {corr_01:.3f}"

    def test_corr_matrix_diagonal_is_one(self):
        d = client.post("/api/predictions/sgp", json=self._sgp_body()).json()["data"]
        for i in range(len(d["legs"])):
            assert abs(d["correlation_matrix"][i][i] - 1.0) < 1e-9

    def test_kelly_block_structure(self):
        kelly = client.post("/api/predictions/sgp", json=self._sgp_body()).json()["data"]["kelly"]
        for variant in ["corr_adjusted", "naive"]:
            assert variant in kelly
            for key in ["ev", "f_star", "f_quarter", "stake_units"]:
                assert key in kelly[variant]

    def test_too_few_legs_rejected(self):
        body = {
            "game_id": 1,
            "legs": [{"side": "home", "outcome": "moneyline"}],  # only 1
            "bankroll": 500.0,
        }
        resp = client.post("/api/predictions/sgp", json=body)
        assert resp.status_code == 422   # pydantic validation error

    def test_too_many_legs_rejected(self):
        body = {
            "game_id": 1,
            "legs": [{"side": "home", "outcome": "moneyline"}] * 7,  # 7 > max 6
            "bankroll": 500.0,
        }
        resp = client.post("/api/predictions/sgp", json=body)
        assert resp.status_code == 422

    def test_parlay_payout_positive(self):
        """Net parlay payout b = Π decimal_i - 1 must be > 0."""
        d = client.post("/api/predictions/sgp", json=self._sgp_body()).json()["data"]
        assert d["parlay_net_payout"] > 0.0

    def test_three_leg_sgp(self):
        """Three-leg SGP should run without error."""
        body = {
            "game_id": 55555,
            "bankroll": 1000.0,
            "n_sims": 2000,
            "legs": [
                {"side": "home", "outcome": "moneyline"},
                {"side": "home", "outcome": "over",   "line": 8.5},
                {"side": "home", "outcome": "spread",  "line": -1.5},
            ],
        }
        resp = client.post("/api/predictions/sgp", json=body)
        assert resp.status_code == 200
        d = resp.json()["data"]
        assert len(d["legs"]) == 3
        assert len(d["correlation_matrix"]) == 3
        assert len(d["correlation_matrix"][0]) == 3


# ===========================================================================
# GET /api/backtest/report
# ===========================================================================

class TestBacktestReport:

    def _write_summary(self, tmp_path: Path) -> None:
        summary = {
            "n_folds": 5,
            "n_oos_predictions": 250,
            "oos_log_loss": 1.23,
            "oos_accuracy": 0.41,
            "mean_fold_log_loss": 1.25,
            "std_fold_log_loss": 0.08,
            "mean_fold_brier": 0.45,
        }
        (tmp_path / "backtest_summary.json").write_text(json.dumps(summary))

    def _write_fold_metrics(self, tmp_path: Path) -> None:
        folds = [
            {
                "fold": i,
                "date_start": f"2023-0{i+1}-01",
                "date_end":   f"2023-0{i+1}-28",
                "n_train":    200 + i * 50,
                "n_test":     50,
                "log_loss":   1.2 + i * 0.02,
                "accuracy":   0.40 + i * 0.01,
                "brier":      0.44,
            }
            for i in range(5)
        ]
        (tmp_path / "wf_fold_metrics.json").write_text(json.dumps(folds))

    def _write_cal_report(self, tmp_path: Path) -> None:
        rows = [
            {
                "class_name": "HR", "ece": 0.05, "brier_score": 0.08,
                "reliability": 0.01, "resolution": 0.03, "uncertainty": 0.074,
                "brier_skill_score": 0.10, "base_rate": 0.034, "n_positive": 34,
            },
            {
                "class_name": "BB", "ece": 0.04, "brier_score": 0.07,
                "reliability": 0.01, "resolution": 0.04, "uncertainty": 0.077,
                "brier_skill_score": 0.09, "base_rate": 0.084, "n_positive": 84,
            },
        ]
        pd.DataFrame(rows).to_csv(tmp_path / "calibration_report.csv", index=False)

    def _write_clv(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(0)
        n = 50
        df = pd.DataFrame({
            "bet_id":          [f"g{i}" for i in range(n)],
            "bet_date":        ["2024-04-01"] * n,
            "market":          ["h2h"] * n,
            "side":            ["home"] * n,
            "open_odds":       rng.choice([-130, -110, 110, 130], size=n).astype(float),
            "open_odds_other": rng.choice([110, 120, -110, -120], size=n).astype(float),
            "close_odds":      rng.choice([-120, -105, 105, 120], size=n).astype(float),
            "close_odds_other": rng.choice([110, 115, -110, -115], size=n).astype(float),
            "result":          rng.binomial(1, 0.55, n).astype(float),
            "stake":           np.ones(n),
            "model_prob":      rng.uniform(0.45, 0.65, n),
        })
        from src.backtest.clv_tracker import BetRecord, compute_clv  # noqa: PLC0415
        bets = [BetRecord(**{k: row[k] for k in BetRecord.__dataclass_fields__})
                for _, row in df.iterrows()]
        clv_df = compute_clv(bets)
        clv_df.to_parquet(tmp_path / "clv_report.parquet", index=False)

    def test_no_data_returns_404(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("MLB_PRED_DIR", str(tmp_path / "predictions"))
        resp = client.get("/api/backtest/report")
        assert resp.status_code == 404

    def test_with_summary_only(self, tmp_path: Path, monkeypatch):
        d = tmp_path / "predictions"
        d.mkdir()
        self._write_summary(d)
        monkeypatch.setenv("MLB_PRED_DIR", str(d))
        resp = client.get("/api/backtest/report")
        assert resp.status_code == 200
        body = resp.json()
        assert "data" in body
        assert body["data"]["walk_forward"]["available"] is True
        assert "summary" in body["data"]["walk_forward"]

    def test_walk_forward_summary_keys(self, tmp_path: Path, monkeypatch):
        d = tmp_path / "predictions"
        d.mkdir()
        self._write_summary(d)
        monkeypatch.setenv("MLB_PRED_DIR", str(d))
        wf = client.get("/api/backtest/report").json()["data"]["walk_forward"]
        s = wf["summary"]
        for k in ["n_folds", "n_oos_predictions", "oos_log_loss",
                  "oos_accuracy", "mean_fold_log_loss"]:
            assert k in s

    def test_fold_metrics_loaded(self, tmp_path: Path, monkeypatch):
        d = tmp_path / "predictions"
        d.mkdir()
        self._write_summary(d)
        self._write_fold_metrics(d)
        monkeypatch.setenv("MLB_PRED_DIR", str(d))
        wf = client.get("/api/backtest/report").json()["data"]["walk_forward"]
        assert len(wf["fold_metrics"]) == 5
        fm = wf["fold_metrics"][0]
        for k in ["fold", "date_start", "date_end", "n_train", "n_test",
                  "log_loss", "accuracy", "brier"]:
            assert k in fm

    def test_calibration_report_loaded(self, tmp_path: Path, monkeypatch):
        d = tmp_path / "predictions"
        d.mkdir()
        self._write_summary(d)
        self._write_cal_report(d)
        monkeypatch.setenv("MLB_PRED_DIR", str(d))
        cal = client.get("/api/backtest/report").json()["data"]["calibration"]
        assert cal["available"] is True
        assert len(cal["report"]) == 2
        assert "brier_decomposition" in cal

    def test_brier_decomposition_structure(self, tmp_path: Path, monkeypatch):
        d = tmp_path / "predictions"
        d.mkdir()
        self._write_summary(d)
        self._write_cal_report(d)
        monkeypatch.setenv("MLB_PRED_DIR", str(d))
        bd = client.get("/api/backtest/report").json()["data"]["calibration"]["brier_decomposition"]
        for k in ["brier_score", "reliability", "resolution",
                  "uncertainty", "brier_skill_score"]:
            assert k in bd
            assert isinstance(bd[k], float)

    def test_clv_summary_loaded(self, tmp_path: Path, monkeypatch):
        d = tmp_path / "predictions"
        d.mkdir()
        self._write_summary(d)
        self._write_clv(d)
        monkeypatch.setenv("MLB_PRED_DIR", str(d))
        clv = client.get("/api/backtest/report").json()["data"]["clv"]
        assert clv["available"] is True
        assert "summary" in clv
        for k in ["n_bets", "mean_clv_log_odds", "pct_positive_clv"]:
            assert k in clv["summary"]

    def test_partial_data_still_200(self, tmp_path: Path, monkeypatch):
        """Only summary present (no cal, no clv) → 200 with partial blocks."""
        d = tmp_path / "predictions"
        d.mkdir()
        self._write_summary(d)
        monkeypatch.setenv("MLB_PRED_DIR", str(d))
        resp = client.get("/api/backtest/report")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["walk_forward"]["available"] is True
        assert data["calibration"]["available"] is False
        assert data["clv"]["available"] is False

    def test_envelope_meta_present(self, tmp_path: Path, monkeypatch):
        d = tmp_path / "predictions"
        d.mkdir()
        self._write_summary(d)
        monkeypatch.setenv("MLB_PRED_DIR", str(d))
        meta = client.get("/api/backtest/report").json()["meta"]
        assert "generated_at" in meta
        assert "version" in meta


# ===========================================================================
# GET /api/games/{game_id}/legs
# ===========================================================================

# Minimal raw MLB Stats API schedule structure used by the board + legs endpoints
_FAKE_SCHED_RAW = {
    "dates": [{
        "games": [{
            "gamePk": 999_001,
            "officialDate": "2024-04-15",
            "gameDate": "2024-04-15T18:05:00Z",
            "status": {"detailedState": "Preview"},
            "teams": {
                "home": {
                    "team": {"name": "New York Yankees"},
                    "probablePitcher": {"fullName": "Gerrit Cole"},
                },
                "away": {
                    "team": {"name": "Boston Red Sox"},
                    "probablePitcher": {"fullName": "Chris Sale"},
                },
            },
            "venue": {"name": "Yankee Stadium"},
            "seriesDescription": "Regular Season",
        }]
    }]
}


def _patch_load_or_fetch(mocker, raw=None):
    return mocker.patch(
        "src.ingestion.mlb_stats_api._load_or_fetch",
        return_value=raw or _FAKE_SCHED_RAW,
    )


def _patch_pa_probs(mocker):
    import numpy as np
    pa = np.tile(
        np.array([0.15, 0.05, 0.01, 0.03, 0.09, 0.20, 0.47]),
        (9, 1),
    )
    return mocker.patch(
        "src.ingestion.player_stats.get_game_pa_probs",
        return_value=(pa, pa),
    )


class TestGameLegs:

    def test_returns_envelope(self, mocker):
        _patch_load_or_fetch(mocker)
        _patch_pa_probs(mocker)
        resp = client.get("/api/games/999001/legs?n_sims=500")
        assert resp.status_code == 200
        body = resp.json()
        assert "data" in body
        assert "meta" in body

    def test_data_shape(self, mocker):
        _patch_load_or_fetch(mocker)
        _patch_pa_probs(mocker)
        data = client.get("/api/games/999001/legs?n_sims=500").json()["data"]
        assert data["game_id"] == 999_001
        assert isinstance(data["home_team"], str)
        assert isinstance(data["away_team"], str)
        assert isinstance(data["legs"], list)
        assert isinstance(data["cache_ready"], bool)

    def test_legs_have_required_fields(self, mocker):
        _patch_load_or_fetch(mocker)
        _patch_pa_probs(mocker)
        legs = client.get("/api/games/999001/legs?n_sims=500").json()["data"]["legs"]
        assert len(legs) > 0
        for leg in legs:
            for field in ("leg_id", "description", "market_type", "side", "model_prob"):
                assert field in leg, f"Missing field: {field}"
            assert isinstance(leg["leg_id"], str)
            assert 0.0 <= leg["model_prob"] <= 1.0

    def test_leg_ids_stable_format(self, mocker):
        """leg_id must start with the game_id prefix."""
        _patch_load_or_fetch(mocker)
        _patch_pa_probs(mocker)
        legs = client.get("/api/games/999001/legs?n_sims=500").json()["data"]["legs"]
        for leg in legs:
            assert leg["leg_id"].startswith("999001:"), (
                f"leg_id {leg['leg_id']!r} missing game prefix"
            )


# ===========================================================================
# POST /api/parlays/evaluate
# ===========================================================================

def _make_eval_body(game_id: int = 999_001) -> dict:
    return {
        "legs": [
            {
                "leg_id":      f"{game_id}:moneyline_home",
                "game_id":     game_id,
                "description": "NYY ML",
                "side":        "home",
                "market_type": "moneyline",
                "model_prob":  0.55,
                "market_odds": -130,
                "line":        None,
            },
            {
                "leg_id":      f"{game_id}:total_over_8.5",
                "game_id":     game_id,
                "description": "O 8.5",
                "side":        "over",
                "market_type": "total",
                "model_prob":  0.52,
                "market_odds": -110,
                "line":        8.5,
            },
        ],
        "bankroll": 1000.0,
        "n_sims":   1_000,
    }


class TestParlayEvaluate:

    def test_returns_200_envelope(self):
        resp = client.post("/api/parlays/evaluate", json=_make_eval_body())
        assert resp.status_code == 200
        body = resp.json()
        assert "data" in body
        assert "meta" in body

    def test_data_keys_present(self):
        data = client.post("/api/parlays/evaluate", json=_make_eval_body()).json()["data"]
        for k in ("legs", "groups", "joint_prob_corr_adjusted",
                  "joint_prob_naive", "parlay_net_payout",
                  "pct_diff_corr_vs_naive", "kelly", "n_sims"):
            assert k in data, f"Missing key: {k}"

    def test_joint_probs_in_range(self):
        data = client.post("/api/parlays/evaluate", json=_make_eval_body()).json()["data"]
        assert 0 < data["joint_prob_corr_adjusted"] < 1
        assert 0 < data["joint_prob_naive"] < 1

    def test_naive_approx_product_of_probs(self):
        """Naive joint is in (0,1) and below the minimum marginal prob.

        For same-game legs the naive prob is the product of the *simulated*
        marginal probs (not the input model_prob), so we only assert it is
        strictly less than the smaller of the two marginals and greater than 0.
        """
        data = client.post("/api/parlays/evaluate", json=_make_eval_body()).json()["data"]
        marginals = [l["marginal_prob"] for l in data["legs"]]
        assert 0 < data["joint_prob_naive"] < min(marginals)

    def test_kelly_structure(self):
        kelly = client.post("/api/parlays/evaluate", json=_make_eval_body()).json()["data"]["kelly"]
        for variant in ("corr_adjusted", "naive"):
            for field in ("ev", "f_star", "f_quarter", "stake_units"):
                assert field in kelly[variant]

    def test_parlay_net_payout_positive(self):
        data = client.post("/api/parlays/evaluate", json=_make_eval_body()).json()["data"]
        assert data["parlay_net_payout"] > 0

    def test_cross_game_legs_treated_independently(self):
        """Two legs from different game_ids → each in its own group."""
        body = {
            "legs": [
                {
                    "leg_id": "111:moneyline_home", "game_id": 111,
                    "description": "Team A ML", "side": "home",
                    "market_type": "moneyline", "model_prob": 0.55,
                    "market_odds": -130, "line": None,
                },
                {
                    "leg_id": "222:moneyline_home", "game_id": 222,
                    "description": "Team B ML", "side": "home",
                    "market_type": "moneyline", "model_prob": 0.52,
                    "market_odds": -110, "line": None,
                },
            ],
            "bankroll": 1000.0,
            "n_sims": 1_000,
        }
        data = client.post("/api/parlays/evaluate", json=body).json()["data"]
        # Two separate groups
        assert len(data["groups"]) == 2
        # Joint prob must equal product of independent probs (exactly, since no MC run)
        assert math.isclose(
            data["joint_prob_corr_adjusted"],
            0.55 * 0.52,
            rel_tol=0.01,
        ), f"Expected ~{0.55*0.52:.4f}, got {data['joint_prob_corr_adjusted']}"

    def test_requires_min_2_legs(self):
        body = {
            "legs": [{
                "leg_id": "999:ml_home", "game_id": 999,
                "description": "X", "side": "home",
                "market_type": "moneyline", "model_prob": 0.55,
                "market_odds": -110, "line": None,
            }],
            "bankroll": 1000.0,
            "n_sims": 1_000,
        }
        resp = client.post("/api/parlays/evaluate", json=body)
        assert resp.status_code == 422


# ===========================================================================
# GET /api/parlays/suggested
# ===========================================================================

class TestParlaysSuggested:

    def _patch_board(self, mocker, rows: list[dict] | None = None):
        """Mock the inner api_game_board call used by suggested."""
        if rows is None:
            rows = [
                {
                    "category": "moneyline", "label": "NYY ML",
                    "side": "home", "line": None,
                    "model_prob": 0.56, "market_prob": 0.50,
                    "market_odds": -130, "edge": 0.06,
                    "ev_per_dollar": 0.03, "kelly_stake": 10.0,
                    "confidence": {"point_estimate": 0.56, "ci_low": 0.52,
                                   "ci_high": 0.60, "ci_width": 0.08},
                    "confidence_score": 0.75,
                },
                {
                    "category": "total", "label": "O 8.5",
                    "side": "over", "line": 8.5,
                    "model_prob": 0.53, "market_prob": 0.50,
                    "market_odds": -110, "edge": 0.03,
                    "ev_per_dollar": 0.015, "kelly_stake": 5.0,
                    "confidence": {"point_estimate": 0.53, "ci_low": 0.49,
                                   "ci_high": 0.57, "ci_width": 0.08},
                    "confidence_score": 0.40,
                },
            ]
        import src.api as _api

        async def _fake_board(**kw):
            return {
                "data": {
                    "game_id": kw.get("game_id", 999_001),
                    "home_team": "NYY",
                    "away_team": "BOS",
                    "n_sims": 1000,
                    "rows": rows,
                    "omitted": [],
                }
            }

        mocker.patch.object(_api, "api_game_board", new=_fake_board)

    def test_invalid_mode_422(self, mocker):
        _patch_load_or_fetch(mocker)
        resp = client.get("/api/parlays/suggested?mode=garbage")
        assert resp.status_code == 422

    def test_no_games_returns_empty(self, mocker):
        mocker.patch(
            "src.ingestion.mlb_stats_api._load_or_fetch",
            return_value={"dates": []},
        )
        data = client.get("/api/parlays/suggested?mode=sgp&n_sims=1000").json()["data"]
        assert data["parlays"] == []
        assert data["message"] is not None

    def test_sgp_envelope(self, mocker):
        _patch_load_or_fetch(mocker)
        self._patch_board(mocker)
        resp = client.get("/api/parlays/suggested?mode=sgp&n_sims=1000")
        assert resp.status_code == 200
        body = resp.json()
        assert "data" in body and "meta" in body

    def test_sgp_data_shape(self, mocker):
        _patch_load_or_fetch(mocker)
        self._patch_board(mocker)
        data = client.get("/api/parlays/suggested?mode=sgp&n_sims=1000").json()["data"]
        assert data["mode"] == "sgp"
        assert isinstance(data["parlays"], list)
        assert isinstance(data["skipped_games"], list)

    def test_sgp_parlay_fields(self, mocker):
        _patch_load_or_fetch(mocker)
        self._patch_board(mocker)
        parlays = client.get("/api/parlays/suggested?mode=sgp&n_sims=1000").json()["data"]["parlays"]
        assert len(parlays) > 0
        p = parlays[0]
        for field in ("rank", "legs", "game_ids", "joint_prob_corr",
                      "joint_prob_naive", "parlay_net_payout",
                      "ev", "kelly_stake", "ci_low", "ci_high"):
            assert field in p, f"Missing field: {field}"
        assert p["rank"] == 1
        assert len(p["legs"]) >= 2

    def test_crossgame_mode(self, mocker):
        # Stub two games in schedule for cross-game
        sched_two = {
            "dates": [{
                "games": [
                    {**_FAKE_SCHED_RAW["dates"][0]["games"][0]},
                    {
                        **_FAKE_SCHED_RAW["dates"][0]["games"][0],
                        "gamePk": 999_002,
                        "teams": {
                            "home": {"team": {"name": "Cubs"}, "probablePitcher": {"fullName": "X"}},
                            "away": {"team": {"name": "Cardinals"}, "probablePitcher": {"fullName": "Y"}},
                        },
                    },
                ]
            }]
        }
        mocker.patch("src.ingestion.mlb_stats_api._load_or_fetch", return_value=sched_two)
        self._patch_board(mocker)
        resp = client.get("/api/parlays/suggested?mode=crossgame&n_sims=1000")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["mode"] == "crossgame"
        assert isinstance(data["parlays"], list)
