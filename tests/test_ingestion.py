"""
tests/test_ingestion.py
=======================
Unit tests for all four ingestion modules.

All external I/O (HTTP requests, pybaseball, file system) is mocked so
these tests run fully offline without API keys.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# ===========================================================================
# ── Helpers ────────────────────────────────────────────────────────────────
# ===========================================================================

def _parks_json(tmp_path: Path) -> Path:
    """Write a minimal parks.json for tests."""
    parks = {
        "33": {
            "name": "Oracle Park",
            "team": "San Francisco Giants",
            "city": "San Francisco, CA",
            "lat": 37.7786,
            "lon": -122.3893,
            "elevation_m": 5,
            "orientation_degrees": 113,  # SF faces toward SE / right-center
        }
    }
    p = tmp_path / "parks.json"
    p.write_text(json.dumps(parks))
    return p


# ===========================================================================
# ── live_odds — odds math helpers (pure, no I/O) ──────────────────────────
# ===========================================================================

class TestOddsMath:
    """All helpers are pure functions — no mocking required."""

    def test_american_to_decimal_positive(self):
        from src.ingestion.live_odds import american_to_decimal
        assert american_to_decimal(150) == pytest.approx(2.5, rel=1e-4)

    def test_american_to_decimal_negative(self):
        from src.ingestion.live_odds import american_to_decimal
        assert american_to_decimal(-110) == pytest.approx(1.909091, rel=1e-4)

    def test_decimal_to_american_pos(self):
        from src.ingestion.live_odds import decimal_to_american
        assert decimal_to_american(2.5) == 150

    def test_decimal_to_american_neg(self):
        from src.ingestion.live_odds import decimal_to_american
        # -110 line → decimal 1.909…
        assert decimal_to_american(1.909091) == pytest.approx(-110, abs=1)

    def test_american_to_implied_prob_positive(self):
        from src.ingestion.live_odds import american_to_implied_prob
        # +100 → 50%
        assert american_to_implied_prob(100) == pytest.approx(0.5, rel=1e-4)

    def test_american_to_implied_prob_negative(self):
        from src.ingestion.live_odds import american_to_implied_prob
        # -200 → 66.67%
        assert american_to_implied_prob(-200) == pytest.approx(0.6667, rel=1e-3)

    def test_devig_pair_sums_to_one(self):
        from src.ingestion.live_odds import devig_pair
        p1, p2 = devig_pair(-110, -110)
        assert p1 + p2 == pytest.approx(1.0, rel=1e-6)

    def test_devig_pair_even_game(self):
        from src.ingestion.live_odds import devig_pair
        # Both sides -110 → each fair prob ~0.5
        p1, p2 = devig_pair(-110, -110)
        assert p1 == pytest.approx(0.5, rel=1e-4)

    def test_devig_pair_favourite(self):
        from src.ingestion.live_odds import devig_pair
        # -200 favourite vs +170 dog
        fav, dog = devig_pair(-200, 170)
        assert fav > dog
        assert fav + dog == pytest.approx(1.0, rel=1e-6)

    def test_roundtrip_american_decimal(self):
        from src.ingestion.live_odds import american_to_decimal, decimal_to_american
        for american in (-150, -110, 100, 120, 200):
            assert decimal_to_american(american_to_decimal(american)) == pytest.approx(american, abs=1)


# ===========================================================================
# ── live_odds — get_game_odds (mocked HTTP) ────────────────────────────────
# ===========================================================================

_FAKE_EVENT = {
    "id": "abc123",
    "sport_key": "baseball_mlb",
    "commence_time": "2024-07-04T18:05:00Z",
    "home_team": "San Francisco Giants",
    "away_team": "Los Angeles Dodgers",
    "bookmakers": [
        {
            "key": "draftkings",
            "last_update": "2024-07-04T17:00:00Z",
            "markets": [
                {
                    "key": "h2h",
                    "outcomes": [
                        {"name": "San Francisco Giants", "price": 130},
                        {"name": "Los Angeles Dodgers", "price": -150},
                    ],
                },
                {
                    "key": "totals",
                    "outcomes": [
                        {"name": "Over", "price": -110, "point": 8.5},
                        {"name": "Under", "price": -110, "point": 8.5},
                    ],
                },
            ],
        }
    ],
}


def test_get_game_odds_structure(tmp_path, monkeypatch):
    """get_game_odds parses the API response and returns GameOdds list."""
    import src.ingestion.live_odds as lo

    monkeypatch.setenv("ODDS_API_KEY", "test-key")
    monkeypatch.setattr(lo, "_CACHE_DIR", tmp_path)

    with patch("src.ingestion.live_odds._fetch", return_value=[_FAKE_EVENT]) as mock_fetch:
        result = lo.get_game_odds(regions="us", markets=["h2h", "totals"])

    mock_fetch.assert_called_once()
    assert len(result) == 2  # one per market

    h2h = next(r for r in result if r["market"] == "h2h")
    assert h2h["home_team"] == "San Francisco Giants"
    assert h2h["away_team"] == "Los Angeles Dodgers"
    assert h2h["home_implied_prob"] is not None
    assert h2h["away_implied_prob"] is not None
    assert h2h["home_implied_prob"] + h2h["away_implied_prob"] == pytest.approx(1.0, rel=1e-5)


def test_get_game_odds_cache_hit(tmp_path, monkeypatch):
    """Second call returns cached Parquet without hitting the API."""
    import src.ingestion.live_odds as lo

    monkeypatch.setenv("ODDS_API_KEY", "test-key")
    monkeypatch.setattr(lo, "_CACHE_DIR", tmp_path)

    with patch("src.ingestion.live_odds._fetch", return_value=[_FAKE_EVENT]):
        lo.get_game_odds(regions="us", markets=["h2h"])

    with patch("src.ingestion.live_odds._fetch") as mock2:
        lo.get_game_odds(regions="us", markets=["h2h"])
        mock2.assert_not_called()


def test_get_game_odds_no_api_key(tmp_path, monkeypatch):
    """Missing ODDS_API_KEY raises EnvironmentError (not a generic crash)."""
    import src.ingestion.live_odds as lo

    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    monkeypatch.setattr(lo, "_CACHE_DIR", tmp_path)

    with pytest.raises(EnvironmentError, match="ODDS_API_KEY"):
        lo.get_game_odds(force_refresh=True)


def test_get_player_props_invalid_market():
    from src.ingestion.live_odds import get_player_props
    with pytest.raises(ValueError, match="No valid prop markets"):
        get_player_props("game123", markets=["invalid_market"])


# ===========================================================================
# ── weather — physics helpers (pure math) ─────────────────────────────────
# ===========================================================================

class TestWeatherPhysics:
    def test_magnus_vapour_pressure_at_20c(self):
        """Magnus formula: ~2.337 kPa at 20°C."""
        from src.ingestion.weather import _magnus_vapour_pressure_kpa
        result = _magnus_vapour_pressure_kpa(20.0)
        assert result == pytest.approx(2.337, rel=0.01)

    def test_magnus_increases_with_temperature(self):
        from src.ingestion.weather import _magnus_vapour_pressure_kpa
        assert _magnus_vapour_pressure_kpa(30.0) > _magnus_vapour_pressure_kpa(20.0)

    def test_air_density_dry_sea_level(self):
        """Standard atmosphere: ~1.225 kg/m³ at 15°C, 0% RH, 101.325 kPa."""
        from src.ingestion.weather import _air_density
        _, rho = _air_density(temp_c=15.0, humidity_pct=0.0, pressure_kpa=101.325)
        assert rho == pytest.approx(1.225, rel=0.005)

    def test_air_density_humid_is_lower(self):
        """Humid air is less dense than dry air (water vapour displaces heavier N₂/O₂)."""
        from src.ingestion.weather import _air_density
        _, rho_dry = _air_density(15.0, 0.0, 101.325)
        _, rho_humid = _air_density(15.0, 100.0, 101.325)
        assert rho_humid < rho_dry

    def test_air_density_high_altitude(self):
        """Coors Field (~1580 m): pressure ~84 kPa → density ~1.0 kg/m³."""
        from src.ingestion.weather import _air_density
        _, rho = _air_density(temp_c=20.0, humidity_pct=30.0, pressure_kpa=84.0)
        assert rho == pytest.approx(1.0, rel=0.05)

    def test_wind_pure_tailwind(self):
        """Wind blowing directly toward CF = full tailwind, zero crosswind."""
        from src.ingestion.weather import _wind_decomposition
        # Park CF at bearing 0° (north); wind FROM south (180°) blows toward north
        tail, cross = _wind_decomposition(
            wind_speed_ms=10.0,
            wind_dir_degrees=180.0,   # FROM south
            park_orientation_degrees=0.0,  # CF at north
        )
        assert tail == pytest.approx(10.0, abs=1e-6)
        assert cross == pytest.approx(0.0, abs=1e-6)

    def test_wind_pure_headwind(self):
        """Wind blowing from CF directly in = full headwind (negative tailwind)."""
        from src.ingestion.weather import _wind_decomposition
        # Park CF at bearing 0° (north); wind FROM north (0°) blows toward south
        tail, cross = _wind_decomposition(
            wind_speed_ms=10.0,
            wind_dir_degrees=0.0,    # FROM north
            park_orientation_degrees=0.0,
        )
        assert tail == pytest.approx(-10.0, abs=1e-6)
        assert cross == pytest.approx(0.0, abs=1e-6)

    def test_wind_pure_crosswind(self):
        """Wind blowing 90° to CF axis = zero tailwind, full crosswind."""
        from src.ingestion.weather import _wind_decomposition
        # Park CF at bearing 0° (north); wind FROM west (270°) → going east
        tail, cross = _wind_decomposition(
            wind_speed_ms=10.0,
            wind_dir_degrees=270.0,  # FROM west
            park_orientation_degrees=0.0,
        )
        assert tail == pytest.approx(0.0, abs=1e-6)
        assert abs(cross) == pytest.approx(10.0, abs=1e-6)

    def test_wind_decomposition_magnitude_preserved(self):
        """Pythagorean: tailwind² + crosswind² = wind_speed²."""
        from src.ingestion.weather import _wind_decomposition
        speed = 8.5
        tail, cross = _wind_decomposition(speed, wind_dir_degrees=45.0, park_orientation_degrees=90.0)
        assert math.sqrt(tail**2 + cross**2) == pytest.approx(speed, rel=1e-6)


# ===========================================================================
# ── weather — get_game_weather (mocked HTTP + cache) ──────────────────────
# ===========================================================================

# OpenWeather-format fake response (obs dict returned by _fetch_obs)
_FAKE_OW_OBS = {
    "temp_c":        18.0,
    "humidity_pct":  72.0,
    "wind_speed_ms": 4.0,    # m/s directly (OpenWeather metric)
    "wind_dir_deg":  270.0,  # FROM west
    "pressure_hpa":  1013.0,
    "conditions":    "Partly cloudy",
}


def test_get_game_weather_returns_all_fields(tmp_path):
    """End-to-end: mocked OpenWeather response → all TypedDict keys present + physics correct."""
    import src.ingestion.weather as w

    parks_path = _parks_json(tmp_path)
    cache_dir = tmp_path / "weather"

    with patch.object(w, "_CACHE_DIR", cache_dir), \
         patch.object(w, "_fetch_obs", return_value=_FAKE_OW_OBS), \
         patch.dict(os.environ, {"OPENWEATHER_API_KEY": "test-key"}):
        result = w.get_game_weather(
            park_id="33",
            game_datetime_utc=datetime(2024, 7, 4, 19, 5, tzinfo=timezone.utc),
            parks_json_path=parks_path,
        )

    expected_keys = {
        "park_id", "game_datetime_utc",
        "temp_c", "humidity_pct", "wind_speed_ms", "wind_dir_degrees",
        "pressure_kpa", "conditions",
        "vapour_pressure_kpa", "air_density_kg_m3",
        "park_orientation_degrees", "tailwind_ms", "crosswind_ms",
    }
    assert expected_keys.issubset(result.keys())
    assert result["park_id"] == "33"
    assert result["temp_c"] == 18.0
    assert result["wind_speed_ms"] == pytest.approx(4.0, rel=0.01)
    # Physics sanity: air density should be near sea-level standard
    assert 1.1 < result["air_density_kg_m3"] < 1.3
    # Magnitude preserved
    assert math.sqrt(result["tailwind_ms"]**2 + result["crosswind_ms"]**2) == \
        pytest.approx(result["wind_speed_ms"], rel=1e-5)


def test_get_game_weather_cache_hit(tmp_path):
    """Second call with same key skips HTTP fetch."""
    import src.ingestion.weather as w

    parks_path = _parks_json(tmp_path)
    cache_dir = tmp_path / "weather"

    with patch.object(w, "_CACHE_DIR", cache_dir), \
         patch.object(w, "_fetch_obs", return_value=_FAKE_OW_OBS) as mock_fetch, \
         patch.dict(os.environ, {"OPENWEATHER_API_KEY": "test-key"}):
        dt = datetime(2024, 7, 4, 19, 5, tzinfo=timezone.utc)
        w.get_game_weather("33", dt, parks_path)
        w.get_game_weather("33", dt, parks_path)
        assert mock_fetch.call_count == 1


def test_get_game_weather_missing_park(tmp_path):
    """Unknown park_id raises KeyError."""
    import src.ingestion.weather as w
    parks_path = _parks_json(tmp_path)
    with pytest.raises(KeyError, match="999"):
        w.get_game_weather("999", datetime(2024, 7, 4, 19, 5), parks_path)


def test_get_game_weather_missing_api_key(tmp_path, monkeypatch):
    """Missing OPENWEATHER_API_KEY raises EnvironmentError."""
    import src.ingestion.weather as w
    monkeypatch.delenv("OPENWEATHER_API_KEY", raising=False)
    parks_path = _parks_json(tmp_path)
    cache_dir = tmp_path / "weather"
    with patch.object(w, "_CACHE_DIR", cache_dir), \
         patch.object(w, "_fetch_obs", side_effect=EnvironmentError("OPENWEATHER_API_KEY")):
        with pytest.raises(EnvironmentError, match="OPENWEATHER_API_KEY"):
            w.get_game_weather("33", datetime(2024, 7, 4, 19, 5), parks_path, force_refresh=True)


# ===========================================================================
# ── mlb_stats_api — mocked HTTP ───────────────────────────────────────────
# ===========================================================================

_FAKE_SCHEDULE = {
    "dates": [
        {
            "date": "2024-07-04",
            "games": [
                {
                    "gamePk": 745528,
                    "officialDate": "2024-07-04",
                    "gameDate": "2024-07-04T22:10:00Z",
                    "gameNumber": 1,
                    "seriesDescription": "Regular Season",
                    "status": {"detailedState": "Scheduled"},
                    "teams": {
                        "home": {
                            "team": {"id": 137, "name": "San Francisco Giants"},
                            "probablePitcher": {
                                "id": 657277,
                                "fullName": "Logan Webb",
                                "pitchHand": {"code": "R"},
                            },
                        },
                        "away": {
                            "team": {"id": 119, "name": "Los Angeles Dodgers"},
                            "probablePitcher": {
                                "id": 708227,
                                "fullName": "Tyler Glasnow",
                                "pitchHand": {"code": "R"},
                            },
                        },
                    },
                    "venue": {"id": 2395, "name": "Oracle Park"},
                }
            ],
        }
    ]
}


def test_get_schedule_structure(tmp_path):
    """get_schedule returns correctly shaped list of game dicts."""
    import src.ingestion.mlb_stats_api as api
    with patch.object(api, "_CACHE_DIR", tmp_path), \
         patch("src.ingestion.mlb_stats_api._fetch", return_value=_FAKE_SCHEDULE):
        games = api.get_schedule("2024-07-04")

    assert len(games) == 1
    g = games[0]
    assert g["game_pk"] == 745528
    assert g["home_team_name"] == "San Francisco Giants"
    assert g["away_team_name"] == "Los Angeles Dodgers"
    assert g["venue_id"] == 2395
    assert g["venue_name"] == "Oracle Park"


def test_get_schedule_cache(tmp_path):
    """Second call returns cached result, no extra fetch."""
    import src.ingestion.mlb_stats_api as api
    with patch.object(api, "_CACHE_DIR", tmp_path), \
         patch("src.ingestion.mlb_stats_api._fetch", return_value=_FAKE_SCHEDULE) as mf:
        api.get_schedule("2024-07-04")
        api.get_schedule("2024-07-04")
        assert mf.call_count == 1


def test_get_probable_pitchers(tmp_path):
    """get_probable_pitchers returns {game_pk: {home, away}} structure."""
    import src.ingestion.mlb_stats_api as api
    with patch.object(api, "_CACHE_DIR", tmp_path), \
         patch("src.ingestion.mlb_stats_api._fetch", return_value=_FAKE_SCHEDULE):
        pitchers = api.get_probable_pitchers("2024-07-04")

    assert 745528 in pitchers
    home = pitchers[745528]["home"]
    away = pitchers[745528]["away"]
    assert home is not None and home["full_name"] == "Logan Webb"
    assert away is not None and away["full_name"] == "Tyler Glasnow"


_FAKE_ROSTER = {
    "roster": [
        {
            "person": {
                "id": 657277,
                "fullName": "Logan Webb",
                "pitchHand": {"code": "R"},
                "batSide": {"code": "R"},
            },
            "position": {"abbreviation": "SP"},
            "jerseyNumber": "62",
            "status": {"description": "Active"},
        }
    ]
}


def test_get_roster_structure(tmp_path):
    """get_roster returns correctly shaped player dicts."""
    import src.ingestion.mlb_stats_api as api
    with patch.object(api, "_CACHE_DIR", tmp_path), \
         patch("src.ingestion.mlb_stats_api._fetch", return_value=_FAKE_ROSTER):
        roster = api.get_roster(team_id=137, roster_date="2024-07-04")

    assert len(roster) == 1
    p = roster[0]
    assert p["player_id"] == 657277
    assert p["full_name"] == "Logan Webb"
    assert p["position"] == "SP"
    assert p["throws"] == "R"


def test_schedule_to_parquet_writes_file(tmp_path):
    """schedule_to_parquet writes a valid Parquet file and returns its path."""
    import src.ingestion.mlb_stats_api as api
    with patch.object(api, "_CACHE_DIR", tmp_path), \
         patch("src.ingestion.mlb_stats_api._fetch", return_value=_FAKE_SCHEDULE):
        out = api.schedule_to_parquet("2024-07-04")

    assert out.exists()
    df = pd.read_parquet(out)
    assert len(df) == 1
    assert "game_pk" in df.columns


# ===========================================================================
# ── statcast — date chunking + dtype normalisation (no network) ────────────
# ===========================================================================

class TestStatcastDateChunks:
    def test_single_day(self):
        from src.ingestion.statcast import _date_chunks
        pairs = list(_date_chunks(date(2024, 4, 1), date(2024, 4, 1), 7))
        assert pairs == [(date(2024, 4, 1), date(2024, 4, 1))]

    def test_exact_chunk_boundary(self):
        from src.ingestion.statcast import _date_chunks
        pairs = list(_date_chunks(date(2024, 4, 1), date(2024, 4, 7), 7))
        assert len(pairs) == 1
        assert pairs[0] == (date(2024, 4, 1), date(2024, 4, 7))

    def test_two_chunks(self):
        from src.ingestion.statcast import _date_chunks
        pairs = list(_date_chunks(date(2024, 4, 1), date(2024, 4, 14), 7))
        assert len(pairs) == 2
        assert pairs[0] == (date(2024, 4, 1), date(2024, 4, 7))
        assert pairs[1] == (date(2024, 4, 8), date(2024, 4, 14))

    def test_partial_final_chunk(self):
        from src.ingestion.statcast import _date_chunks
        pairs = list(_date_chunks(date(2024, 4, 1), date(2024, 4, 10), 7))
        assert len(pairs) == 2
        assert pairs[-1] == (date(2024, 4, 8), date(2024, 4, 10))

    def test_full_month_chunk_count(self):
        from src.ingestion.statcast import _date_chunks
        pairs = list(_date_chunks(date(2024, 4, 1), date(2024, 4, 30), 7))
        # 30 days / 7 = 4 full + 1 partial = 5 chunks (but days are 1-7, 8-14, 15-21, 22-28, 29-30)
        assert len(pairs) == 5


class TestStatcastDtypeNormalisation:
    def test_float_cols_cast(self):
        from src.ingestion.statcast import _normalise_dtypes
        df = pd.DataFrame({"release_speed": ["95.1", "88.3", None], "game_pk": [12345, 12345, 12346]})
        result = _normalise_dtypes(df)
        assert result["release_speed"].dtype == "float32"

    def test_int_cols_cast(self):
        from src.ingestion.statcast import _normalise_dtypes
        df = pd.DataFrame({"inning": ["1", "2", "9"], "game_pk": [1, 2, 3]})
        result = _normalise_dtypes(df)
        assert str(result["inning"].dtype) == "Int16"

    def test_game_pk_cast(self):
        from src.ingestion.statcast import _normalise_dtypes
        df = pd.DataFrame({"game_pk": ["745528", "745529"]})
        result = _normalise_dtypes(df)
        assert str(result["game_pk"].dtype) == "Int32"

    def test_missing_cols_skipped(self):
        """Columns not present in the frame should not cause errors."""
        from src.ingestion.statcast import _normalise_dtypes
        df = pd.DataFrame({"unrelated_col": [1, 2, 3]})
        result = _normalise_dtypes(df)
        assert list(result.columns) == ["unrelated_col"]


def test_pull_statcast_date_validation():
    """start_date after end_date raises ValueError."""
    from src.ingestion.statcast import pull_statcast
    with pytest.raises(ValueError, match="after end_date"):
        pull_statcast("2024-04-10", "2024-04-01")


def test_pull_statcast_uses_cache(tmp_path):
    """If a chunk Parquet already exists, pybaseball is never called."""
    from src.ingestion.statcast import pull_statcast, _normalise_dtypes

    # Pre-populate a chunk file
    chunk_path = tmp_path / "statcast_2024-04-01_2024-04-07.parquet"
    df_fake = pd.DataFrame({
        "game_pk": pd.array([745528], dtype="Int32"),
        "at_bat_number": pd.array([1], dtype="Int16"),
        "pitch_number": pd.array([1], dtype="Int16"),
        "release_speed": pd.array([95.0], dtype="float32"),
    })
    df_fake.to_parquet(chunk_path, index=False)

    with patch("src.ingestion.statcast._pull_chunk", wraps=lambda s, e, cd, fr: pd.read_parquet(chunk_path)) as mock_pc:
        result = pull_statcast("2024-04-01", "2024-04-07", cache_dir=tmp_path, inter_chunk_sleep=0)

    assert not result.empty
    assert result.iloc[0]["game_pk"] == 745528


def test_pull_statcast_deduplication(tmp_path):
    """Duplicate pitches across chunks are dropped."""
    from src.ingestion.statcast import pull_statcast

    row = {
        "game_pk": pd.array([745528], dtype="Int32"),
        "at_bat_number": pd.array([1], dtype="Int16"),
        "pitch_number": pd.array([1], dtype="Int16"),
    }
    chunk1 = tmp_path / "statcast_2024-04-01_2024-04-07.parquet"
    chunk2 = tmp_path / "statcast_2024-04-08_2024-04-14.parquet"
    df1 = pd.DataFrame(row)
    df2 = pd.DataFrame(row)  # exact duplicate
    df1.to_parquet(chunk1, index=False)
    df2.to_parquet(chunk2, index=False)

    result = pull_statcast("2024-04-01", "2024-04-14", cache_dir=tmp_path, inter_chunk_sleep=0)
    assert len(result) == 1  # deduped
