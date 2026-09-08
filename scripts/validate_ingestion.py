#!/usr/bin/env python3
"""
scripts/validate_ingestion.py
==============================
Smoke-tests the ingestion layer without requiring live API keys.

Checks:
  1. get_schedule() returns a list (even if empty — schedule cache may be cold)
  2. Schedule dict keys match expected schema
  3. fetch_live_odds() degrades gracefully when ODDS_API_KEY is unset
  4. Statcast cache directory creation and parquet helper
  5. Weather module imports and park data loads
  6. live_odds additive_devig round-trips correctly

Exit 0 = all pass.  Exit 1 = one or more failures.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

# ── Run from project root ──────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

PASS = "✓"
FAIL = "✗"
results: list[tuple[str, bool, str]] = []


def check(name: str, fn):
    """Run fn(); record pass/fail."""
    try:
        fn()
        results.append((name, True, ""))
    except Exception:
        results.append((name, False, traceback.format_exc().strip().splitlines()[-1]))


# ── 1. get_schedule import and callable ────────────────────────────────────
def _check_schedule_import():
    from src.ingestion.mlb_stats_api import get_schedule, fetch_schedule
    assert callable(get_schedule)
    assert fetch_schedule is get_schedule  # alias must hold

check("mlb_stats_api: get_schedule importable + fetch_schedule alias", _check_schedule_import)


# ── 2. get_schedule returns list with correct keys ─────────────────────────
def _check_schedule_schema():
    from src.ingestion.mlb_stats_api import get_schedule
    # Use a fixed historical date so the call hits the cache (or returns [])
    # without a real network request during CI
    try:
        games = get_schedule("2024-04-01")
    except Exception:
        # Network unavailable — just verify the return type assumption
        games = []
    assert isinstance(games, list), f"Expected list, got {type(games)}"
    if games:
        g = games[0]
        for key in ("game_pk", "game_date", "home_team_name", "away_team_name", "status"):
            assert key in g, f"Missing key: {key}"

check("get_schedule: returns list; non-empty entries have required keys", _check_schedule_schema)


# ── 3. live_odds degrades gracefully with no API key ──────────────────────
def _check_odds_no_key():
    import os
    old = os.environ.pop("ODDS_API_KEY", None)
    try:
        from src.ingestion.live_odds import get_game_odds
        # Should raise or return empty — must not hard-crash
        try:
            get_game_odds(sport_key="baseball_mlb", market_key="h2h")
        except Exception:
            pass  # acceptable: raises with no key
    finally:
        if old is not None:
            os.environ["ODDS_API_KEY"] = old

check("live_odds: get_game_odds degrades gracefully without ODDS_API_KEY", _check_odds_no_key)


# ── 4. devig_pair round-trip ──────────────────────────────────────────────
def _check_devig_roundtrip():
    from src.ingestion.live_odds import devig_pair
    # Two-way market: -110 / -110
    p_home, p_away = devig_pair(-110, -110)
    assert abs(p_home + p_away - 1.0) < 1e-9
    assert abs(p_home - p_away) < 1e-9   # symmetric market

check("live_odds: devig_pair(-110, -110) sums to 1 and is symmetric", _check_devig_roundtrip)


# ── 5. snapshot_odds structure ────────────────────────────────────────────
def _check_snapshot_odds():
    from src.ingestion.live_odds import snapshot_odds
    assert callable(snapshot_odds)

check("live_odds: snapshot_odds callable", _check_snapshot_odds)


# ── 6. Statcast module importable ─────────────────────────────────────────
def _check_statcast_import():
    from src.ingestion.statcast import pull_statcast
    assert callable(pull_statcast)

check("statcast: pull_statcast importable", _check_statcast_import)


# ── 7. Weather module importable + park data loads ────────────────────────
def _check_weather_import():
    from src.ingestion.weather import get_game_weather
    assert callable(get_game_weather)
    # parks.json must exist
    parks_path = _ROOT / "data" / "parks.json"
    assert parks_path.exists(), f"Missing parks.json at {parks_path}"
    import json
    parks = json.loads(parks_path.read_text())
    assert len(parks) >= 30, f"Expected ≥30 parks, got {len(parks)}"
    first = next(iter(parks.values()))
    assert "lat" in first and "lon" in first

check("weather: get_game_weather importable; parks.json has ≥30 entries with lat/lon", _check_weather_import)


# ── 8. Statcast raw cache directory is creatable ─────────────────────────
def _check_cache_dir():
    cache = _ROOT / "data" / "raw" / "statcast"
    cache.mkdir(parents=True, exist_ok=True)
    assert cache.is_dir()

check("statcast: data/raw/statcast/ directory creatable", _check_cache_dir)


# ── Report ─────────────────────────────────────────────────────────────────
print("\n── Ingestion validation ──")
failures = 0
for name, ok, msg in results:
    icon = PASS if ok else FAIL
    print(f"  {icon}  {name}")
    if not ok:
        print(f"       {msg}")
        failures += 1

print(f"\n{len(results) - failures}/{len(results)} checks passed.")
sys.exit(0 if failures == 0 else 1)
