"""
ingestion.mlb_stats_api
=======================
Wrapper around the MLB Stats API (statsapi.mlb.com/api/v1).

Public API
----------
get_schedule(date)          -> list[dict]   one entry per game
get_boxscore(game_pk)       -> dict          linescore + batting/pitching splits
get_probable_pitchers(date) -> dict          {game_pk: {home: {...}, away: {...}}}
get_roster(team_id, date)   -> list[dict]    active 26-man roster

Caching
-------
All responses are cached to data/raw/mlb_stats/ as Parquet files keyed by
(endpoint, date/game_pk).  A cached file is reused unless force_refresh=True.

Environment
-----------
MLB_STATS_API_BASE  (default: https://statsapi.mlb.com/api/v1)
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_BASE_URL: str = os.getenv("MLB_STATS_API_BASE", "https://statsapi.mlb.com/api/v1").rstrip("/")
_CACHE_DIR: Path = Path(__file__).resolve().parents[3] / "data" / "raw" / "mlb_stats"
_TIMEOUT: int = 20  # seconds
_RETRY_DELAYS: tuple[int, ...] = (2, 5, 10)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cache_path(key: str) -> Path:
    """Return a .parquet path for a given cache key string."""
    safe = key.replace("/", "_").replace(":", "_")
    return _CACHE_DIR / f"{safe}.parquet"


def _fetch(endpoint: str, params: dict[str, Any] | None = None) -> Any:
    """GET {_BASE_URL}/{endpoint} with retry/backoff.  Returns parsed JSON."""
    url = f"{_BASE_URL}/{endpoint.lstrip('/')}"
    last_exc: Exception | None = None
    for attempt, delay in enumerate((*_RETRY_DELAYS, None), start=1):
        try:
            resp = requests.get(url, params=params, timeout=_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            last_exc = exc
            logger.warning("MLB Stats API attempt %d failed for %s: %s", attempt, url, exc)
            if delay is not None:
                time.sleep(delay)
    raise RuntimeError(f"MLB Stats API unreachable after {len(_RETRY_DELAYS)+1} attempts: {url}") from last_exc


def _load_or_fetch(cache_key: str, endpoint: str, params: dict | None, force_refresh: bool) -> Any:
    """Return cached JSON (as Python object) or fetch and persist."""
    path = _cache_path(cache_key)
    if path.exists() and not force_refresh:
        logger.debug("Cache hit: %s", path)
        # stored as single-row parquet with a 'json' text column
        df = pd.read_parquet(path)
        return json.loads(df.at[0, "json"])
    data = _fetch(endpoint, params)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"json": [json.dumps(data)]}).to_parquet(path, index=False)
    return data


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------

def get_schedule(
    game_date: date | str | None = None,
    sport_id: int = 1,
    force_refresh: bool = False,
) -> list[dict]:
    """
    Return list of games for *game_date* (defaults to today).

    Each dict contains:
        game_pk, game_date, game_time_utc, status,
        home_team_id, home_team_name, away_team_id, away_team_name,
        venue_id, venue_name, series_description, game_number
    """
    if game_date is None:
        game_date = date.today()
    date_str = game_date.isoformat() if isinstance(game_date, date) else game_date

    data = _load_or_fetch(
        cache_key=f"schedule_{date_str}",
        endpoint="schedule",
        params={
            "sportId": sport_id,
            "date": date_str,
            "hydrate": "team,venue,probablePitcher,linescore,seriesStatus",
        },
        force_refresh=force_refresh,
    )

    games: list[dict] = []
    for date_entry in data.get("dates", []):
        for g in date_entry.get("games", []):
            teams = g.get("teams", {})
            home = teams.get("home", {})
            away = teams.get("away", {})
            games.append(
                {
                    "game_pk": g["gamePk"],
                    "game_date": g.get("officialDate"),
                    "game_time_utc": g.get("gameDate"),
                    "status": g.get("status", {}).get("detailedState"),
                    "home_team_id": home.get("team", {}).get("id"),
                    "home_team_name": home.get("team", {}).get("name"),
                    "away_team_id": away.get("team", {}).get("id"),
                    "away_team_name": away.get("team", {}).get("name"),
                    "venue_id": g.get("venue", {}).get("id"),
                    "venue_name": g.get("venue", {}).get("name"),
                    "series_description": g.get("seriesDescription"),
                    "game_number": g.get("gameNumber", 1),
                }
            )
    return games


# ---------------------------------------------------------------------------
# Boxscore
# ---------------------------------------------------------------------------

def get_boxscore(game_pk: int, force_refresh: bool = False) -> dict:
    """
    Return structured boxscore for *game_pk*.

    Returned keys:
        game_pk, linescore, home_batting, away_batting,
        home_pitching, away_pitching
    where *_batting and *_pitching are lists of per-player stat dicts.
    """
    data = _load_or_fetch(
        cache_key=f"boxscore_{game_pk}",
        endpoint=f"game/{game_pk}/boxscore",
        params=None,
        force_refresh=force_refresh,
    )

    def _extract_players(side: dict, group: str) -> list[dict]:
        rows = []
        for pid, pdata in side.get("players", {}).items():
            stats = pdata.get("stats", {}).get(group, {}).get("summary", {})
            season = pdata.get("seasonStats", {}).get(group, {})
            rows.append(
                {
                    "player_id": pdata.get("person", {}).get("id"),
                    "player_name": pdata.get("person", {}).get("fullName"),
                    "position": pdata.get("position", {}).get("abbreviation"),
                    "batting_order": pdata.get("battingOrder"),
                    **{f"game_{k}": v for k, v in stats.items()},
                    **{f"season_{k}": v for k, v in season.items()},
                }
            )
        return rows

    teams = data.get("teams", {})
    return {
        "game_pk": game_pk,
        "linescore": data.get("info", []),
        "home_batting": _extract_players(teams.get("home", {}), "batting"),
        "away_batting": _extract_players(teams.get("away", {}), "batting"),
        "home_pitching": _extract_players(teams.get("home", {}), "pitching"),
        "away_pitching": _extract_players(teams.get("away", {}), "pitching"),
    }


# ---------------------------------------------------------------------------
# Probable pitchers
# ---------------------------------------------------------------------------

def get_probable_pitchers(
    game_date: date | str | None = None,
    force_refresh: bool = False,
) -> dict[int, dict]:
    """
    Return {game_pk: {home: pitcher_dict, away: pitcher_dict}} for all
    games on *game_date*.

    pitcher_dict keys: player_id, full_name, throws
    """
    games = get_schedule(game_date, force_refresh=force_refresh)
    # Reload raw schedule data to access probablePitcher hydration
    if game_date is None:
        game_date = date.today()
    date_str = game_date.isoformat() if isinstance(game_date, date) else game_date

    raw = _load_or_fetch(
        cache_key=f"schedule_{date_str}",
        endpoint="schedule",
        params={
            "sportId": 1,
            "date": date_str,
            "hydrate": "team,venue,probablePitcher,linescore,seriesStatus",
        },
        force_refresh=False,  # already fetched above
    )

    result: dict[int, dict] = {}
    for date_entry in raw.get("dates", []):
        for g in date_entry.get("games", []):
            pk = g["gamePk"]
            teams = g.get("teams", {})

            def _pitcher(side: dict) -> dict | None:
                pp = side.get("probablePitcher")
                if not pp:
                    return None
                return {
                    "player_id": pp.get("id"),
                    "full_name": pp.get("fullName"),
                    "throws": pp.get("pitchHand", {}).get("code"),
                }

            result[pk] = {
                "home": _pitcher(teams.get("home", {})),
                "away": _pitcher(teams.get("away", {})),
            }
    return result


# ---------------------------------------------------------------------------
# Roster
# ---------------------------------------------------------------------------

def get_roster(
    team_id: int,
    roster_date: date | str | None = None,
    roster_type: str = "active",
    force_refresh: bool = False,
) -> list[dict]:
    """
    Return the active (or specified) roster for *team_id* on *roster_date*.

    Each dict: player_id, full_name, position, jersey_number, status, throws, bats
    """
    if roster_date is None:
        roster_date = date.today()
    date_str = roster_date.isoformat() if isinstance(roster_date, date) else roster_date

    data = _load_or_fetch(
        cache_key=f"roster_{team_id}_{date_str}_{roster_type}",
        endpoint=f"teams/{team_id}/roster",
        params={"rosterType": roster_type, "date": date_str},
        force_refresh=force_refresh,
    )

    roster: list[dict] = []
    for entry in data.get("roster", []):
        person = entry.get("person", {})
        roster.append(
            {
                "player_id": person.get("id"),
                "full_name": person.get("fullName"),
                "position": entry.get("position", {}).get("abbreviation"),
                "jersey_number": entry.get("jerseyNumber"),
                "status": entry.get("status", {}).get("description"),
                "throws": person.get("pitchHand", {}).get("code"),
                "bats": person.get("batSide", {}).get("code"),
            }
        )
    return roster


# ---------------------------------------------------------------------------
# Convenience: save schedule as Parquet DataFrame
# ---------------------------------------------------------------------------

def schedule_to_parquet(game_date: date | str | None = None, force_refresh: bool = False) -> Path:
    """Persist today's schedule to data/raw/mlb_stats/schedule_df_{date}.parquet."""
    if game_date is None:
        game_date = date.today()
    date_str = game_date.isoformat() if isinstance(game_date, date) else game_date
    games = get_schedule(game_date, force_refresh=force_refresh)
    out_path = _CACHE_DIR / f"schedule_df_{date_str}.parquet"
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(games).to_parquet(out_path, index=False)
    logger.info("Saved %d games to %s", len(games), out_path)
    return out_path
