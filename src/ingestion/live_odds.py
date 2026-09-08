"""
ingestion.live_odds
===================
Pulls MLB game lines and player props from The-Odds-API and persists
snapshots for closing-line-value (CLV) tracking.

Supported markets
-----------------
Game markets    : h2h (moneyline), spreads (run line), totals
Player props    : batter_hits, batter_total_bases, batter_home_runs,
                  pitcher_strikeouts, pitcher_earned_runs

Public API
----------
get_game_odds(regions, markets, force_refresh)
    -> list[GameOdds]   one entry per game, keyed by game_id + bookmaker

get_player_props(game_id, markets, regions, force_refresh)
    -> list[PropLine]   one entry per player × market × bookmaker

snapshot_odds(snapshot_date, regions)
    -> Path             path of written Parquet snapshot

american_to_decimal(american)   -> float
decimal_to_american(decimal)    -> float
american_to_implied_prob(american) -> float
devig_pair(odds_a, odds_b)      -> tuple[float, float]   additive de-vig

Interface contract
------------------
All odds are stored in *American* format in the raw dicts; the helper
functions convert on demand.  The downstream optimizer (src/optimizer/)
consumes the implied_prob fields computed by devig_pair().

Environment
-----------
ODDS_API_KEY   (required)
"""

from __future__ import annotations

import logging
import os
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_API_BASE = "https://api.the-odds-api.com/v4"
_SPORT = "baseball_mlb"
_TIMEOUT = 15
_RETRY_DELAYS: tuple[float, ...] = (3.0, 8.0, 20.0)
_CACHE_DIR: Path = Path(__file__).resolve().parents[3] / "data" / "raw" / "odds"

# Markets available via the events/{id}/odds endpoint
_PROP_MARKET_MAP: dict[str, str] = {
    "batter_hits":            "batter_hits",
    "batter_total_bases":     "batter_total_bases",
    "batter_home_runs":       "batter_home_runs",
    "pitcher_strikeouts":     "pitcher_strikeouts",
    "pitcher_earned_runs":    "pitcher_outs",  # API uses pitcher_outs for ER props
}
_GAME_MARKETS = ("h2h", "spreads", "totals")


# ---------------------------------------------------------------------------
# TypedDicts
# ---------------------------------------------------------------------------

class OutcomeOdds(TypedDict):
    name: str           # team name or "Over"/"Under" or player name
    price: int          # American odds
    point: float | None # spread / total point value; None for h2h/props


class BookmakerLine(TypedDict):
    bookmaker: str
    last_update: str
    outcomes: list[OutcomeOdds]


class GameOdds(TypedDict):
    game_id: str
    sport: str
    commence_time: str
    home_team: str
    away_team: str
    market: str
    lines: list[BookmakerLine]
    # Derived convenience fields (de-vigged)
    home_implied_prob: float | None
    away_implied_prob: float | None


class PropLine(TypedDict):
    game_id: str
    market: str
    player_name: str
    bookmaker: str
    last_update: str
    point: float | None   # line value (e.g. 0.5 hits)
    over_odds: int | None
    under_odds: int | None
    over_implied_prob: float | None
    under_implied_prob: float | None


# ---------------------------------------------------------------------------
# Odds math helpers (public — used by downstream optimizer)
# ---------------------------------------------------------------------------

def american_to_decimal(american: int | float) -> float:
    """Convert American odds to decimal (European) format."""
    a = float(american)
    if a >= 100:
        return round(a / 100.0 + 1.0, 6)
    return round(100.0 / abs(a) + 1.0, 6)


def decimal_to_american(decimal: float) -> int:
    """Convert decimal odds to American odds (rounded to nearest integer)."""
    if decimal >= 2.0:
        return round((decimal - 1.0) * 100)
    return round(-100.0 / (decimal - 1.0))


def american_to_implied_prob(american: int | float) -> float:
    """Raw (vigged) implied probability from American odds."""
    a = float(american)
    if a >= 100:
        return round(100.0 / (a + 100.0), 6)
    return round(abs(a) / (abs(a) + 100.0), 6)


def devig_pair(odds_a: int | float, odds_b: int | float) -> tuple[float, float]:
    """
    Additive de-vig for a two-outcome market.

    p_fair = p_raw / (p_raw_a + p_raw_b)

    Returns (fair_prob_a, fair_prob_b) — sum to 1.0.
    """
    pa = american_to_implied_prob(odds_a)
    pb = american_to_implied_prob(odds_b)
    total = pa + pb
    return round(pa / total, 6), round(pb / total, 6)


# ---------------------------------------------------------------------------
# Internal HTTP helper
# ---------------------------------------------------------------------------

def _fetch(endpoint: str, params: dict[str, Any], max_retries: int | None = None) -> Any:
    """
    GET _API_BASE/endpoint with API key and retry/backoff.

    Parameters
    ----------
    max_retries : if 0, try exactly once with NO urllib3-level retries and a
                  short timeout (fail-fast for live API endpoints).
                  If None, use the full _RETRY_DELAYS schedule.
    """
    api_key = os.getenv("ODDS_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "ODDS_API_KEY is not set. Add it to your .env file."
        )
    params = {**params, "apiKey": api_key}
    url = f"{_API_BASE}/{endpoint.lstrip('/')}"

    # Fail-fast mode: mount a Session with Retry(total=0) so urllib3 does not
    # retry on connection errors (SSL handshake failures, etc.), and use a
    # short connect timeout so the call fails in <3 s instead of 30 s.
    if max_retries == 0:
        from requests.adapters import HTTPAdapter  # noqa: PLC0415
        from urllib3.util.retry import Retry        # noqa: PLC0415
        sess = requests.Session()
        adapter = HTTPAdapter(max_retries=Retry(total=0, raise_on_status=False))
        sess.mount("https://", adapter)
        sess.mount("http://",  adapter)
        try:
            resp = sess.get(url, params=params, timeout=(3, 8))  # (connect, read)
            remaining = resp.headers.get("x-requests-remaining")
            if remaining is not None:
                logger.debug("Odds API quota remaining: %s", remaining)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            logger.warning("Odds API (try_once) failed: %s", exc)
            raise RuntimeError(f"Odds API unreachable: {url}") from exc
        finally:
            sess.close()

    # Normal mode: retry with backoff
    delays: tuple[float, ...] = _RETRY_DELAYS
    last_exc: Exception | None = None
    for attempt, delay in enumerate((*delays, None), start=1):
        try:
            resp = requests.get(url, params=params, timeout=_TIMEOUT)
            remaining = resp.headers.get("x-requests-remaining")
            if remaining is not None:
                logger.debug("Odds API quota remaining: %s", remaining)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            last_exc = exc
            logger.warning("Odds API attempt %d failed (%s): %s", attempt, url, exc)
            if delay is not None:
                time.sleep(delay)

    raise RuntimeError(f"Odds API unreachable: {url}") from last_exc


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_path(key: str) -> Path:
    safe = key.replace("/", "_").replace(":", "_")
    return _CACHE_DIR / f"{safe}.parquet"


def _cache_save(key: str, rows: list[dict]) -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(key)
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def _cache_load(key: str) -> list[dict] | None:
    path = _cache_path(key)
    if path.exists():
        return pd.read_parquet(path).to_dict("records")
    return None


# ---------------------------------------------------------------------------
# Game odds (h2h / spreads / totals)
# ---------------------------------------------------------------------------

def _parse_game_odds(event: dict, market_key: str) -> GameOdds | None:
    """Parse a single event's bookmaker lines for one market."""
    bookmakers = event.get("bookmakers", [])
    lines: list[BookmakerLine] = []
    for bm in bookmakers:
        for market in bm.get("markets", []):
            if market["key"] != market_key:
                continue
            outcomes: list[OutcomeOdds] = [
                {
                    "name": o["name"],
                    "price": int(o["price"]),
                    "point": o.get("point"),
                }
                for o in market.get("outcomes", [])
            ]
            lines.append(
                {
                    "bookmaker": bm["key"],
                    "last_update": bm.get("last_update", ""),
                    "outcomes": outcomes,
                }
            )

    if not lines:
        return None

    # Derive fair probabilities for h2h using consensus best price
    home_prob: float | None = None
    away_prob: float | None = None
    if market_key == "h2h":
        # Find the best (highest) price for each side across all books
        home_name = event.get("home_team", "")
        away_name = event.get("away_team", "")
        home_prices = [
            o["price"]
            for bm_line in lines
            for o in bm_line["outcomes"]
            if o["name"] == home_name
        ]
        away_prices = [
            o["price"]
            for bm_line in lines
            for o in bm_line["outcomes"]
            if o["name"] == away_name
        ]
        if home_prices and away_prices:
            # Use median price for a robust consensus
            import statistics
            med_home = statistics.median(home_prices)
            med_away = statistics.median(away_prices)
            home_prob, away_prob = devig_pair(med_home, med_away)

    return GameOdds(
        game_id=event["id"],
        sport=_SPORT,
        commence_time=event.get("commence_time", ""),
        home_team=event.get("home_team", ""),
        away_team=event.get("away_team", ""),
        market=market_key,
        lines=lines,
        home_implied_prob=home_prob,
        away_implied_prob=away_prob,
    )


def get_game_odds(
    regions: str = "us",
    markets: tuple[str, ...] | list[str] = _GAME_MARKETS,
    force_refresh: bool = False,
    try_once: bool = False,
) -> list[GameOdds]:
    """
    Return game-level odds for all upcoming MLB games.

    Parameters
    ----------
    regions       : comma-separated region string e.g. "us,uk"
    markets       : iterable of market keys to fetch
    force_refresh : bypass cache
    try_once      : if True, attempt the HTTP call exactly once with no
                    retry delays (use in live API endpoints to avoid
                    blocking the request for 30+ seconds on failure).

    Returns
    -------
    list[GameOdds] — one entry per game × market
    """
    markets_str = ",".join(markets)
    cache_key = f"game_odds_{date.today().isoformat()}_{regions}_{markets_str}"

    if not force_refresh:
        cached = _cache_load(cache_key)
        if cached is not None:
            logger.debug("Odds cache hit: %s", cache_key)
            return cached  # type: ignore[return-value]

    events = _fetch(
        f"sports/{_SPORT}/odds",
        params={"regions": regions, "markets": markets_str, "oddsFormat": "american"},
        max_retries=0 if try_once else None,
    )

    result: list[GameOdds] = []
    for event in events:
        for market_key in markets:
            parsed = _parse_game_odds(event, market_key)
            if parsed:
                result.append(parsed)

    # Flatten for Parquet (nested dicts → JSON strings)
    _cache_save(cache_key, [_flatten_game_odds(r) for r in result])
    return result


def _flatten_game_odds(g: GameOdds) -> dict:
    """Flatten nested lines list to JSON string for Parquet storage."""
    import json
    flat = dict(g)
    flat["lines"] = json.dumps(g["lines"])
    return flat


# ---------------------------------------------------------------------------
# Player props
# ---------------------------------------------------------------------------

def get_player_props(
    game_id: str,
    markets: tuple[str, ...] | list[str] | None = None,
    regions: str = "us",
    force_refresh: bool = False,
) -> list[PropLine]:
    """
    Return player prop lines for a specific *game_id*.

    Parameters
    ----------
    game_id  : The-Odds-API event ID (from get_game_odds)
    markets  : subset of _PROP_MARKET_MAP keys; defaults to all five
    regions  : comma-separated region string

    Returns
    -------
    list[PropLine]
    """
    if markets is None:
        markets = list(_PROP_MARKET_MAP.keys())

    api_markets = [_PROP_MARKET_MAP[m] for m in markets if m in _PROP_MARKET_MAP]
    if not api_markets:
        raise ValueError(f"No valid prop markets given. Valid: {list(_PROP_MARKET_MAP)}")

    markets_str = ",".join(api_markets)
    cache_key = f"props_{game_id}_{markets_str}_{date.today().isoformat()}"

    if not force_refresh:
        cached = _cache_load(cache_key)
        if cached is not None:
            logger.debug("Props cache hit: %s", cache_key)
            return cached  # type: ignore[return-value]

    raw = _fetch(
        f"sports/{_SPORT}/events/{game_id}/odds",
        params={"regions": regions, "markets": markets_str, "oddsFormat": "american"},
    )

    result: list[PropLine] = []
    for bm in raw.get("bookmakers", []):
        for market in bm.get("markets", []):
            market_key = market["key"]
            # Reverse-map API key → our canonical name
            canonical = next(
                (k for k, v in _PROP_MARKET_MAP.items() if v == market_key), market_key
            )
            # Props are structured as player name → over/under pairs
            outcomes = market.get("outcomes", [])
            # Group into (player, over, under) triples
            player_map: dict[str, dict] = {}
            for o in outcomes:
                pname = o.get("description") or o.get("name", "")
                side = o.get("name", "").lower()
                player_map.setdefault(pname, {})
                player_map[pname]["point"] = o.get("point")
                if side == "over":
                    player_map[pname]["over_odds"] = int(o["price"])
                elif side == "under":
                    player_map[pname]["under_odds"] = int(o["price"])

            for player_name, sides in player_map.items():
                over_odds = sides.get("over_odds")
                under_odds = sides.get("under_odds")
                over_prob: float | None = None
                under_prob: float | None = None
                if over_odds is not None and under_odds is not None:
                    over_prob, under_prob = devig_pair(over_odds, under_odds)
                result.append(
                    PropLine(
                        game_id=game_id,
                        market=canonical,
                        player_name=player_name,
                        bookmaker=bm["key"],
                        last_update=bm.get("last_update", ""),
                        point=sides.get("point"),
                        over_odds=over_odds,
                        under_odds=under_odds,
                        over_implied_prob=over_prob,
                        under_implied_prob=under_prob,
                    )
                )

    _cache_save(cache_key, result)  # type: ignore[arg-type]
    return result


# ---------------------------------------------------------------------------
# Snapshot (CLV tracking)
# ---------------------------------------------------------------------------

def snapshot_odds(
    snapshot_date: date | str | None = None,
    regions: str = "us",
) -> Path:
    """
    Pull and persist a full odds snapshot (game + all props) for *snapshot_date*.

    Snapshot files are written to:
        data/raw/odds/snapshot_{date}_{timestamp}.parquet

    Returns the path of the written Parquet file.
    """
    if snapshot_date is None:
        snapshot_date = date.today()
    date_str = snapshot_date.isoformat() if isinstance(snapshot_date, date) else snapshot_date
    ts = datetime.now(tz=timezone.utc).strftime("%H%M%S")

    game_odds = get_game_odds(regions=regions, force_refresh=True)

    rows: list[dict] = []
    for g in game_odds:
        flat = _flatten_game_odds(g)
        flat["snapshot_date"] = date_str
        flat["snapshot_ts"] = ts
        rows.append(flat)

    out_path = _CACHE_DIR / f"snapshot_{date_str}_{ts}.parquet"
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    logger.info("Odds snapshot written to %s (%d rows)", out_path, len(rows))
    return out_path
