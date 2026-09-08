"""
ingestion.weather
=================
Fetches park-level weather at first pitch from the OpenWeather API
(api.openweathermap.org), then computes two derived physical quantities:

    1. Air density  — ideal gas law with Magnus-formula vapour pressure
       ρ = (Pd / (Rd · T)) + (Pv / (Rv · T))
       where
           T   = temperature in Kelvin
           Pd  = partial pressure of dry air  = P_total - Pv
           Pv  = vapour pressure (Magnus):
                 Pv = 0.61078 · exp(17.27·Tc / (Tc + 237.3)) kPa
           Rd  = 287.058 J/(kg·K)  (specific gas constant, dry air)
           Rv  = 461.495 J/(kg·K)  (specific gas constant, water vapour)

    2. Wind decomposition  — vector projection onto park orientation
       The park has a defined "batter-to-CF" bearing angle
       (orientation_degrees, measured clockwise from north).
       Wind *coming from* wind_dir_degrees is decomposed into:
           tailwind_ms  — positive = blowing out to CF (batter's back)
           crosswind_ms — positive = left-to-right from batter's POV

Endpoints used
--------------
Future/same-day  : api.openweathermap.org/data/2.5/forecast
                   (48-h ahead, 3-h intervals — free tier)
Current          : api.openweathermap.org/data/2.5/weather
Historical (>2h) : api.openweathermap.org/data/3.0/onecall/timemachine
                   (One Call 3.0 — requires paid subscription;
                    falls back to current+forecast gracefully)

Public API
----------
get_game_weather(park_id, game_datetime_utc, parks_json_path, force_refresh)
    -> WeatherResult  (TypedDict)

Environment
-----------
OPENWEATHER_API_KEY   (required for live fetch)
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import TypedDict

import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_BASE_CURRENT  = "https://api.openweathermap.org/data/2.5/weather"
_BASE_FORECAST = "https://api.openweathermap.org/data/2.5/forecast"
_BASE_ONECALL  = "https://api.openweathermap.org/data/3.0/onecall/timemachine"

_TIMEOUT = 15
_RETRY_DELAYS: tuple[float, ...] = (3.0, 8.0, 20.0)
_CACHE_DIR: Path = Path(__file__).resolve().parents[3] / "data" / "raw" / "weather"

# Physical constants
_Rd: float = 287.058   # J/(kg·K) — dry air
_Rv: float = 461.495   # J/(kg·K) — water vapour


# ---------------------------------------------------------------------------
# TypedDict result
# ---------------------------------------------------------------------------

class WeatherResult(TypedDict):
    park_id: str
    game_datetime_utc: str
    # Raw fields
    temp_c: float
    humidity_pct: float
    wind_speed_ms: float
    wind_dir_degrees: float
    pressure_kpa: float
    conditions: str
    # Derived: air density
    vapour_pressure_kpa: float
    air_density_kg_m3: float
    # Derived: wind decomposition
    park_orientation_degrees: float
    tailwind_ms: float
    crosswind_ms: float


# ---------------------------------------------------------------------------
# Park lookup
# ---------------------------------------------------------------------------

def _load_park(park_id: str, parks_json_path: Path | str | None) -> dict:
    if parks_json_path is None:
        parks_json_path = Path(__file__).resolve().parents[3] / "data" / "parks.json"
    parks_path = Path(parks_json_path)
    if not parks_path.exists():
        raise FileNotFoundError(f"parks.json not found at {parks_path}")
    with parks_path.open() as fh:
        parks: dict = json.load(fh)
    if str(park_id) not in parks:
        raise KeyError(f"park_id '{park_id}' not found in {parks_path}")
    return parks[str(park_id)]


# ---------------------------------------------------------------------------
# Physics helpers (unchanged from Visual Crossing version)
# ---------------------------------------------------------------------------

def _magnus_vapour_pressure_kpa(temp_c: float) -> float:
    return 0.61078 * math.exp(17.27 * temp_c / (temp_c + 237.3))


def _air_density(
    temp_c: float, humidity_pct: float, pressure_kpa: float
) -> tuple[float, float]:
    T_k = temp_c + 273.15
    es = _magnus_vapour_pressure_kpa(temp_c)
    Pv = es * (humidity_pct / 100.0)
    Pd = pressure_kpa - Pv
    rho = (Pd * 1000.0) / (_Rd * T_k) + (Pv * 1000.0) / (_Rv * T_k)
    return Pv, rho


def _wind_decomposition(
    wind_speed_ms: float,
    wind_dir_degrees: float,
    park_orientation_degrees: float,
) -> tuple[float, float]:
    wind_toward_deg = (wind_dir_degrees + 180.0) % 360.0

    def _bearing_to_rad(b: float) -> float:
        return math.radians(90.0 - b)

    wind_angle = _bearing_to_rad(wind_toward_deg)
    park_angle = _bearing_to_rad(park_orientation_degrees)

    wx, wy = math.cos(wind_angle), math.sin(wind_angle)
    px, py = math.cos(park_angle),  math.sin(park_angle)

    tailwind_ms  = wind_speed_ms * (wx * px + wy * py)
    crosswind_ms = wind_speed_ms * (wx * py - wy * px)
    return tailwind_ms, crosswind_ms


# ---------------------------------------------------------------------------
# OpenWeather API fetch helpers
# ---------------------------------------------------------------------------

def _ow_get(url: str, params: dict) -> dict:
    """GET with retry/backoff. Injects OPENWEATHER_API_KEY."""
    api_key = os.getenv("OPENWEATHER_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "OPENWEATHER_API_KEY is not set. Add it to your .env file."
        )
    params = {**params, "appid": api_key, "units": "metric"}

    last_exc: Exception | None = None
    for attempt, delay in enumerate((*_RETRY_DELAYS, None), start=1):
        try:
            resp = requests.get(url, params=params, timeout=_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            last_exc = exc
            logger.warning("OpenWeather attempt %d failed: %s", attempt, exc)
            if delay is not None:
                time.sleep(delay)
    raise RuntimeError("OpenWeather API unreachable") from last_exc


def _obs_from_current(lat: float, lon: float) -> dict:
    """Current conditions — used when game is within ~2 h of now."""
    raw = _ow_get(_BASE_CURRENT, {"lat": lat, "lon": lon})
    main = raw.get("main", {})
    wind = raw.get("wind", {})
    weather = raw.get("weather", [{}])
    return {
        "temp_c":         float(main.get("temp", 20.0)),
        "humidity_pct":   float(main.get("humidity", 50.0)),
        "wind_speed_ms":  float(wind.get("speed", 0.0)),
        "wind_dir_deg":   float(wind.get("deg", 0.0)),
        "pressure_hpa":   float(main.get("pressure", 1013.25)),
        "conditions":     weather[0].get("description", ""),
    }


def _obs_from_forecast(lat: float, lon: float, target_utc: datetime) -> dict:
    """
    Pick the 3-h forecast slot closest to *target_utc* from the free
    5-day/3-h forecast endpoint.  Falls back gracefully if no match.
    """
    raw = _ow_get(_BASE_FORECAST, {"lat": lat, "lon": lon, "cnt": 40})
    slots = raw.get("list", [])
    if not slots:
        return _obs_from_current(lat, lon)

    target_ts = target_utc.timestamp()
    best = min(slots, key=lambda s: abs(s.get("dt", 0) - target_ts))
    main = best.get("main", {})
    wind = best.get("wind", {})
    weather = best.get("weather", [{}])
    return {
        "temp_c":         float(main.get("temp", 20.0)),
        "humidity_pct":   float(main.get("humidity", 50.0)),
        "wind_speed_ms":  float(wind.get("speed", 0.0)),
        "wind_dir_deg":   float(wind.get("deg", 0.0)),
        "pressure_hpa":   float(main.get("pressure", 1013.25)),
        "conditions":     weather[0].get("description", ""),
    }


def _obs_from_timemachine(lat: float, lon: float, target_utc: datetime) -> dict:
    """
    Historical One Call 3.0 endpoint.  Requires a paid OpenWeather
    subscription — degrades to forecast if the call fails.
    """
    try:
        raw = _ow_get(
            _BASE_ONECALL,
            {"lat": lat, "lon": lon, "dt": int(target_utc.timestamp())},
        )
        hourly = raw.get("data", [{}])
        obs = hourly[0] if hourly else {}
        return {
            "temp_c":         float(obs.get("temp", 20.0)),
            "humidity_pct":   float(obs.get("humidity", 50.0)),
            "wind_speed_ms":  float(obs.get("wind_speed", 0.0)),
            "wind_dir_deg":   float(obs.get("wind_deg", 0.0)),
            "pressure_hpa":   float(obs.get("pressure", 1013.25)),
            "conditions":     obs.get("weather", [{}])[0].get("description", ""),
        }
    except Exception as exc:
        logger.warning("One Call 3.0 failed (%s) — falling back to forecast", exc)
        return _obs_from_forecast(lat, lon, target_utc)


def _fetch_obs(lat: float, lon: float, target_utc: datetime) -> dict:
    """
    Route to the best available endpoint based on how far ahead *target_utc* is.

    ±2 h of now  → current conditions
    2 h – 5 days → 3-h forecast
    >5 days past → One Call timemachine (paid) or forecast fallback
    """
    now = datetime.now(tz=timezone.utc)
    delta_h = (target_utc - now).total_seconds() / 3600.0

    if abs(delta_h) <= 2:
        return _obs_from_current(lat, lon)
    if delta_h > 0:                     # future, within forecast window
        return _obs_from_forecast(lat, lon, target_utc)
    # Historical — try One Call 3.0 then fall back
    return _obs_from_timemachine(lat, lon, target_utc)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_key(park_id: str, dt_utc: datetime) -> str:
    return f"weather_{park_id}_{dt_utc.strftime('%Y%m%dT%H%M')}"


def _load_cached(key: str) -> WeatherResult | None:
    path = _CACHE_DIR / f"{key}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        return dict(df.iloc[0])  # type: ignore[return-value]
    return None


def _save_cached(key: str, result: WeatherResult) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_parquet(_CACHE_DIR / f"{key}.parquet", index=False)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_game_weather(
    park_id: str | int,
    game_datetime_utc: datetime | str,
    parks_json_path: Path | str | None = None,
    force_refresh: bool = False,
) -> WeatherResult:
    """
    Return raw and derived weather conditions for a game at *park_id*.

    Parameters
    ----------
    park_id            : key matching an entry in data/parks.json
    game_datetime_utc  : UTC datetime of first pitch (datetime or ISO string)
    parks_json_path    : path to parks.json (defaults to data/parks.json)
    force_refresh      : bypass on-disk cache

    Returns
    -------
    WeatherResult TypedDict with raw + derived fields (see module docstring)
    """
    if isinstance(game_datetime_utc, str):
        game_datetime_utc = datetime.fromisoformat(game_datetime_utc)
    if game_datetime_utc.tzinfo is None:
        game_datetime_utc = game_datetime_utc.replace(tzinfo=timezone.utc)

    park_id = str(park_id)
    ckey = _cache_key(park_id, game_datetime_utc)

    if not force_refresh:
        cached = _load_cached(ckey)
        if cached is not None:
            logger.debug("Weather cache hit: %s", ckey)
            return cached  # type: ignore[return-value]

    park = _load_park(park_id, parks_json_path)
    lat: float  = float(park["lat"])
    lon: float  = float(park["lon"])
    orientation_deg: float = float(park["orientation_degrees"])

    obs = _fetch_obs(lat, lon, game_datetime_utc)

    temp_c        = obs["temp_c"]
    humidity_pct  = obs["humidity_pct"]
    wind_speed_ms = obs["wind_speed_ms"]
    wind_dir_deg  = obs["wind_dir_deg"]
    pressure_kpa  = obs["pressure_hpa"] / 10.0   # hPa → kPa
    conditions    = obs["conditions"]

    Pv, rho = _air_density(temp_c, humidity_pct, pressure_kpa)
    tailwind_ms, crosswind_ms = _wind_decomposition(
        wind_speed_ms, wind_dir_deg, orientation_deg
    )

    result: WeatherResult = {
        "park_id":                   park_id,
        "game_datetime_utc":         game_datetime_utc.isoformat(),
        "temp_c":                    temp_c,
        "humidity_pct":              humidity_pct,
        "wind_speed_ms":             round(wind_speed_ms, 3),
        "wind_dir_degrees":          wind_dir_deg,
        "pressure_kpa":              round(pressure_kpa, 4),
        "conditions":                conditions,
        "vapour_pressure_kpa":       round(Pv, 5),
        "air_density_kg_m3":         round(rho, 5),
        "park_orientation_degrees":  orientation_deg,
        "tailwind_ms":               round(tailwind_ms, 3),
        "crosswind_ms":              round(crosswind_ms, 3),
    }

    _save_cached(ckey, result)
    return result
