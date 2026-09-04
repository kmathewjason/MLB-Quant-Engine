"""
ingestion.weather
=================
Fetches park-level weather at first pitch from the Visual Crossing Weather
API, then computes two derived physical quantities:

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
       The park has a defined "batter-to-CF" bearing angle (orientation_degrees,
       measured clockwise from north, i.e. standard meteorological bearing).
       Wind *coming from* wind_dir_degrees is decomposed into:
           tailwind_ms  — positive = blowing out to CF (batter's back)
           crosswind_ms — positive = left-to-right from batter's perspective

Public API
----------
get_game_weather(park_id, game_datetime_utc, parks_json_path, force_refresh)
    -> WeatherResult  (TypedDict)

Environment
-----------
VISUAL_CROSSING_API_KEY   (required for live fetch)
"""

from __future__ import annotations

import logging
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

import json
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_API_BASE = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
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
    wind_dir_degrees: float   # meteorological: direction wind is coming FROM
    pressure_kpa: float
    conditions: str
    # Derived: air density
    vapour_pressure_kpa: float
    air_density_kg_m3: float
    # Derived: wind decomposition (relative to park CF axis)
    park_orientation_degrees: float  # bearing from home plate toward CF
    tailwind_ms: float    # + = blowing out (batter's back); – = headwind
    crosswind_ms: float   # + = left-to-right from batter's POV; – = right-to-left


# ---------------------------------------------------------------------------
# Park lookup
# ---------------------------------------------------------------------------

def _load_park(park_id: str, parks_json_path: Path | str | None) -> dict:
    """Return the park entry for *park_id* from parks.json."""
    if parks_json_path is None:
        parks_json_path = Path(__file__).resolve().parents[3] / "data" / "parks.json"
    parks_path = Path(parks_json_path)
    if not parks_path.exists():
        raise FileNotFoundError(
            f"parks.json not found at {parks_path}. "
            "Please provide a parks.json with lat, lon, orientation_degrees per park."
        )
    with parks_path.open() as fh:
        parks: dict = json.load(fh)
    if str(park_id) not in parks:
        raise KeyError(f"park_id '{park_id}' not found in {parks_path}")
    return parks[str(park_id)]


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------

def _magnus_vapour_pressure_kpa(temp_c: float) -> float:
    """
    Magnus approximation for saturation vapour pressure at *temp_c* (°C).
    Returns pressure in kPa.

    Formula: es = 0.61078 · exp(17.27 · T / (T + 237.3))
    """
    return 0.61078 * math.exp(17.27 * temp_c / (temp_c + 237.3))


def _air_density(temp_c: float, humidity_pct: float, pressure_kpa: float) -> tuple[float, float]:
    """
    Compute (vapour_pressure_kpa, air_density_kg_m3) from observed conditions.

    Parameters
    ----------
    temp_c       : dry-bulb temperature in Celsius
    humidity_pct : relative humidity 0–100
    pressure_kpa : station pressure in kPa

    Returns
    -------
    (Pv_kpa, rho_kg_m3)
    """
    T_k = temp_c + 273.15                             # Kelvin
    es = _magnus_vapour_pressure_kpa(temp_c)           # saturation vapour pressure
    Pv = es * (humidity_pct / 100.0)                   # actual vapour pressure (kPa)
    Pd = pressure_kpa - Pv                             # partial pressure dry air (kPa)

    # Convert kPa → Pa for SI calculation
    rho = (Pd * 1000.0) / (_Rd * T_k) + (Pv * 1000.0) / (_Rv * T_k)
    return Pv, rho


def _wind_decomposition(
    wind_speed_ms: float,
    wind_dir_degrees: float,
    park_orientation_degrees: float,
) -> tuple[float, float]:
    """
    Decompose wind into tailwind and crosswind components relative to a park.

    Conventions
    -----------
    wind_dir_degrees       : met convention — direction wind is coming FROM,
                             clockwise from north.
    park_orientation_degrees: bearing from home plate toward CF (batter faces
                             this direction), clockwise from north.

    A wind blowing FROM 180° (south) toward 0° (north) is a tailwind if the
    CF is at 0° (batter facing north, wind at the batter's back).

    Returns
    -------
    tailwind_ms  : positive = blowing out to CF
    crosswind_ms : positive = from batter's left (1B side) to right (3B side)
                   when facing CF; sign depends on cross-product direction
    """
    # Convert wind "coming from" to "going toward" vector
    wind_toward_deg = (wind_dir_degrees + 180.0) % 360.0

    # Convert both bearings to standard math angles (CCW from east)
    def _bearing_to_rad(bearing_deg: float) -> float:
        return math.radians(90.0 - bearing_deg)

    wind_angle = _bearing_to_rad(wind_toward_deg)
    park_angle = _bearing_to_rad(park_orientation_degrees)

    # Unit vector for each direction
    wx = math.cos(wind_angle)
    wy = math.sin(wind_angle)
    px = math.cos(park_angle)
    py = math.sin(park_angle)

    # Tailwind: projection of wind vector onto park-CF axis
    tailwind_ms = wind_speed_ms * (wx * px + wy * py)

    # Crosswind: magnitude of perpendicular component
    # Sign: positive = wind pushes from batter's left to right (1B→3B side)
    # Cross product z-component gives signed perpendicular
    crosswind_ms = wind_speed_ms * (wx * py - wy * px)

    return tailwind_ms, crosswind_ms


# ---------------------------------------------------------------------------
# Visual Crossing API fetch
# ---------------------------------------------------------------------------

def _fetch_visual_crossing(lat: float, lon: float, dt_utc: datetime) -> dict:
    """Call Visual Crossing Timeline API for a single datetime point."""
    api_key = os.getenv("VISUAL_CROSSING_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "VISUAL_CROSSING_API_KEY is not set. "
            "Add it to your .env file."
        )

    # Visual Crossing expects local datetime; pass as UTC ISO string
    dt_str = dt_utc.strftime("%Y-%m-%dT%H:%M:%S")
    url = f"{_API_BASE}/{lat},{lon}/{dt_str}/{dt_str}"
    params = {
        "unitGroup": "metric",
        "include": "hours",
        "elements": "temp,humidity,windspeed,winddir,pressure,conditions",
        "key": api_key,
        "contentType": "json",
    }

    last_exc: Exception | None = None
    for attempt, delay in enumerate((*_RETRY_DELAYS, None), start=1):
        try:
            resp = requests.get(url, params=params, timeout=_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            last_exc = exc
            logger.warning("Visual Crossing attempt %d failed: %s", attempt, exc)
            if delay is not None:
                time.sleep(delay)

    raise RuntimeError("Visual Crossing API unreachable") from last_exc


def _extract_hourly(raw: dict, target_hour: int) -> dict:
    """Pull the hourly observation closest to *target_hour* (0–23)."""
    days = raw.get("days", [])
    if not days:
        raise ValueError("Visual Crossing response contains no day data")
    hours = days[0].get("hours", [])
    if not hours:
        # Fall back to day-level aggregate
        return days[0]
    # Find closest hour
    best = min(hours, key=lambda h: abs(int(h.get("datetime", "0:0:0").split(":")[0]) - target_hour))
    return best


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
# Main public function
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
    park_id            : key matching an entry in parks.json
    game_datetime_utc  : UTC datetime of first pitch (datetime or ISO string)
    parks_json_path    : path to parks.json (defaults to data/parks.json)
    force_refresh      : bypass cache

    Returns
    -------
    WeatherResult TypedDict with raw + derived fields (see module docstring)
    """
    if isinstance(game_datetime_utc, str):
        game_datetime_utc = datetime.fromisoformat(game_datetime_utc)
    if game_datetime_utc.tzinfo is None:
        game_datetime_utc = game_datetime_utc.replace(tzinfo=timezone.utc)

    park_id = str(park_id)
    cache_key = _cache_key(park_id, game_datetime_utc)

    if not force_refresh:
        cached = _load_cached(cache_key)
        if cached is not None:
            logger.debug("Weather cache hit: %s", cache_key)
            return cached  # type: ignore[return-value]

    park = _load_park(park_id, parks_json_path)
    lat: float = park["lat"]
    lon: float = park["lon"]
    orientation_deg: float = park["orientation_degrees"]

    raw = _fetch_visual_crossing(lat, lon, game_datetime_utc)
    obs = _extract_hourly(raw, target_hour=game_datetime_utc.hour)

    # Unpack — Visual Crossing metric units: temp °C, windspeed km/h, pressure hPa
    temp_c: float = float(obs.get("temp", 20.0))
    humidity_pct: float = float(obs.get("humidity", 50.0))
    wind_speed_kmh: float = float(obs.get("windspeed", 0.0))
    wind_dir_deg: float = float(obs.get("winddir", 0.0))
    pressure_hpa: float = float(obs.get("pressure", 1013.25))
    conditions: str = str(obs.get("conditions", ""))

    wind_speed_ms = wind_speed_kmh / 3.6          # km/h → m/s
    pressure_kpa = pressure_hpa / 10.0            # hPa → kPa

    Pv, rho = _air_density(temp_c, humidity_pct, pressure_kpa)
    tailwind_ms, crosswind_ms = _wind_decomposition(wind_speed_ms, wind_dir_deg, orientation_deg)

    result: WeatherResult = {
        "park_id": park_id,
        "game_datetime_utc": game_datetime_utc.isoformat(),
        # Raw
        "temp_c": temp_c,
        "humidity_pct": humidity_pct,
        "wind_speed_ms": round(wind_speed_ms, 3),
        "wind_dir_degrees": wind_dir_deg,
        "pressure_kpa": round(pressure_kpa, 4),
        "conditions": conditions,
        # Derived
        "vapour_pressure_kpa": round(Pv, 5),
        "air_density_kg_m3": round(rho, 5),
        "park_orientation_degrees": orientation_deg,
        "tailwind_ms": round(tailwind_ms, 3),
        "crosswind_ms": round(crosswind_ms, 3),
    }

    _save_cached(cache_key, result)
    return result
