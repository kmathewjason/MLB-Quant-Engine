"""
features.park_weather
=====================
Computes per-stat park factors from multi-year home/road splits and merges
derived weather features (air density, wind) from ingestion.weather.

Statistical justification
--------------------------
A park factor for statistic S is estimated as:

    PF_S(venue) = (home_rate_S / road_rate_S) / (league_home_rate_S / league_road_rate_S)

where "rate" = S events per plate appearance (or per AB for counting stats).

The ratio-of-ratios normalises out the league-wide home-field tendency
(batters hit slightly better at home in every park, not just hitter-friendly
ones).  A PF of 1.0 = perfectly neutral; 1.10 = 10% inflation.

Multi-year stability
--------------------
Single-season park factors have high variance (~±0.05 for HRs, ±0.02 for
runs).  Three years of data with exponential weighting reduces variance:

    PF_weighted = Σ_t w_t · PF_t / Σ_t w_t

with w_t = decay^(current_year - t), decay = 0.5 by default
(current year carries weight 1, prior year 0.5, two years ago 0.25).

Weather integration
--------------------
Air density from weather.py affects ball flight distance:
    distance_factor = (rho_standard / rho_actual)^0.5  ≈ 1 ± 0.03

Wind tailwind/crosswind components modulate HR probability:
    wind_hr_factor = 1 + tailwind_coef * tailwind_ms / 10
                       + crosswind_coef * |crosswind_ms| / 10

where tailwind_coef ≈ 0.05 (each 10 m/s tailwind ≈ 5% HR inflation) and
crosswind_coef ≈ -0.01 (crosswind slightly suppresses HR).

Combined run-environment multiplier:
    env_factor = PF_runs * air_factor * wind_factor

Public API
----------
compute_park_factor(home_away_df, stat, years, decay)     -> pd.DataFrame
ewma_park_factor(annual_pf_df, decay)                     -> pd.Series
merge_weather(pf_series, weather_result)                  -> EnvFactor
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Supported stat categories for park factor computation
StatCategory = Literal["runs", "HR", "2B", "3B", "1B", "BB", "BABIP"]

# Standard sea-level air density used as reference for flight-distance adjustment
_RHO_STANDARD: float = 1.2250  # kg/m³ at 15°C, 0% RH, 101.325 kPa

# Wind effect coefficients (empirically calibrated; see Statcast Hit Probability literature)
_TAILWIND_HR_COEF:   float = 0.050   # per 10 m/s tailwind
_CROSSWIND_HR_COEF:  float = -0.010  # per 10 m/s |crosswind| (slight suppression)
_TAILWIND_RUN_COEF:  float = 0.025   # runs are less sensitive than HRs
_CROSSWIND_RUN_COEF: float = -0.005


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EnvFactor:
    """
    Combined run-environment multiplier for a single game.

    Fields
    ------
    park_factor_runs : raw multi-year EWMA park factor for runs
    park_factor_hr   : raw multi-year EWMA park factor for HR
    air_density_factor: (rho_standard / rho_actual)^0.5 — ball-flight effect
    wind_hr_factor   : multiplicative wind adjustment for HR probability
    wind_run_factor  : multiplicative wind adjustment for run scoring
    env_run_factor   : final combined run-environment multiplier
    env_hr_factor    : final combined HR-environment multiplier
    """
    park_factor_runs: float
    park_factor_hr: float
    air_density_factor: float
    wind_hr_factor: float
    wind_run_factor: float

    @property
    def env_run_factor(self) -> float:
        return self.park_factor_runs * self.air_density_factor * self.wind_run_factor

    @property
    def env_hr_factor(self) -> float:
        return self.park_factor_hr * self.air_density_factor * self.wind_hr_factor


# ---------------------------------------------------------------------------
# Park factor computation
# ---------------------------------------------------------------------------

def compute_park_factor(
    home_away_df: pd.DataFrame,
    stat: StatCategory = "runs",
    venue_col: str = "venue_id",
    season_col: str = "season",
    home_col: str = "is_home",
    stat_col: str | None = None,
    pa_col: str = "PA",
) -> pd.DataFrame:
    """
    Compute per-venue, per-season park factors for *stat* from a
    home/away game log.

    Parameters
    ----------
    home_away_df : DataFrame with one row per team-game, columns:
                   venue_id, season, is_home (bool), <stat_col>, PA
    stat         : which stat to compute PF for (used to derive stat_col default)
    stat_col     : column name for the raw count; defaults to stat
    pa_col       : column name for plate appearances (denominator)

    Returns
    -------
    DataFrame with columns: venue_id, season, park_factor
    indexed by (venue_id, season).
    """
    if stat_col is None:
        stat_col = stat

    needed = {venue_col, season_col, home_col, stat_col, pa_col}
    missing = needed - set(home_away_df.columns)
    if missing:
        raise ValueError(f"home_away_df missing columns: {missing}")

    df = home_away_df.copy()
    df["_rate"] = df[stat_col] / df[pa_col].replace(0, np.nan)

    # Aggregate: (venue, season, home_flag) → mean rate
    agg = (
        df.groupby([venue_col, season_col, home_col])["_rate"]
        .mean()
        .unstack(fill_value=np.nan)  # columns: True (home), False (away)
    )

    # Rename columns to home/away
    rename = {True: "home_rate", False: "away_rate"}
    agg = agg.rename(columns={c: rename.get(c, str(c)) for c in agg.columns})

    if "home_rate" not in agg.columns or "away_rate" not in agg.columns:
        raise ValueError(
            "Could not identify home/away split columns. "
            "Ensure is_home column contains True/False values."
        )

    # League-level home/away rates per season (average across all venues)
    league_agg = (
        df.groupby([season_col, home_col])["_rate"]
        .mean()
        .unstack(fill_value=np.nan)
        .rename(columns={True: "lg_home_rate", False: "lg_away_rate"})
    )

    # Join league rates back
    agg = agg.join(league_agg, on=season_col)

    # Park factor formula
    agg["park_factor"] = (agg["home_rate"] / agg["away_rate"]) / (
        agg["lg_home_rate"] / agg["lg_away_rate"]
    )

    result = (
        agg[["park_factor"]]
        .reset_index()
        .rename(columns={venue_col: "venue_id", season_col: "season"})
    )
    return result


def ewma_park_factor(
    annual_pf_df: pd.DataFrame,
    current_season: int | None = None,
    decay: float = 0.5,
    venue_col: str = "venue_id",
    season_col: str = "season",
    pf_col: str = "park_factor",
) -> pd.Series:
    """
    Compute exponentially weighted multi-year park factor per venue.

    Weight for season t = decay^(current_season - t).
    Returns pd.Series indexed by venue_id.

    Parameters
    ----------
    annual_pf_df : output of compute_park_factor(), one row per (venue, season)
    current_season : reference year for weighting (defaults to max season in data)
    decay        : per-year discount factor (default 0.5 → halves each year)
    """
    if current_season is None:
        current_season = int(annual_pf_df[season_col].max())

    df = annual_pf_df.copy()
    df["weight"] = decay ** (current_season - df[season_col].astype(int))

    weighted = (
        df.groupby(venue_col)
        .apply(lambda g: np.average(g[pf_col].fillna(1.0), weights=g["weight"]))
        .rename("ewma_park_factor")
    )
    return weighted


# ---------------------------------------------------------------------------
# Weather integration
# ---------------------------------------------------------------------------

def _air_density_factor(rho_actual: float) -> float:
    """
    Return the ball-flight distance multiplier for *rho_actual* kg/m³.

    Based on aerodynamic drag: distance ∝ 1/sqrt(ρ) for long fly balls.
    We express this relative to standard sea-level density (1.225 kg/m³).
    """
    if rho_actual <= 0:
        raise ValueError(f"Air density must be positive; got {rho_actual}")
    return float(np.sqrt(_RHO_STANDARD / rho_actual))


def _wind_factor(
    tailwind_ms: float,
    crosswind_ms: float,
    tailwind_coef: float,
    crosswind_coef: float,
) -> float:
    """Return wind adjustment factor (multiplicative, centred at 1.0)."""
    return max(
        0.5,
        min(
            2.0,
            1.0
            + tailwind_coef  * tailwind_ms   / 10.0
            + crosswind_coef * abs(crosswind_ms) / 10.0,
        ),
    )


def merge_weather(
    park_factor_runs: float,
    park_factor_hr: float,
    weather: dict,
) -> EnvFactor:
    """
    Combine park factors with live weather data into an EnvFactor.

    Parameters
    ----------
    park_factor_runs : EWMA park factor for run scoring
    park_factor_hr   : EWMA park factor for HR rate
    weather          : WeatherResult dict from src.ingestion.weather.get_game_weather()

    Returns
    -------
    EnvFactor with all raw and derived fields populated.
    """
    rho = float(weather.get("air_density_kg_m3", _RHO_STANDARD))
    tailwind  = float(weather.get("tailwind_ms",  0.0))
    crosswind = float(weather.get("crosswind_ms", 0.0))

    air_factor = _air_density_factor(rho)
    wf_hr  = _wind_factor(tailwind, crosswind, _TAILWIND_HR_COEF,  _CROSSWIND_HR_COEF)
    wf_run = _wind_factor(tailwind, crosswind, _TAILWIND_RUN_COEF, _CROSSWIND_RUN_COEF)

    return EnvFactor(
        park_factor_runs=park_factor_runs,
        park_factor_hr=park_factor_hr,
        air_density_factor=round(air_factor, 5),
        wind_hr_factor=round(wf_hr, 5),
        wind_run_factor=round(wf_run, 5),
    )


# ---------------------------------------------------------------------------
# Convenience: neutral environment (PF=1, standard air, no wind)
# ---------------------------------------------------------------------------

NEUTRAL_ENV = EnvFactor(
    park_factor_runs=1.0,
    park_factor_hr=1.0,
    air_density_factor=1.0,
    wind_hr_factor=1.0,
    wind_run_factor=1.0,
)
