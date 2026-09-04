"""
ingestion.statcast
==================
Wrapper around pybaseball to pull pitch-level Statcast data from
Baseball Savant.

Public API
----------
pull_statcast(start_date, end_date, cache_dir, chunk_days, force_refresh)
    -> pd.DataFrame   pitch-level Statcast with normalised dtypes

The range is pulled in *chunk_days*-day slices (default 7) to stay within
pybaseball rate limits.  A configurable sleep (default 1–2 s) is injected
between chunks, plus exponential back-off on failure.

Cache
-----
Each chunk is saved to:
    data/raw/statcast/statcast_{start}_{end}.parquet

On re-run the existing file is reused unless force_refresh=True.  After all
chunks are collected they are concatenated and deduplicated on (game_pk,
at_bat_number, pitch_number).
"""

from __future__ import annotations

import logging
import os
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Generator

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_DEFAULT_CACHE_DIR: Path = Path(__file__).resolve().parents[3] / "data" / "raw" / "statcast"
_CHUNK_DAYS: int = 7
_SLEEP_MIN: float = 1.0
_SLEEP_MAX: float = 2.0
_RETRY_DELAYS: tuple[float, ...] = (5.0, 15.0, 30.0)

# Columns to cast to specific dtypes after pull to reduce memory footprint
_FLOAT_COLS = [
    "release_speed", "release_spin_rate", "pfx_x", "pfx_z",
    "plate_x", "plate_z", "launch_speed", "launch_angle",
    "estimated_ba_using_speedangle", "estimated_woba_using_speedangle",
    "woba_value", "woba_denom", "babip_value",
    "iso_value", "launch_speed_angle",
]
_INT_COLS = [
    "at_bat_number", "pitch_number", "outs_when_up",
    "inning", "balls", "strikes",
]
_DEDUP_KEYS = ["game_pk", "at_bat_number", "pitch_number"]


# ---------------------------------------------------------------------------
# Date chunking
# ---------------------------------------------------------------------------

def _date_chunks(
    start: date, end: date, chunk_days: int
) -> Generator[tuple[date, date], None, None]:
    """Yield (chunk_start, chunk_end) pairs of at most *chunk_days* each."""
    current = start
    while current <= end:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end)
        yield current, chunk_end
        current = chunk_end + timedelta(days=1)


# ---------------------------------------------------------------------------
# Single-chunk pull with retry
# ---------------------------------------------------------------------------

def _pull_chunk(start: date, end: date, cache_dir: Path, force_refresh: bool) -> pd.DataFrame:
    """
    Pull one date-range chunk.  Returns a DataFrame (may be empty if no
    games in window).  Caches to Parquet on success.
    """
    start_str = start.isoformat()
    end_str = end.isoformat()
    cache_path = cache_dir / f"statcast_{start_str}_{end_str}.parquet"

    if cache_path.exists() and not force_refresh:
        logger.debug("Statcast cache hit: %s", cache_path)
        return pd.read_parquet(cache_path)

    # Lazy import — pybaseball is optional at import time
    try:
        from pybaseball import statcast as _statcast  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "pybaseball is required for Statcast pulls. "
            "Install it with: pip install pybaseball"
        ) from exc

    last_exc: Exception | None = None
    for attempt, delay in enumerate((*_RETRY_DELAYS, None), start=1):
        try:
            logger.info(
                "Pulling Statcast %s → %s (attempt %d)", start_str, end_str, attempt
            )
            # pybaseball suppress_warnings prevents noisy HTML parse warnings
            try:
                from pybaseball import cache as _pb_cache  # noqa: PLC0415
                _pb_cache.enable()
            except Exception:
                pass

            df: pd.DataFrame = _statcast(
                start_dt=start_str,
                end_dt=end_str,
                parallel=False,
            )
            if df is None or df.empty:
                logger.info("No Statcast data for %s → %s", start_str, end_str)
                df = pd.DataFrame()

            # Persist even empty frame so we don't re-query
            cache_dir.mkdir(parents=True, exist_ok=True)
            if not df.empty:
                df = _normalise_dtypes(df)
                df.to_parquet(cache_path, index=False)
            return df

        except Exception as exc:
            last_exc = exc
            logger.warning(
                "Statcast pull attempt %d failed (%s → %s): %s",
                attempt, start_str, end_str, exc,
            )
            if delay is not None:
                time.sleep(delay)

    raise RuntimeError(
        f"Statcast pull failed after {len(_RETRY_DELAYS)+1} attempts "
        f"({start_str} → {end_str})"
    ) from last_exc


# ---------------------------------------------------------------------------
# dtype normalisation
# ---------------------------------------------------------------------------

def _normalise_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Cast known columns to space-efficient numeric types."""
    for col in _FLOAT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
    for col in _INT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int16")
    # game_pk as Int32
    if "game_pk" in df.columns:
        df["game_pk"] = pd.to_numeric(df["game_pk"], errors="coerce").astype("Int32")
    # game_date as date
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date
    return df


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def pull_statcast(
    start_date: date | str,
    end_date: date | str | None = None,
    cache_dir: Path | str | None = None,
    chunk_days: int = _CHUNK_DAYS,
    force_refresh: bool = False,
    inter_chunk_sleep: float | None = None,
) -> pd.DataFrame:
    """
    Pull pitch-level Statcast data for *start_date* through *end_date*.

    Parameters
    ----------
    start_date      : first date of the range (YYYY-MM-DD or date object)
    end_date        : last date inclusive (defaults to start_date)
    cache_dir       : directory for Parquet chunks (default data/raw/statcast/)
    chunk_days      : days per API slice (default 7)
    force_refresh   : ignore existing cached files
    inter_chunk_sleep : seconds to sleep between chunks (default random 1–2 s)

    Returns
    -------
    pd.DataFrame    deduplicated pitch-level frame; empty DataFrame if no data
    """
    import random  # stdlib only used for sleep jitter — not cryptographic

    if isinstance(start_date, str):
        start_date = date.fromisoformat(start_date)
    if end_date is None:
        end_date = start_date
    if isinstance(end_date, str):
        end_date = date.fromisoformat(end_date)
    if start_date > end_date:
        raise ValueError(f"start_date {start_date} is after end_date {end_date}")

    cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR

    chunks: list[pd.DataFrame] = []
    date_pairs = list(_date_chunks(start_date, end_date, chunk_days))

    for i, (chunk_start, chunk_end) in enumerate(date_pairs):
        df_chunk = _pull_chunk(chunk_start, chunk_end, cache_dir, force_refresh)
        if not df_chunk.empty:
            chunks.append(df_chunk)
        # Rate-limit: sleep between chunks (skip after last chunk)
        if i < len(date_pairs) - 1:
            sleep_s = inter_chunk_sleep if inter_chunk_sleep is not None else random.uniform(
                _SLEEP_MIN, _SLEEP_MAX
            )
            logger.debug("Rate-limit sleep %.1f s", sleep_s)
            time.sleep(sleep_s)

    if not chunks:
        logger.info("No Statcast data found for %s → %s", start_date, end_date)
        return pd.DataFrame()

    combined = pd.concat(chunks, ignore_index=True)

    # Deduplicate on (game_pk, at_bat_number, pitch_number)
    present_keys = [k for k in _DEDUP_KEYS if k in combined.columns]
    if present_keys:
        before = len(combined)
        combined = combined.drop_duplicates(subset=present_keys)
        dropped = before - len(combined)
        if dropped:
            logger.info("Dropped %d duplicate pitches during concat", dropped)

    return combined.reset_index(drop=True)
