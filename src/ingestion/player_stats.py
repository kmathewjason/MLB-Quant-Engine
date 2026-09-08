"""
ingestion.player_stats
======================
Fetches current-season batting and pitching rate stats from the MLB Stats API
for use in the per-game simulation pipeline.

Public API
----------
get_pitcher_rates(player_id, season)  -> dict[str, float]
    PA-outcome rate dict keyed by PA_OUTCOMES canonical names.
    Returns shrunk-to-league-average rates on failure.

get_batter_rates(player_id, season)   -> dict[str, float]
    PA-outcome rate dict.

get_team_lineup_rates(game_pk, side)  -> np.ndarray  shape (9, 7)
    Build a (9 batters × 7 outcomes) PA-prob matrix for one side of a game,
    using log5 matchup adjustment against the opposing probable pitcher.

get_game_pa_probs(game_pk)            -> tuple[np.ndarray, np.ndarray]
    Returns (home_pa_probs, away_pa_probs), each shape (9, 7).

Outcome ordering (7 outcomes matches game_simulator._simulate_half_inning_variable)
---------------------------------------------------------------------------
Index  Outcome   Notes
  0    1B
  1    2B
  2    3B
  3    HR
  4    BB        includes HBP
  5    K
  6    OUT       all other outs

Environment
-----------
MLB_STATS_API_BASE  (default: https://statsapi.mlb.com/api/v1)
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# League-average fallback rates in 7-outcome space
# (2019-2024 MLB averages, excluding Covid-shortened 2020)
# ---------------------------------------------------------------------------
_LEAGUE_AVG_7: dict[str, float] = {
    "1B":  0.1479,
    "2B":  0.0472,
    "3B":  0.0046,
    "HR":  0.0337,
    "BB":  0.0948,   # BB + HBP combined
    "K":   0.2217,
    "OUT": 0.4501,
}

# Canonical 7-outcome ordering used by game_simulator
_OUTCOMES_7 = ["1B", "2B", "3B", "HR", "BB", "K", "OUT"]


def _league_avg_array() -> np.ndarray:
    """Return normalised (7,) league-average PA rate array."""
    arr = np.array([_LEAGUE_AVG_7[k] for k in _OUTCOMES_7], dtype=np.float64)
    return arr / arr.sum()


def _lg_tile() -> np.ndarray:
    """Return (9, 7) league-average PA matrix (same row for all 9 lineup slots)."""
    return np.tile(_league_avg_array(), (9, 1))


# ---------------------------------------------------------------------------
# MLB Stats API helpers (reuse mlb_stats_api._fetch/_load_or_fetch pattern)
# ---------------------------------------------------------------------------

def _get_player_season_stats(player_id: int, season: int, group: str) -> dict[str, Any]:
    """
    Fetch {group} stats for a player from /people/{id}/stats.

    group: 'hitting' or 'pitching'
    Returns the first stats block dict, or {} on failure.
    """
    from src.ingestion.mlb_stats_api import _load_or_fetch  # noqa: PLC0415
    try:
        data = _load_or_fetch(
            cache_key=f"player_stats_{player_id}_{season}_{group}",
            endpoint=f"people/{player_id}/stats",
            params={"stats": "season", "season": season, "group": group},
            force_refresh=False,
        )
        for stat_block in data.get("stats", []):
            splits = stat_block.get("splits", [])
            if splits:
                return splits[0].get("stat", {})
    except Exception as exc:
        logger.debug("Player stats fetch failed player_id=%d group=%s: %s", player_id, group, exc)
    return {}


# ---------------------------------------------------------------------------
# Rate converters: MLB Stats API stat keys → 7-outcome rate dict
# ---------------------------------------------------------------------------

def _hitting_stats_to_rates(stat: dict[str, Any]) -> dict[str, float] | None:
    """
    Convert MLB Stats API hitting stat block → PA-outcome rate dict.

    Returns None if the sample is too small (< 50 PA) to be informative.
    """
    try:
        pa  = int(stat.get("plateAppearances") or stat.get("atBats") or 0)
        if pa < 50:
            return None
        h   = int(stat.get("hits") or 0)
        d   = int(stat.get("doubles") or 0)
        t   = int(stat.get("triples") or 0)
        hr  = int(stat.get("homeRuns") or 0)
        bb  = int(stat.get("baseOnBalls") or 0)
        hbp = int(stat.get("hitByPitch") or 0)
        so  = int(stat.get("strikeOuts") or 0)
        singles = h - d - t - hr
        bb_total = bb + hbp
        other_out = pa - singles - d - t - hr - bb_total - so
        if other_out < 0:
            other_out = 0
        rates = {
            "1B":  singles   / pa,
            "2B":  d         / pa,
            "3B":  t         / pa,
            "HR":  hr        / pa,
            "BB":  bb_total  / pa,
            "K":   so        / pa,
            "OUT": other_out / pa,
        }
        # Ensure non-negative and normalise
        arr = np.array([max(0.0, rates[k]) for k in _OUTCOMES_7], dtype=np.float64)
        total = arr.sum()
        if total <= 0:
            return None
        arr = arr / total
        return {k: float(arr[i]) for i, k in enumerate(_OUTCOMES_7)}
    except Exception as exc:
        logger.debug("hitting_stats_to_rates failed: %s", exc)
        return None


def _pitching_stats_to_rates(stat: dict[str, Any]) -> dict[str, float] | None:
    """
    Convert MLB Stats API pitching stat block → PA-outcome rate dict
    (how the pitcher allows outcomes, from the batter's perspective).

    Returns None if sample is too small (< 150 BF).
    """
    try:
        bf  = int(stat.get("battersFaced") or stat.get("atBats") or 0)
        if bf < 100:
            return None
        h   = int(stat.get("hits") or 0)
        hr  = int(stat.get("homeRuns") or 0)
        bb  = int(stat.get("baseOnBalls") or 0)
        hbp = int(stat.get("hitByPitch") or 0)
        so  = int(stat.get("strikeOuts") or 0)
        # Estimate doubles/triples from h and typical MLB ratios
        # (MLB Stats API does not expose hit-type splits on pitching side)
        # Use: ~51% of XBH are doubles, ~6% triples, ~43% HRs (of non-HR hits)
        non_hr_hits = max(0, h - hr)
        d_est = round(non_hr_hits * 0.12)    # ~12% of hits are doubles
        t_est = round(non_hr_hits * 0.012)   # ~1.2% triples
        singles = max(0, non_hr_hits - d_est - t_est)
        bb_total = bb + hbp
        other_out = max(0, bf - singles - d_est - t_est - hr - bb_total - so)
        rates = {
            "1B":  singles   / bf,
            "2B":  d_est     / bf,
            "3B":  t_est     / bf,
            "HR":  hr        / bf,
            "BB":  bb_total  / bf,
            "K":   so        / bf,
            "OUT": other_out / bf,
        }
        arr = np.array([max(0.0, rates[k]) for k in _OUTCOMES_7], dtype=np.float64)
        total = arr.sum()
        if total <= 0:
            return None
        arr = arr / total
        return {k: float(arr[i]) for i, k in enumerate(_OUTCOMES_7)}
    except Exception as exc:
        logger.debug("pitching_stats_to_rates failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Public rate fetchers
# ---------------------------------------------------------------------------

def get_pitcher_rates(player_id: int, season: int | None = None) -> dict[str, float]:
    """
    Return (7,) PA-outcome rate dict for a pitcher.
    Falls back to league average if data unavailable or sample too small.
    """
    import datetime
    if season is None:
        season = datetime.date.today().year
    stat = _get_player_season_stats(player_id, season, "pitching")
    rates = _pitching_stats_to_rates(stat)
    if rates is None:
        logger.debug("Pitcher %d: insufficient data, using league avg", player_id)
        return dict(zip(_OUTCOMES_7, _league_avg_array().tolist()))
    return rates


def get_batter_rates(player_id: int, season: int | None = None) -> dict[str, float]:
    """
    Return (7,) PA-outcome rate dict for a batter.
    Falls back to league average if data unavailable.
    """
    import datetime
    if season is None:
        season = datetime.date.today().year
    stat = _get_player_season_stats(player_id, season, "hitting")
    rates = _hitting_stats_to_rates(stat)
    if rates is None:
        logger.debug("Batter %d: insufficient data, using league avg", player_id)
        return dict(zip(_OUTCOMES_7, _league_avg_array().tolist()))
    return rates


# ---------------------------------------------------------------------------
# Log5 matchup adjustment (7-outcome version)
# ---------------------------------------------------------------------------

def _log5_7(batter: dict[str, float], pitcher: dict[str, float]) -> np.ndarray:
    """
    Multinomial log5: matchup_k ∝ batter_k × pitcher_k / league_k
    Returns normalised (7,) array.
    """
    lg = _league_avg_array()
    b  = np.array([batter.get(k, lg[i]) for i, k in enumerate(_OUTCOMES_7)], dtype=np.float64)
    p  = np.array([pitcher.get(k, lg[i]) for i, k in enumerate(_OUTCOMES_7)], dtype=np.float64)

    # Clip and normalise
    b  = np.maximum(b,  1e-9); b  /= b.sum()
    p  = np.maximum(p,  1e-9); p  /= p.sum()
    lg = np.maximum(lg, 1e-9); lg /= lg.sum()

    q = b * p / lg
    q = np.maximum(q, 0.0)
    total = q.sum()
    if total <= 0:
        return lg
    return q / total


# ---------------------------------------------------------------------------
# Per-game lineup builder
# ---------------------------------------------------------------------------

def _get_lineup_player_ids(game_pk: int, side: str) -> list[int]:
    """
    Try to get the batting order player IDs for {side} ('home'/'away').
    Returns an empty list if not available (pre-game or no lineup posted).
    """
    try:
        from src.ingestion.mlb_stats_api import _load_or_fetch  # noqa: PLC0415
        data = _load_or_fetch(
            cache_key=f"lineup_{game_pk}_{side}",
            endpoint=f"game/{game_pk}/lineups",
            params=None,
            force_refresh=False,
        )
        lineups = data.get("homePlayers" if side == "home" else "awayPlayers", [])
        ids = [int(p["id"]) for p in lineups if p.get("id")]
        return ids[:9]
    except Exception as exc:
        logger.debug("Lineup fetch failed game_pk=%d side=%s: %s", game_pk, side, exc)
        return []


def _get_probable_pitcher_id(game_pk: int, side: str) -> int | None:
    """Return the probable pitcher player_id for {side}, or None."""
    try:
        from src.ingestion.mlb_stats_api import _load_or_fetch  # noqa: PLC0415
        import datetime
        date_str = datetime.date.today().isoformat()
        raw = _load_or_fetch(
            cache_key=f"schedule_{date_str}",
            endpoint="schedule",
            params={
                "sportId": 1,
                "date": date_str,
                "hydrate": "team,probablePitcher",
            },
            force_refresh=False,
        )
        for date_entry in raw.get("dates", []):
            for g in date_entry.get("games", []):
                if int(g.get("gamePk", 0)) == game_pk:
                    teams = g.get("teams", {})
                    team = teams.get(side, {})
                    pp = team.get("probablePitcher")
                    if pp:
                        return int(pp.get("id", 0)) or None
    except Exception as exc:
        logger.debug("Probable pitcher fetch failed game_pk=%d side=%s: %s", game_pk, side, exc)
    return None


def get_game_pa_probs(game_pk: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build per-game (9×7) PA probability matrices for home and away.

    Strategy:
    1. Get probable pitchers for each side.
    2. Get pitcher season rates.
    3. Get lineup player IDs (or fall back to roster-average).
    4. For each batting slot, get batter season rates.
    5. Apply log5 matchup adjustment: batter vs opposing pitcher.
    6. Return (home_pa_probs, away_pa_probs) each shape (9,7).

    If a pitcher's data is unavailable, use league avg for that side.
    If a batter's data is unavailable, use league avg for that slot.
    """
    import datetime
    season = datetime.date.today().year

    # ── Get probable pitchers ─────────────────────────────────────────────
    home_pitcher_id = _get_probable_pitcher_id(game_pk, "home")
    away_pitcher_id = _get_probable_pitcher_id(game_pk, "away")

    home_pitcher_rates = (
        get_pitcher_rates(home_pitcher_id, season)
        if home_pitcher_id else dict(zip(_OUTCOMES_7, _league_avg_array().tolist()))
    )
    away_pitcher_rates = (
        get_pitcher_rates(away_pitcher_id, season)
        if away_pitcher_id else dict(zip(_OUTCOMES_7, _league_avg_array().tolist()))
    )

    # ── Get lineup player IDs ─────────────────────────────────────────────
    home_batter_ids = _get_lineup_player_ids(game_pk, "home")
    away_batter_ids = _get_lineup_player_ids(game_pk, "away")

    # ── Fallback: team roster average (top 9 by games played) ────────────
    if not home_batter_ids:
        home_batter_ids = _get_team_batter_ids(game_pk, "home", season)
    if not away_batter_ids:
        away_batter_ids = _get_team_batter_ids(game_pk, "away", season)

    # ── Build PA prob matrices ─────────────────────────────────────────────
    def _build_matrix(batter_ids: list[int], opp_pitcher_rates: dict) -> np.ndarray:
        rows = []
        for i in range(9):
            if i < len(batter_ids):
                b_rates = get_batter_rates(batter_ids[i], season)
            else:
                b_rates = dict(zip(_OUTCOMES_7, _league_avg_array().tolist()))
            row = _log5_7(b_rates, opp_pitcher_rates)
            rows.append(row)
        mat = np.array(rows, dtype=np.float64)  # (9, 7)
        # Normalise each row
        sums = mat.sum(axis=1, keepdims=True)
        sums = np.where(sums <= 0, 1.0, sums)
        return mat / sums

    # Home batters face the AWAY pitcher; away batters face the HOME pitcher
    home_pa = _build_matrix(home_batter_ids, away_pitcher_rates)
    away_pa = _build_matrix(away_batter_ids, home_pitcher_rates)

    return home_pa, away_pa


def _get_team_batter_ids(game_pk: int, side: str, season: int) -> list[int]:
    """
    Fallback: get the team ID for this game's {side} and pull the active
    roster, returning the first 9 position-player IDs.
    """
    try:
        from src.ingestion.mlb_stats_api import _load_or_fetch, get_roster  # noqa: PLC0415
        import datetime
        date_str = datetime.date.today().isoformat()
        raw = _load_or_fetch(
            cache_key=f"schedule_{date_str}",
            endpoint="schedule",
            params={"sportId": 1, "date": date_str, "hydrate": "team"},
            force_refresh=False,
        )
        team_id: int | None = None
        for date_entry in raw.get("dates", []):
            for g in date_entry.get("games", []):
                if int(g.get("gamePk", 0)) == game_pk:
                    teams = g.get("teams", {})
                    team_id = teams.get(side, {}).get("team", {}).get("id")
                    break
        if team_id is None:
            return []

        roster = get_roster(team_id)
        # Filter to non-pitchers
        batters = [
            r["player_id"] for r in roster
            if r.get("position") not in ("P", "SP", "RP", "CP")
            and r.get("player_id")
        ]
        return batters[:9]
    except Exception as exc:
        logger.debug("Team batter IDs fallback failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Convenience: build league-average (9,7) matrix (fast path / fallback)
# ---------------------------------------------------------------------------

def league_avg_pa_matrix() -> np.ndarray:
    """Return (9, 7) league-average PA probability matrix."""
    return _lg_tile()
