"""
ingestion.retrosheet
====================
Reads Retrosheet data from the local ``RETROSHEET_DATA_DIR`` directory
(set in .env) and exposes it as tidy pandas DataFrames.

Data formats handled
--------------------
gamelogs/glYYYY.txt  — Retrosheet Game Log files (one row per game,
                       161 fixed-position columns per Retrosheet spec)
events/YYYYTTT.EVN  — Retrosheet Event files (play-by-play; format:
  .EVN = NL home, .EVA = AL home)
biodata/biofile.csv  — Player biographical file

Public API
----------
load_gamelogs(seasons, data_dir)
    -> pd.DataFrame   one row per game, key columns renamed to snake_case

load_event_file(year, team, data_dir)
    -> pd.DataFrame   raw rows from a single .EVN/.EVA file

parse_play_by_play(year_or_years, data_dir)
    -> pd.DataFrame   tidy play-by-play with inning/base/outs columns

player_season_rates(play_df)
    -> pd.DataFrame   PA-level rate stats (K%, BB%, 1B%, 2B%, 3B%, HR%)
                      per player-season — ready for bayesian_shrinkage

load_bio(data_dir)
    -> pd.DataFrame   player bio with retrosheet_id, full_name, bats, throws

Environment
-----------
RETROSHEET_DATA_DIR  — root of the alldata/ tree  (required unless data_dir
                       is passed explicitly to each function)
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Locate data root
# ---------------------------------------------------------------------------

def _data_root(data_dir: str | Path | None) -> Path:
    """Resolve the Retrosheet data root directory."""
    if data_dir is not None:
        p = Path(data_dir)
    else:
        env = os.getenv("RETROSHEET_DATA_DIR", "")
        if not env:
            raise EnvironmentError(
                "RETROSHEET_DATA_DIR is not set in .env and data_dir was not provided."
            )
        p = Path(env)
    if not p.exists():
        raise FileNotFoundError(f"Retrosheet data directory not found: {p}")
    return p


# ---------------------------------------------------------------------------
# Game Log loader
# ---------------------------------------------------------------------------

# Retrosheet Game Log column positions (0-based) and names.
# Full 161-column spec: https://www.retrosheet.org/gamelogs/glfields.txt
# We keep only the analytically useful subset.
_GAMELOG_COLS: list[tuple[int, str]] = [
    (0,  "date"),
    (1,  "game_number"),       # 0 = single game, 1/2 = DH
    (2,  "day_of_week"),
    (3,  "visitor_team"),
    (4,  "visitor_league"),
    (5,  "visitor_game_number"),
    (6,  "home_team"),
    (7,  "home_league"),
    (8,  "home_game_number"),
    (9,  "visitor_score"),
    (10, "home_score"),
    (11, "innings"),           # 9 for regulation; actual if different
    (12, "day_night"),         # D/N/blank
    (13, "completion_info"),
    (16, "park_id"),
    (17, "attendance"),
    (18, "duration_min"),
    (19, "visitor_line_score"),
    (20, "home_line_score"),
    (21, "visitor_ab"),
    (22, "visitor_hits"),
    (23, "visitor_doubles"),
    (24, "visitor_triples"),
    (25, "visitor_hr"),
    (26, "visitor_rbi"),
    (27, "visitor_sh"),
    (28, "visitor_sf"),
    (29, "visitor_hbp"),
    (30, "visitor_bb"),
    (31, "visitor_ibb"),
    (32, "visitor_k"),
    (33, "visitor_sb"),
    (34, "visitor_cs"),
    (35, "visitor_gdp"),
    (36, "visitor_ci"),
    (37, "visitor_lob"),
    (38, "visitor_pitchers"),
    (39, "visitor_er"),
    (40, "visitor_ter"),
    (41, "visitor_wp"),
    (42, "visitor_bk"),
    (43, "home_ab"),
    (44, "home_hits"),
    (45, "home_doubles"),
    (46, "home_triples"),
    (47, "home_hr"),
    (48, "home_rbi"),
    (49, "home_sh"),
    (50, "home_sf"),
    (51, "home_hbp"),
    (52, "home_bb"),
    (53, "home_ibb"),
    (54, "home_k"),
    (55, "home_sb"),
    (56, "home_cs"),
    (57, "home_gdp"),
    (58, "home_ci"),
    (59, "home_lob"),
    (60, "home_pitchers"),
    (61, "home_er"),
    (62, "home_ter"),
    (63, "home_wp"),
    (64, "home_bk"),
    (65, "visitor_po"),
    (66, "visitor_assists"),
    (67, "visitor_errors"),
    (68, "visitor_pb"),
    (69, "visitor_dp"),
    (70, "visitor_tp"),
    (71, "home_po"),
    (72, "home_assists"),
    (73, "home_errors"),
    (74, "home_pb"),
    (75, "home_dp"),
    (76, "home_tp"),
    (77, "ump_hp_id"),
    (78, "ump_hp_name"),
    (79, "ump_1b_id"),
    (80, "ump_1b_name"),
    (107, "visitor_manager_id"),
    (108, "visitor_manager_name"),
    (109, "home_manager_id"),
    (110, "home_manager_name"),
    (111, "winning_pitcher_id"),
    (112, "winning_pitcher_name"),
    (113, "losing_pitcher_id"),
    (114, "losing_pitcher_name"),
    (115, "save_pitcher_id"),
    (116, "save_pitcher_name"),
    (117, "gw_rbi_id"),
    (118, "gw_rbi_name"),
    (121, "visitor_sp_id"),
    (122, "visitor_sp_name"),
    (123, "home_sp_id"),
    (124, "home_sp_name"),
]

_GL_IDX  = [c[0] for c in _GAMELOG_COLS]
_GL_NAMES = [c[1] for c in _GAMELOG_COLS]


def load_gamelogs(
    seasons: int | Sequence[int] | None = None,
    data_dir: str | Path | None = None,
) -> pd.DataFrame:
    """
    Load Retrosheet Game Log files and return a tidy DataFrame.

    Parameters
    ----------
    seasons  : single year, list of years, or None (all available).
    data_dir : root of alldata/ tree; defaults to RETROSHEET_DATA_DIR env var.

    Returns
    -------
    DataFrame with one row per game.  Key derived columns added:
        game_date (datetime), total_runs, home_win (bool), run_diff
    """
    root = _data_root(data_dir)
    gl_dir = root / "gamelogs"
    if not gl_dir.exists():
        raise FileNotFoundError(f"Gamelogs directory not found: {gl_dir}")

    if seasons is None:
        files = sorted(gl_dir.glob("gl[0-9][0-9][0-9][0-9].txt"))
    else:
        if isinstance(seasons, int):
            seasons = [seasons]
        files = []
        for yr in seasons:
            p = gl_dir / f"gl{yr}.txt"
            if p.exists():
                files.append(p)
            else:
                logger.warning("Gamelog not found for season %d: %s", yr, p)

    if not files:
        raise FileNotFoundError(f"No gamelog files found in {gl_dir}")

    dfs = []
    for f in files:
        try:
            # Gamelogs are comma-separated with optional quoting
            raw = pd.read_csv(
                f,
                header=None,
                dtype=str,
                keep_default_na=False,
            )
            # Extract only the columns we care about (handle files with
            # fewer columns gracefully — older seasons have 161 cols, some
            # play-off files have fewer)
            max_col = raw.shape[1] - 1
            valid_pairs = [(i, n) for i, n in _GAMELOG_COLS if i <= max_col]
            idx   = [p[0] for p in valid_pairs]
            names = [p[1] for p in valid_pairs]

            df = raw.iloc[:, idx].copy()
            df.columns = names
            dfs.append(df)
            logger.debug("Loaded %d games from %s", len(df), f.name)
        except Exception as exc:
            logger.warning("Failed to load %s: %s", f.name, exc)

    if not dfs:
        raise RuntimeError("No gamelog data loaded — check RETROSHEET_DATA_DIR")

    combined = pd.concat(dfs, ignore_index=True)

    # Type coercions
    combined["game_date"]    = pd.to_datetime(combined["date"], format="%Y%m%d", errors="coerce")
    combined["season"]       = combined["game_date"].dt.year

    for col in ("visitor_score", "home_score", "innings", "attendance",
                "duration_min", "visitor_ab", "visitor_hits", "visitor_hr",
                "visitor_bb", "visitor_k", "visitor_hbp",
                "home_ab", "home_hits", "home_hr", "home_bb", "home_k",
                "home_hbp"):
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    combined["total_runs"]   = combined["visitor_score"] + combined["home_score"]
    combined["home_win"]     = combined["home_score"] > combined["visitor_score"]
    combined["run_diff"]     = combined["home_score"] - combined["visitor_score"]

    combined.sort_values("game_date", inplace=True)
    combined.reset_index(drop=True, inplace=True)

    logger.info(
        "Loaded %d games across %d seasons",
        len(combined),
        combined["season"].nunique(),
    )
    return combined


# ---------------------------------------------------------------------------
# Event file loader
# ---------------------------------------------------------------------------

# Event file record types we care about:
#   id      — game identifier
#   version — file version
#   info    — game-level metadata (visteam, hometeam, date, site, etc.)
#   start   — starting lineup
#   sub     — substitution
#   play    — play event (the core record)
#   data    — earned-run data per pitcher
#   com     — comment (ignored)

def load_event_file(
    year: int,
    team: str,
    data_dir: str | Path | None = None,
) -> pd.DataFrame:
    """
    Load all raw records from a single Retrosheet event file.

    Parameters
    ----------
    year    : season (e.g. 2024)
    team    : Retrosheet team code (e.g. "NYA", "LAN")
    data_dir: root of alldata/; defaults to RETROSHEET_DATA_DIR

    Returns
    -------
    DataFrame with columns: game_id, record_type, fields (list→joined string)
    Each row is one line of the event file.
    """
    root = _data_root(data_dir)
    ev_dir = root / "events"

    # Try both home-park suffixes
    path: Path | None = None
    for suffix in ("EVA", "EVN"):
        candidate = ev_dir / f"{year}{team}.{suffix}"
        if candidate.exists():
            path = candidate
            break

    if path is None:
        raise FileNotFoundError(
            f"Event file not found for {year} {team} in {ev_dir}"
        )

    rows = []
    current_game_id = ""
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n\r")
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            rec_type = parts[0].lower()
            if rec_type == "id":
                current_game_id = parts[1] if len(parts) > 1 else ""
            rows.append({
                "game_id":     current_game_id,
                "record_type": rec_type,
                "fields":      ",".join(parts[1:]),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Play-by-play parser
# ---------------------------------------------------------------------------

# Simplified PA outcome extraction from Retrosheet play codes.
# Full spec: https://www.retrosheet.org/eventfile.htm
# We map to the same 7-class scheme used in pa_outcome_model.py:
#   K, BB_HBP, 1B, 2B, 3B, HR, out_in_play

_PLAY_RE = re.compile(
    r"^(?P<advance>[^.]*)"   # base/advance codes
)

# Outcome classifiers — applied in priority order
def _classify_play(play_str: str) -> str | None:
    """
    Map a Retrosheet play code to one of our 7 PA outcome labels.
    Returns None for non-PA events (balks, stolen bases, etc.).
    """
    # Strip modifiers (everything after '/' or '.')
    core = re.split(r"[/.]", play_str)[0].upper()

    # Strikeout
    if core.startswith("K"):
        return "K"
    # Walk / Intent walk / HBP
    if core.startswith(("W", "IW", "I", "HP")):
        return "BB_HBP"
    # Home run
    if core.startswith("HR") or core == "H":
        return "HR"
    # Triple
    if core.startswith("T"):
        return "3B"
    # Double
    if core.startswith("D"):
        return "2B"
    # Single
    if core.startswith("S"):
        return "1B"
    # Error — treated as out_in_play for outcome model purposes
    if core.startswith("E"):
        return "out_in_play"
    # Fielder's choice
    if core.startswith("FC"):
        return "out_in_play"
    # Standard outs (numeric defensive codes)
    if core and core[0].isdigit():
        return "out_in_play"
    # Non-PA events: stolen base, caught stealing, balk, wild pitch, etc.
    return None


def parse_play_by_play(
    year_or_years: int | Sequence[int],
    data_dir: str | Path | None = None,
    teams: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Parse Retrosheet event files into a tidy play-by-play DataFrame.

    Parameters
    ----------
    year_or_years : season or list of seasons to parse.
    data_dir      : root of alldata/; defaults to RETROSHEET_DATA_DIR.
    teams         : if provided, only load event files for these team codes.
                    Defaults to all teams found for each season.

    Returns
    -------
    DataFrame with columns:
        game_id, season, game_date, home_team, visitor_team,
        inning, half (0=top/away, 1=bottom/home), outs_before_play,
        batter_id, pitcher_id,
        play_str, pa_outcome (K/BB_HBP/1B/2B/3B/HR/out_in_play or NaN)
    """
    root = _data_root(data_dir)
    ev_dir = root / "events"

    if isinstance(year_or_years, int):
        years = [year_or_years]
    else:
        years = list(year_or_years)

    all_rows: list[dict] = []

    for year in years:
        # Discover available event files for this year
        if teams is not None:
            candidates = []
            for t in teams:
                for suf in ("EVA", "EVN"):
                    p = ev_dir / f"{year}{t}.{suf}"
                    if p.exists():
                        candidates.append(p)
        else:
            candidates = list(ev_dir.glob(f"{year}???.EV[AN]"))

        if not candidates:
            logger.warning("No event files found for season %d", year)
            continue

        for ev_path in sorted(candidates):
            try:
                rows = _parse_one_event_file(ev_path, year)
                all_rows.extend(rows)
                logger.debug("Parsed %d PA records from %s", len(rows), ev_path.name)
            except Exception as exc:
                logger.warning("Failed to parse %s: %s", ev_path.name, exc)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["season"] = df["season"].astype("int16")
    df["inning"] = df["inning"].astype("int8")
    df["half"]   = df["half"].astype("int8")

    logger.info(
        "Parsed %d play records across %d season(s)",
        len(df), df["season"].nunique(),
    )
    return df


def _parse_one_event_file(path: Path, year: int) -> list[dict]:
    """Parse a single .EVN/.EVA file into a list of play dicts."""
    rows: list[dict] = []

    game_id    = ""
    home_team  = ""
    away_team  = ""
    game_date  = ""
    inning     = 0
    half       = 0   # 0=top (away bats), 1=bottom (home bats)
    outs       = 0

    # Active lineup: {slot: player_id} for home and away
    lineup_home: dict[int, str] = {}
    lineup_away: dict[int, str] = {}
    current_pitcher_home = ""
    current_pitcher_away = ""

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n\r")
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            rec   = parts[0].lower()

            if rec == "id":
                game_id   = parts[1] if len(parts) > 1 else ""
                # Reset game state
                inning    = 1
                half      = 0
                outs      = 0
                lineup_home.clear()
                lineup_away.clear()
                current_pitcher_home = ""
                current_pitcher_away = ""

            elif rec == "info":
                if len(parts) >= 3:
                    key, val = parts[1].lower(), parts[2]
                    if key == "hometeam":
                        home_team = val
                    elif key == "visteam":
                        away_team = val
                    elif key == "date":
                        game_date = val

            elif rec == "start":
                # start,player_id,name,home_or_away(0=away,1=home),batting_order,fielding_pos
                if len(parts) >= 6:
                    pid    = parts[1]
                    side   = int(parts[3])  # 0=away, 1=home
                    slot   = int(parts[4])  # 1–9
                    fpos   = int(parts[5])  # 1=P
                    if side == 1:
                        lineup_home[slot] = pid
                        if fpos == 1:
                            current_pitcher_home = pid
                    else:
                        lineup_away[slot] = pid
                        if fpos == 1:
                            current_pitcher_away = pid

            elif rec == "sub":
                # sub,player_id,name,home_or_away,batting_order,fielding_pos
                if len(parts) >= 6:
                    pid    = parts[1]
                    side   = int(parts[3])
                    slot   = int(parts[4])
                    fpos   = int(parts[5])
                    if side == 1:
                        lineup_home[slot] = pid
                        if fpos == 1:
                            current_pitcher_home = pid
                    else:
                        lineup_away[slot] = pid
                        if fpos == 1:
                            current_pitcher_away = pid

            elif rec == "play":
                # play,inning,home_or_away,batter_id,count,pitches,play_str
                if len(parts) < 7:
                    continue
                try:
                    rec_inning = int(parts[1])
                    rec_half   = int(parts[2])   # 0=top(away), 1=bottom(home)
                    batter_id  = parts[3]
                    play_str   = parts[6]
                except (ValueError, IndexError):
                    continue

                # Update inning/half tracking
                if rec_inning != inning or rec_half != half:
                    inning = rec_inning
                    half   = rec_half
                    outs   = 0

                # Identify the pitcher
                if rec_half == 0:   # away batting → home pitcher pitching
                    pitcher_id = current_pitcher_home
                else:               # home batting → away pitcher pitching
                    pitcher_id = current_pitcher_away

                outcome = _classify_play(play_str)

                rows.append({
                    "game_id":        game_id,
                    "season":         year,
                    "game_date":      game_date,
                    "home_team":      home_team,
                    "visitor_team":   away_team,
                    "inning":         inning,
                    "half":           half,
                    "outs_before":    outs,
                    "batter_id":      batter_id,
                    "pitcher_id":     pitcher_id,
                    "play_str":       play_str,
                    "pa_outcome":     outcome,   # None if non-PA
                })

                # Advance outs counter
                if outcome is not None:
                    # PA occurred — count outs from play string
                    outs_on_play = _count_outs(play_str)
                    outs = (outs + outs_on_play) % 3

    return rows


def _count_outs(play_str: str) -> int:
    """
    Rough out-count from a Retrosheet play string.
    For strikeouts and standard outs: 1.  For double plays: 2.
    """
    core = re.split(r"[/.]", play_str)[0].upper()
    # Double play
    if "DP" in play_str.upper() or re.search(r"\d\d\d\(\d\)", play_str):
        return 2
    # Triple play
    if "TP" in play_str.upper():
        return 3
    # Strikeout
    if core.startswith("K"):
        # K+ = K + stolen base (no out counted yet for 3rd strike)
        if "+" in play_str:
            return 0   # dropped third strike, runner safe
        return 1
    # Walk, HBP, Hit — no out
    if core.startswith(("W", "IW", "I", "HP", "S", "D", "T", "HR", "H")):
        return 0
    # Error — batter safe
    if core.startswith("E"):
        return 0
    # Fielder's choice
    if core.startswith("FC"):
        return 1
    # Numeric = standard out
    if core and core[0].isdigit():
        return 1
    return 0


# ---------------------------------------------------------------------------
# Player rate aggregation
# ---------------------------------------------------------------------------

def player_season_rates(
    play_df: pd.DataFrame,
    min_pa: int = 30,
) -> pd.DataFrame:
    """
    Aggregate play-by-play into per-player-season PA outcome rates.

    Input *play_df* must come from parse_play_by_play() and have a
    non-null ``pa_outcome`` column.

    Returns
    -------
    DataFrame indexed by (batter_id, season) with columns:
        pa, k_rate, bb_hbp_rate, single_rate, double_rate,
        triple_rate, hr_rate, out_in_play_rate
    Only players with ≥ min_pa plate appearances are returned.
    """
    pa_df = play_df[play_df["pa_outcome"].notna()].copy()

    outcomes = ["K", "BB_HBP", "1B", "2B", "3B", "HR", "out_in_play"]
    for o in outcomes:
        pa_df[o] = (pa_df["pa_outcome"] == o).astype(int)

    grp = pa_df.groupby(["batter_id", "season"])
    agg = grp[outcomes].sum()
    agg["pa"] = grp.size()
    agg = agg.reset_index()
    agg = agg[agg["pa"] >= min_pa].copy()

    for o in outcomes:
        col = o.lower().replace("bb_hbp", "bb_hbp").replace("out_in_play", "out_in_play")
        agg[f"{col}_rate"] = agg[o] / agg["pa"]

    # Rename for clarity
    rename_map = {
        "K_rate":          "k_rate",
        "BB_HBP_rate":     "bb_hbp_rate",
        "1B_rate":         "single_rate",
        "2B_rate":         "double_rate",
        "3B_rate":         "triple_rate",
        "HR_rate":         "hr_rate",
        "out_in_play_rate":"out_in_play_rate",
    }
    agg.rename(columns=rename_map, inplace=True, errors="ignore")

    keep = ["batter_id", "season", "pa", "k_rate", "bb_hbp_rate",
            "single_rate", "double_rate", "triple_rate", "hr_rate",
            "out_in_play_rate"]
    return agg[[c for c in keep if c in agg.columns]].reset_index(drop=True)


def pitcher_season_rates(
    play_df: pd.DataFrame,
    min_bf: int = 30,
) -> pd.DataFrame:
    """
    Same as player_season_rates but from the pitcher's perspective.
    Returns rates of outcomes *allowed* per pitcher-season.
    """
    pa_df = play_df[play_df["pa_outcome"].notna()].copy()

    outcomes = ["K", "BB_HBP", "1B", "2B", "3B", "HR", "out_in_play"]
    for o in outcomes:
        pa_df[o] = (pa_df["pa_outcome"] == o).astype(int)

    grp = pa_df.groupby(["pitcher_id", "season"])
    agg = grp[outcomes].sum()
    agg["bf"] = grp.size()
    agg = agg.reset_index()
    agg = agg[agg["bf"] >= min_bf].copy()

    for o in outcomes:
        agg[f"{o.lower()}_rate"] = agg[o] / agg["bf"]

    rename_map = {
        "k_rate":          "pitcher_k_rate",
        "bb_hbp_rate":     "pitcher_bb_hbp_rate",
        "1b_rate":         "pitcher_1b_rate",
        "2b_rate":         "pitcher_2b_rate",
        "3b_rate":         "pitcher_3b_rate",
        "hr_rate":         "pitcher_hr_rate",
        "out_in_play_rate": "pitcher_out_in_play_rate",
    }
    agg.rename(columns=rename_map, inplace=True, errors="ignore")

    keep = ["pitcher_id", "season", "bf",
            "pitcher_k_rate", "pitcher_bb_hbp_rate", "pitcher_1b_rate",
            "pitcher_2b_rate", "pitcher_3b_rate", "pitcher_hr_rate",
            "pitcher_out_in_play_rate"]
    return agg[[c for c in keep if c in agg.columns]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Bio data
# ---------------------------------------------------------------------------

def load_bio(data_dir: str | Path | None = None) -> pd.DataFrame:
    """
    Load the Retrosheet biographical file (biodata/biofile.csv).

    Returns
    -------
    DataFrame with retrosheet_id, last, first, bats, throws,
    play_debut, play_lastgame columns.
    """
    root = _data_root(data_dir)
    path = root / "biodata" / "biofile.csv"
    if not path.exists():
        raise FileNotFoundError(f"Biofile not found: {path}")

    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    # Standardise column names
    df.columns = [c.strip().lower().replace(".", "_") for c in df.columns]
    rename = {
        "playerid": "retrosheet_id",
        "last":     "last_name",
        "first":    "first_name",
        "bats":     "bats",
        "throws":   "throws",
        "play_debut":    "debut",
        "play_lastgame": "last_game",
    }
    df.rename(columns=rename, inplace=True, errors="ignore")
    if "retrosheet_id" in df.columns:
        df["full_name"] = df.get("first_name", "") + " " + df.get("last_name", "")
    return df
