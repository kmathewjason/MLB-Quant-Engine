"""
ingestion.mlb_stats_api
=======================
Thin wrapper around the MLB Stats API (statsapi.mlb.com/api/v1).

Phase 1 will implement:
- Schedule fetching (today's games, series info)
- Boxscore parsing (linescore, batting/pitching splits)
- Roster + probable-pitcher endpoints
"""

# TODO (Phase 1): implement get_schedule(date), get_boxscore(game_pk), get_roster(team_id)
