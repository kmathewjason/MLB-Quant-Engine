"""
ingestion.live_odds
===================
Pulls pre-game and live moneyline / totals / run-line odds from The-Odds-API.

Phase 1 will implement:
- /sports/baseball_mlb/odds endpoint wrapper
- American ↔ decimal ↔ implied-probability converters
- Snapshot persistence to Parquet for CLV tracking
"""

# TODO (Phase 1): implement get_odds(market, regions), snapshot_odds(date)
