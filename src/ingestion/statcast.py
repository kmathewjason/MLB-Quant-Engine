"""
ingestion.statcast
==================
Pulls pitch-level Statcast data from Baseball Savant via pybaseball.

Phase 1 will implement:
- Date-range bulk pulls with incremental caching to Parquet
- Column normalisation and dtypes
- Retry / back-off wrapper around pybaseball rate limits
"""

# TODO (Phase 1): implement pull_statcast(start_date, end_date, cache_dir)
