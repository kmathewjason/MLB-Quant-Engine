"""
backtest.clv_tracker
====================
Closing-line value (CLV) tracker — the gold-standard edge validation metric.

Theory (Phase 5):
    CLV = log(closing_odds / opening_odds_taken)
    Positive mean CLV → model is finding mispriced lines before the market corrects.

Phase 5 will implement:
- Bet record ingestion (open odds, close odds, result)
- CLV per-bet and rolling CLV series
- Statistical significance test (t-test vs zero)
- CLV decomposition by market type / sport / book
"""

# TODO (Phase 5): implement record_bet(bet_id, open_odds, close_odds), compute_clv_summary()
