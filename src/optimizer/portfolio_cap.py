"""
optimizer.portfolio_cap
=======================
Hard caps and risk guardrails applied after Kelly sizing.

Phase 4 will implement:
- Per-bet maximum exposure cap (e.g., 3% bankroll)
- Correlated game cap (same-day same-team exposure limit)
- Daily total risk cap (e.g., 15% bankroll)
- Drawdown circuit-breaker (halt betting after N% drawdown)
"""

# TODO (Phase 4): implement apply_caps(sized_bets_df, bankroll, config)
