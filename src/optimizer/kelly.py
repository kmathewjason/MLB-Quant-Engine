"""
optimizer.kelly
===============
Kelly criterion bet sizing — single-bet and covariance-adjusted portfolio Kelly.

Math (Phase 4):
    Single bet:  f* = (b*p - q) / b
        b = decimal_odds - 1, p = model win prob, q = 1 - p

    Fractional Kelly: f = f* / divisor  (default divisor = 4)

    Portfolio (multi-bet covariance-adjusted):
        Maximise E[log(W)] subject to correlation constraints
        Solved via quadratic programming (scipy.optimize)

Phase 4 will implement:
- kelly_fraction(model_prob, fair_odds, fractional_divisor)
- portfolio_kelly(bets_df, cov_matrix, bankroll)
"""

# TODO (Phase 4): implement kelly_fraction(p, b, divisor), portfolio_kelly(bets, cov, bankroll)
