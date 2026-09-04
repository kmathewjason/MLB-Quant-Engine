"""
features.matchup_features
=========================
Computes batter-vs-pitcher matchup probabilities using log5 / odds-ratio method.

Math (Phase 2):
    P(event | batter, pitcher) = (b * p / lg) / (b*p/lg + (1-b)*(1-p)/(1-lg))

where b = batter rate, p = pitcher rate, lg = league average rate.

Phase 2 will implement:
- log5 for K%, BB%, HR%, BABIP
- Bayesian-shrunk inputs (see bayesian_shrinkage.py)
- Historical head-to-head overlay
"""

# TODO (Phase 2): implement log5_matchup(batter_rate, pitcher_rate, lg_rate)
