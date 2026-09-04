"""
optimizer.vig_removal
=====================
Removes bookmaker vig (overround) to extract fair implied probabilities.

Methods (Phase 4):
    Power method:   p_fair_i = p_raw_i^k  where k solves Σ p_fair_i = 1
    Additive:       p_fair_i = p_raw_i - vig/n
    Shin model:     accounts for inside-trader probability distortion

Phase 4 will implement:
- power_devig(odds_list) → fair_probs
- shin_devig(odds_list) → fair_probs
- vig_percentage(odds_list) → float
"""

# TODO (Phase 4): implement power_devig(american_odds_list), shin_devig(american_odds_list)
