"""
features.bayesian_shrinkage
============================
Empirical Bayes regression-to-mean for small-sample rate stabilisation.

Math (Phase 2):
    shrunk_rate = (PA * observed + k * prior) / (PA + k)

where k is the stabilisation point (PA at which observed and true talent
have equal weight). Stabilisation points derived from Tango's work:
  - K%:   ~60 PA
  - BB%:  ~120 PA
  - HR/FB: ~300 PA
  - BABIP: ~820 PA

Phase 2 will implement:
- Per-stat shrinkage with configurable stabilisation points
- Hierarchical partial-pooling via pymc / scipy (optional)
"""

# TODO (Phase 2): implement shrink(observed, pa, prior, k_stabilisation)
