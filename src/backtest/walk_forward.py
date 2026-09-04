"""
backtest.walk_forward
=====================
Walk-forward cross-validation engine for temporal model evaluation.

Phase 5 will implement:
- Expanding / sliding window splitter (no future leakage)
- Per-fold: train model → predict → size bets → record P&L
- Aggregated metrics: ROI, Sharpe, max drawdown, hit rate
- Comparison against closing-line benchmark
"""

# TODO (Phase 5): implement WalkForwardBacktest(train_window, test_window).run(df, model, sizer)
