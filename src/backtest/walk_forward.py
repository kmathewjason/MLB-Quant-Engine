"""
backtest.walk_forward
=====================
Expanding-window walk-forward validation harness.

How it works
------------
Given a chronologically sorted dataset of (game_date, features X, labels y):

1. Start with an initial training window containing all rows before
   date T_start.
2. Fit the model on [0, T].
3. Predict on rows in (T, T + step_days].
4. Roll T forward by step_days and repeat until the end of the dataset.

This guarantees: every prediction is made on a game the model has NEVER
seen in training.  No information from the future leaks into any training
fold.

                  ┌────────────────────┬──────────┐
Fold 1:           │  TRAIN (expanding) │ PREDICT  │
                  ├────────────────────┴──────────┤
Fold 2:           │  TRAIN (expanding) │ PREDICT  │
                  ├─────────────────────┴──────────┤
                  ...

Retrain flag
------------
By default, the model is re-fitted from scratch on every fold.  Pass
retrain=False to skip re-fitting (use when the model is already warm and
incremental updating is not yet supported — useful for a fast dry-run).

Output: WalkForwardResult
-------------------------
  predictions_df : DataFrame with columns:
                   game_date, fold, X (as flat columns), y_true,
                   p_0 .. p_{K-1}  (predicted class probabilities),
                   y_pred  (argmax class)
  fold_metrics   : list of per-fold dicts  (log_loss, accuracy, brier, n)
  summary        : aggregate metrics dict

Public API
----------
WalkForwardResult                             — dataclass
WalkForwardConfig                             — dataclass (hyper-parameters)
walk_forward_validate(X, y, dates, model_factory, config)  -> WalkForwardResult
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardConfig:
    """
    Parameters controlling the walk-forward harness.

    Attributes
    ----------
    initial_train_days : minimum calendar days in the first training window.
                         (rows before this point in the sorted data are used
                          as the initial training set)
    step_days          : calendar days in each test fold window.
    min_train_samples  : minimum number of training rows required to fit;
                         folds with fewer are skipped.
    min_test_samples   : minimum number of test rows required to evaluate;
                         folds with fewer are skipped.
    retrain            : if True (default), refit the model on each fold.
    n_jobs             : reserved for future parallel fold execution.
    """
    initial_train_days: int = 90
    step_days: int = 7
    min_train_samples: int = 500
    min_test_samples: int = 10
    retrain: bool = True
    n_jobs: int = 1


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardResult:
    """
    All out-of-sample predictions and per-fold metrics from a walk-forward run.

    Attributes
    ----------
    predictions_df : one row per out-of-sample PA/game, with columns:
                     - row_idx    : original row index in the input arrays
                     - game_date  : date string or date-like
                     - fold       : fold number (0-based)
                     - y_true     : true label
                     - y_pred     : predicted label (argmax)
                     - p_0..p_{K-1}: predicted probability for each class
                     - loss       : per-row cross-entropy
    fold_metrics   : list of per-fold dicts with keys:
                     fold, date_start, date_end, n_train, n_test,
                     log_loss, accuracy, brier
    summary        : aggregate dict over all folds
    n_folds        : number of completed folds
    """
    predictions_df: pd.DataFrame
    fold_metrics: list[dict]
    summary: dict
    n_folds: int

    def oos_log_loss(self) -> float:
        """Full out-of-sample log-loss across all folds."""
        return float(self.predictions_df["loss"].mean())

    def oos_accuracy(self) -> float:
        """Full out-of-sample top-1 accuracy across all folds."""
        df = self.predictions_df
        return float((df["y_pred"] == df["y_true"]).mean())


# ---------------------------------------------------------------------------
# Per-row metrics helpers
# ---------------------------------------------------------------------------

def _row_log_loss(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-row cross-entropy: -log P(true class). Shape (N,)."""
    n = len(y)
    p_true = probs[np.arange(n), y]
    return -np.log(np.clip(p_true, 1e-12, 1.0))


def _row_brier(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Per-row multi-class Brier score: Σ_k (p_k - 1{y==k})².  Shape (N,).
    """
    n, k = probs.shape
    targets = np.zeros_like(probs)
    targets[np.arange(n), y] = 1.0
    return float(np.mean(((probs - targets) ** 2).sum(axis=1)))


def _fold_metrics(
    probs: np.ndarray,
    y: np.ndarray,
    fold: int,
    date_start: str,
    date_end: str,
    n_train: int,
) -> dict:
    losses = _row_log_loss(probs, y)
    preds  = probs.argmax(axis=1)
    n, k   = probs.shape
    targets = np.zeros_like(probs)
    targets[np.arange(n), y] = 1.0
    brier = float(np.mean(((probs - targets) ** 2).sum(axis=1)))
    return {
        "fold":       fold,
        "date_start": date_start,
        "date_end":   date_end,
        "n_train":    n_train,
        "n_test":     n,
        "log_loss":   float(losses.mean()),
        "accuracy":   float((preds == y).mean()),
        "brier":      brier,
    }


# ---------------------------------------------------------------------------
# Walk-forward engine
# ---------------------------------------------------------------------------

def walk_forward_validate(
    X: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray | pd.Series,
    model_factory: Callable[[], Any],
    config: WalkForwardConfig | None = None,
) -> WalkForwardResult:
    """
    Run expanding-window walk-forward validation.

    Parameters
    ----------
    X             : (N, F) feature matrix, rows in chronological order
    y             : (N,) integer class labels
    dates         : (N,) array of date-like objects (date, datetime, or str)
                    MUST be aligned with X and y (same row order)
    model_factory : callable that returns a fresh, unfitted model with a
                    .fit(X, y) method and a .predict_proba(X) method.
                    Called once per fold when config.retrain=True.
    config        : WalkForwardConfig; defaults are used if None.

    Returns
    -------
    WalkForwardResult

    Design invariant
    ----------------
    For every row r in predictions_df, dates[r] > max(dates[tr_idx]) for that
    fold's training set.  No future data can appear in any training fold.
    """
    if config is None:
        config = WalkForwardConfig()

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)
    dates_arr = pd.to_datetime(dates).values   # numpy datetime64
    n = len(y)

    if n == 0:
        raise ValueError("Empty dataset passed to walk_forward_validate.")
    if len(dates_arr) != n:
        raise ValueError(f"dates length {len(dates_arr)} != X length {n}.")

    # Sort rows chronologically (ensures expanding windows are contiguous)
    sort_idx = np.argsort(dates_arr, kind="stable")
    X_s = X[sort_idx]
    y_s = y[sort_idx]
    d_s = dates_arr[sort_idx]

    # Initial training cutoff date
    d_min = d_s[0]
    cutoff = d_min + np.timedelta64(config.initial_train_days, "D")

    # Collect all output rows
    all_preds: list[dict] = []
    fold_metrics_list: list[dict] = []
    fold_idx = 0
    current_model = None

    while True:
        # Training mask: all rows with date < cutoff
        tr_mask = d_s < cutoff
        te_mask = (d_s >= cutoff) & (d_s < cutoff + np.timedelta64(config.step_days, "D"))

        n_train = int(tr_mask.sum())
        n_test  = int(te_mask.sum())

        if n_test == 0:
            # No more test data; exit
            break

        if n_train < config.min_train_samples:
            logger.debug(
                "Fold %d: insufficient training samples (%d < %d), skipping.",
                fold_idx, n_train, config.min_train_samples,
            )
            cutoff += np.timedelta64(config.step_days, "D")
            continue

        if n_test < config.min_test_samples:
            logger.debug(
                "Fold %d: insufficient test samples (%d < %d), skipping.",
                fold_idx, n_test, config.min_test_samples,
            )
            cutoff += np.timedelta64(config.step_days, "D")
            continue

        X_tr, y_tr = X_s[tr_mask], y_s[tr_mask]
        X_te, y_te = X_s[te_mask], y_s[te_mask]
        te_orig_idx = sort_idx[te_mask]
        te_dates    = d_s[te_mask]

        date_start = str(np.datetime_as_string(te_dates[0],  unit="D"))
        date_end   = str(np.datetime_as_string(te_dates[-1], unit="D"))

        # ── Fit ───────────────────────────────────────────────────────
        if config.retrain or current_model is None:
            current_model = model_factory()
            logger.info(
                "Fold %d | train=%d | test=%d | %s → %s",
                fold_idx, n_train, n_test, date_start, date_end,
            )
            current_model.fit(X_tr, y_tr)

        # ── Predict ───────────────────────────────────────────────────
        probs = np.asarray(current_model.predict_proba(X_te), dtype=np.float64)
        probs = np.clip(probs, 1e-12, 1.0)
        probs /= probs.sum(axis=1, keepdims=True)

        losses = _row_log_loss(probs, y_te)
        preds  = probs.argmax(axis=1)
        n_classes = probs.shape[1]

        for i in range(n_test):
            row: dict = {
                "row_idx":   int(te_orig_idx[i]),
                "game_date": str(np.datetime_as_string(te_dates[i], unit="D")),
                "fold":      fold_idx,
                "y_true":    int(y_te[i]),
                "y_pred":    int(preds[i]),
                "loss":      float(losses[i]),
            }
            for k in range(n_classes):
                row[f"p_{k}"] = float(probs[i, k])
            all_preds.append(row)

        fold_metrics_list.append(
            _fold_metrics(probs, y_te, fold_idx, date_start, date_end, n_train)
        )

        fold_idx += 1
        cutoff += np.timedelta64(config.step_days, "D")

    if not all_preds:
        raise RuntimeError(
            "walk_forward_validate produced zero predictions. "
            "Check that initial_train_days is not larger than the dataset range, "
            "and that min_train_samples is reachable."
        )

    predictions_df = pd.DataFrame(all_preds)
    summary = _aggregate_summary(fold_metrics_list, predictions_df)

    logger.info(
        "Walk-forward complete: %d folds, %d OOS predictions, OOS log-loss=%.4f",
        fold_idx, len(predictions_df), summary["oos_log_loss"],
    )

    return WalkForwardResult(
        predictions_df=predictions_df,
        fold_metrics=fold_metrics_list,
        summary=summary,
        n_folds=fold_idx,
    )


def _aggregate_summary(fold_metrics: list[dict], predictions_df: pd.DataFrame) -> dict:
    """Aggregate per-fold metrics into a single summary dict."""
    df = predictions_df
    all_losses = df["loss"].values
    all_correct = (df["y_pred"] == df["y_true"]).values

    log_losses  = [f["log_loss"] for f in fold_metrics]
    accuracies  = [f["accuracy"] for f in fold_metrics]
    briers      = [f["brier"]    for f in fold_metrics]

    return {
        "n_folds":            len(fold_metrics),
        "n_oos_predictions":  len(df),
        "oos_log_loss":       float(all_losses.mean()),
        "oos_accuracy":       float(all_correct.mean()),
        "mean_fold_log_loss": float(np.mean(log_losses)),
        "std_fold_log_loss":  float(np.std(log_losses)),
        "mean_fold_accuracy": float(np.mean(accuracies)),
        "mean_fold_brier":    float(np.mean(briers)),
    }
