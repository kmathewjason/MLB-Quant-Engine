"""
backtest.run_report
===================
Generate a self-contained HTML backtest report from walk-forward and CLV results.

The report is a single-file HTML document containing:

  Section 1 — Walk-Forward Summary
    • Aggregate OOS metrics: log-loss, accuracy, Brier score, BSS
    • Per-fold table: date range, n_train, n_test, log_loss, accuracy, Brier
    • Fold-over-fold metric trend table

  Section 2 — Calibration
    • Per-class ECE, Brier decomposition (REL / RES / UNC)
    • Brier skill score vs climatology

  Section 3 — CLV Analysis
    • Summary table: mean CLV, t-stat, p-value, CLV-result correlation
    • Per-market breakdown
    • Rolling P&L in tabular form (first and last N rows)

  Section 4 — Dispersion Check (optional)
    • Underdispersed game count / % / ratio statistics

Usage
-----
python -m src.backtest.run_report --help

Or programmatically::

    from src.backtest.run_report import build_report, save_report
    html = build_report(wf_result, clv_df, cal_df)
    save_report(html, "docs/backtest_report.html")

Public API
----------
build_report(wf_result, clv_df, cal_df, dispersion_summary, title)  -> str  (HTML)
save_report(html, path)                                              -> Path
"""

from __future__ import annotations

import datetime
import html as _html
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def build_report(
    wf_result=None,
    clv_df: Optional[pd.DataFrame] = None,
    cal_df: Optional[pd.DataFrame] = None,
    dispersion_summary: Optional[dict] = None,
    title: str = "MLB Quant Engine — Backtest Report",
) -> str:
    """
    Render all available backtest results into a self-contained HTML string.

    Parameters
    ----------
    wf_result          : WalkForwardResult from walk_forward_validate(); optional.
    clv_df             : DataFrame from compute_clv(); optional.
    cal_df             : DataFrame from calibration_report(); optional.
    dispersion_summary : dict from dispersion_summary(); optional.
    title              : report title shown at the top of the HTML.

    Returns
    -------
    str — complete self-contained HTML document (no external resources).
    """
    sections: list[str] = []

    if wf_result is not None:
        sections.append(_section_walk_forward(wf_result))

    if cal_df is not None and not cal_df.empty:
        sections.append(_section_calibration(cal_df))

    if clv_df is not None and not clv_df.empty:
        sections.append(_section_clv(clv_df))

    if dispersion_summary is not None:
        sections.append(_section_dispersion(dispersion_summary))

    if not sections:
        sections.append("<p>No backtest data was provided.</p>")

    body = "\n".join(sections)
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return _wrap_html(title, body, timestamp)


def save_report(html: str, path: str | Path) -> Path:
    """
    Write the HTML report to *path*.  Creates parent directories as needed.

    Returns
    -------
    Path to the written file.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _section_walk_forward(wf_result) -> str:
    s = wf_result.summary
    fold_df = pd.DataFrame(wf_result.fold_metrics)

    summary_rows = [
        ("OOS predictions",   f"{s['n_oos_predictions']:,}"),
        ("Completed folds",   str(s["n_folds"])),
        ("OOS log-loss",      f"{s['oos_log_loss']:.4f}"),
        ("OOS accuracy",      f"{s['oos_accuracy']:.4f}"),
        ("Mean fold log-loss",f"{s['mean_fold_log_loss']:.4f}  (σ={s['std_fold_log_loss']:.4f})"),
        ("Mean fold Brier",   f"{s['mean_fold_brier']:.4f}"),
    ]

    summary_tbl = _kv_table(summary_rows, caption="Walk-Forward Aggregate Metrics")

    # Per-fold table
    fold_display = fold_df[["fold", "date_start", "date_end", "n_train", "n_test",
                             "log_loss", "accuracy", "brier"]].copy()
    fold_display["log_loss"] = fold_display["log_loss"].map("{:.4f}".format)
    fold_display["accuracy"] = fold_display["accuracy"].map("{:.4f}".format)
    fold_display["brier"]    = fold_display["brier"].map("{:.4f}".format)
    fold_tbl = _df_table(fold_display, caption="Per-Fold Metrics")

    return _section("Walk-Forward Validation", summary_tbl + fold_tbl)


def _section_calibration(cal_df: pd.DataFrame) -> str:
    tbl = _df_table(cal_df, caption="Per-Class Calibration (ECE, Brier Decomposition)")
    return _section("Calibration Diagnostics", tbl)


def _section_clv(clv_df: pd.DataFrame) -> str:
    from src.backtest.clv_tracker import clv_summary, clv_by_market, rolling_clv  # noqa: PLC0415

    summary = clv_summary(clv_df)
    summary_rows = [
        ("Bets tracked",          f"{summary.get('n_bets', 0):,}"),
        ("Mean CLV (log-odds)",   f"{summary.get('mean_clv_log_odds', 0):.5f}"),
        ("Mean CLV (prob)",       f"{summary.get('mean_clv_prob', 0):.5f}"),
        ("% Positive CLV",        f"{summary.get('pct_positive_clv', 0):.1f}%"),
        ("CLV t-stat",            f"{summary.get('clv_tstat', 0):.4f}"),
        ("CLV p-value",           f"{summary.get('clv_pvalue', 1):.4f}"),
        ("CLV-result correlation",f"{summary.get('clv_result_corr', 'N/A')}"),
        ("Total P&L (units)",     f"{summary.get('total_pnl_units', 0):.2f}"),
        ("ROI",                   f"{summary.get('roi_pct', 0):.2f}%"),
    ]
    summary_tbl = _kv_table(summary_rows, caption="CLV Summary")

    # Per-market table
    mkt_df = clv_by_market(clv_df)
    parts = [summary_tbl]
    if not mkt_df.empty:
        mkt_display = mkt_df[["n_bets", "mean_clv_log_odds", "pct_positive_clv",
                               "roi_pct", "total_pnl_units"]].copy()
        parts.append(_df_table(mkt_display.reset_index(), caption="CLV by Market"))

    # Rolling P&L snippet (first 20 + last 20 rows)
    roll = rolling_clv(clv_df).reset_index(drop=True)
    cum_pnl = clv_df["pnl_units"].cumsum().reset_index(drop=True)
    pnl_df = pd.DataFrame({
        "bet_n":       roll.index,
        "rolling_clv": roll.round(5),
        "cum_pnl":     cum_pnl.round(3),
    })
    display_rows = pd.concat([pnl_df.head(20), pnl_df.tail(20)]).drop_duplicates()
    parts.append(_df_table(display_rows, caption="Rolling CLV & Cumulative P&L (first 20 + last 20 bets)"))

    return _section("Closing-Line Value (CLV) Analysis", "\n".join(parts))


def _section_dispersion(summary: dict) -> str:
    rows = [(str(k).replace("_", " ").title(), str(v)) for k, v in summary.items()]
    tbl = _kv_table(rows, caption="Monte Carlo Dispersion vs NegBin Baseline")
    return _section("Run-Score Dispersion Check", tbl)


# ---------------------------------------------------------------------------
# HTML primitives
# ---------------------------------------------------------------------------

def _section(heading: str, content: str) -> str:
    return (
        f'<section>\n'
        f'  <h2>{_h(heading)}</h2>\n'
        f'  {content}\n'
        f'</section>\n'
    )


def _kv_table(rows: list[tuple[str, str]], caption: str = "") -> str:
    """Two-column key/value table."""
    cap = f"<caption>{_h(caption)}</caption>" if caption else ""
    trs = "".join(
        f"<tr><th>{_h(k)}</th><td>{_h(v)}</td></tr>"
        for k, v in rows
    )
    return f"<table>{cap}{trs}</table>"


def _df_table(df: pd.DataFrame, caption: str = "") -> str:
    """Render a DataFrame as an HTML table (no external CSS required)."""
    cap = f"<caption>{_h(caption)}</caption>" if caption else ""
    header = "<tr>" + "".join(f"<th>{_h(str(c))}</th>" for c in df.columns) + "</tr>"
    body_rows = []
    for _, row in df.iterrows():
        cells = "".join(f"<td>{_h(str(v))}</td>" for v in row)
        body_rows.append(f"<tr>{cells}</tr>")
    tbody = "".join(body_rows)
    return f"<table>{cap}<thead>{header}</thead><tbody>{tbody}</tbody></table>"


def _h(text: str) -> str:
    """HTML-escape a string."""
    return _html.escape(str(text))


def _wrap_html(title: str, body: str, timestamp: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{_h(title)}</title>
<style>
  body {{
    font-family: -apple-system, "Segoe UI", system-ui, sans-serif;
    font-size: 14px;
    line-height: 1.6;
    color: #1f2328;
    background: #ffffff;
    max-width: 960px;
    margin: 0 auto;
    padding: 24px 16px 48px;
  }}
  h1 {{
    font-size: 22px;
    font-weight: 600;
    border-bottom: 2px solid #e5e7eb;
    padding-bottom: 8px;
    margin-bottom: 4px;
  }}
  h2 {{
    font-size: 16px;
    font-weight: 600;
    color: #3b82d4;
    margin-top: 32px;
    margin-bottom: 8px;
    border-left: 3px solid #3b82d4;
    padding-left: 8px;
  }}
  table {{
    border-collapse: collapse;
    width: 100%;
    margin-bottom: 20px;
    font-size: 13px;
  }}
  caption {{
    text-align: left;
    font-weight: 600;
    color: #57606a;
    font-size: 12px;
    margin-bottom: 4px;
    caption-side: top;
    padding: 0 0 4px 0;
  }}
  th, td {{
    border: 1px solid #e5e7eb;
    padding: 6px 10px;
    text-align: left;
  }}
  th {{
    background: #f7f8fa;
    font-weight: 600;
    color: #57606a;
  }}
  tr:nth-child(even) td {{
    background: #fafbfc;
  }}
  section {{
    margin-bottom: 40px;
  }}
  .meta {{
    font-size: 12px;
    color: #57606a;
    margin-bottom: 24px;
  }}
  footer {{
    text-align: center;
    font-size: 12px;
    color: #57606a;
    border-top: 1px solid #e5e7eb;
    padding-top: 12px;
    margin-top: 40px;
  }}
</style>
</head>
<body>
<h1>{_h(title)}</h1>
<p class="meta">Generated: {_h(timestamp)}</p>
{body}
<footer>Made with IBM Bob</footer>
</body>
</html>"""


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser():
    import argparse
    p = argparse.ArgumentParser(
        description="Generate HTML backtest report from walk-forward and CLV parquet files."
    )
    p.add_argument("--wf-predictions",  help="Parquet: walk-forward predictions_df")
    p.add_argument("--wf-metrics",      help="Parquet: walk-forward fold_metrics (JSON-lines)")
    p.add_argument("--clv",             help="Parquet: CLV DataFrame from compute_clv()")
    p.add_argument("--calibration",     help="Parquet: calibration_report() DataFrame")
    p.add_argument("--output",          default="docs/backtest_report.html")
    p.add_argument("--title",           default="MLB Quant Engine — Backtest Report")
    return p


def main(argv=None) -> int:
    import json
    import sys
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_parser().parse_args(argv)

    wf_result = None
    if args.wf_predictions and Path(args.wf_predictions).exists():
        from src.backtest.walk_forward import WalkForwardResult  # noqa: PLC0415
        preds_df = pd.read_parquet(args.wf_predictions)
        fold_metrics: list[dict] = []
        if args.wf_metrics and Path(args.wf_metrics).exists():
            with open(args.wf_metrics) as fh:
                fold_metrics = json.load(fh)
        summary = {
            "n_folds":            len(fold_metrics),
            "n_oos_predictions":  len(preds_df),
            "oos_log_loss":       float(preds_df["loss"].mean()) if "loss" in preds_df else 0.0,
            "oos_accuracy":       float((preds_df["y_pred"] == preds_df["y_true"]).mean())
                                  if "y_pred" in preds_df else 0.0,
            "mean_fold_log_loss": float(np.mean([f["log_loss"] for f in fold_metrics])) if fold_metrics else 0.0,
            "std_fold_log_loss":  float(np.std([f["log_loss"]  for f in fold_metrics])) if fold_metrics else 0.0,
            "mean_fold_accuracy": float(np.mean([f["accuracy"] for f in fold_metrics])) if fold_metrics else 0.0,
            "mean_fold_brier":    float(np.mean([f["brier"]    for f in fold_metrics])) if fold_metrics else 0.0,
        }
        wf_result = WalkForwardResult(
            predictions_df=preds_df,
            fold_metrics=fold_metrics,
            summary=summary,
            n_folds=len(fold_metrics),
        )

    clv_df  = pd.read_parquet(args.clv)          if args.clv          and Path(args.clv).exists()          else None
    cal_df  = pd.read_parquet(args.calibration)   if args.calibration  and Path(args.calibration).exists()  else None

    html = build_report(wf_result, clv_df, cal_df, title=args.title)
    out  = save_report(html, args.output)
    logging.getLogger(__name__).info("Report written to %s", out)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
