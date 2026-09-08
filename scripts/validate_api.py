#!/usr/bin/env python3
"""
scripts/validate_api.py
=========================
Smoke-tests all API endpoints against a running FastAPI server.

Usage:
    .venv/bin/python scripts/validate_api.py
    .venv/bin/python scripts/validate_api.py --base-url http://127.0.0.1:8000

The server must already be running:
    .venv/bin/python main.py

Checks (all endpoints, both legacy and /api/ prefix):
  /health                         → 200, status=="ok"
  /games/today                    → 200, list
  /predictions/{pk}               → 200 or 404 (cached or missing)
  /recommendations/today          → 200, list
  /backtest/summary               → 200 or 404
  /calibration/report             → 200 or 404
  /api/predictions/daily          → 200, envelope with data + meta
  /api/games/{id}/simulation      → 200, envelope with histogram data
  POST /api/predictions/sgp       → 200, envelope, joint probs in [0,1]
  /api/backtest/report            → 200 or 404, envelope if 200

Exit 0 = all pass.  Exit 1 = one or more failures.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

try:
    import requests
except ImportError:
    print("ERROR: 'requests' not installed. Run: .venv/bin/pip install requests")
    sys.exit(1)

PASS = "✓"
FAIL = "✗"
results: list[tuple[str, bool, str]] = []


def check(name: str, fn):
    try:
        fn()
        results.append((name, True, ""))
    except Exception:
        results.append((name, False, traceback.format_exc().strip().splitlines()[-1]))


def get(base: str, path: str, params: dict | None = None, timeout: int = 30):
    url = base.rstrip("/") + path
    r = requests.get(url, params=params, timeout=timeout)
    return r


def post(base: str, path: str, body: dict, timeout: int = 60):
    url = base.rstrip("/") + path
    r = requests.post(url, json=body, timeout=timeout)
    return r


def run_checks(base_url: str) -> None:

    # ── /health ──────────────────────────────────────────────────────────
    def _health():
        r = get(base_url, "/health")
        assert r.status_code == 200, f"status={r.status_code}"
        body = r.json()
        assert body.get("status") == "ok", f"body={body}"

    check("/health → 200, status=ok", _health)

    # ── /games/today ─────────────────────────────────────────────────────
    def _games_today():
        r = get(base_url, "/games/today")
        assert r.status_code == 200, f"status={r.status_code}"
        body = r.json()
        assert isinstance(body, list), f"Expected list, got {type(body)}"

    check("/games/today → 200, returns list", _games_today)

    # ── /predictions/{pk} ────────────────────────────────────────────────
    def _predictions_pk():
        # game_pk=1 may not be cached; 200 or 404 both acceptable
        r = get(base_url, "/predictions/1")
        assert r.status_code in (200, 404), f"Unexpected status {r.status_code}"
        if r.status_code == 200:
            body = r.json()
            assert "home_win_prob" in body or "simulation" in body

    check("/predictions/1 → 200 or 404", _predictions_pk)

    # ── /recommendations/today ───────────────────────────────────────────
    def _recs():
        r = get(base_url, "/recommendations/today")
        assert r.status_code in (200, 502), f"Unexpected status {r.status_code}"
        if r.status_code == 200:
            assert isinstance(r.json(), list)

    check("/recommendations/today → 200 list or 502 (no odds key)", _recs)

    # ── /backtest/summary ────────────────────────────────────────────────
    def _bt_summary():
        r = get(base_url, "/backtest/summary")
        assert r.status_code in (200, 404), f"Unexpected status {r.status_code}"

    check("/backtest/summary → 200 or 404", _bt_summary)

    # ── /calibration/report ──────────────────────────────────────────────
    def _cal_report():
        r = get(base_url, "/calibration/report")
        assert r.status_code in (200, 404), f"Unexpected status {r.status_code}"
        if r.status_code == 200:
            assert isinstance(r.json(), list)

    check("/calibration/report → 200 list or 404", _cal_report)

    # ── /api/predictions/daily ───────────────────────────────────────────
    def _api_daily():
        r = get(base_url, "/api/predictions/daily", params={"n_sims": 500})
        assert r.status_code == 200, f"status={r.status_code}; body={r.text[:200]}"
        body = r.json()
        assert "data" in body, "Missing 'data' key"
        assert "meta" in body, "Missing 'meta' key"
        assert isinstance(body["data"], list)
        assert "generated_at" in body["meta"]
        assert "version" in body["meta"]

    check("/api/predictions/daily → 200 envelope {data:list, meta}", _api_daily)

    # ── /api/games/{id}/simulation ───────────────────────────────────────
    def _api_sim():
        r = get(base_url, "/api/games/99999/simulation", params={"n_sims": 500})
        assert r.status_code == 200, f"status={r.status_code}"
        body = r.json()
        assert "data" in body
        d = body["data"]
        assert "home_win_prob" in d and "away_win_prob" in d
        total = d["home_win_prob"] + d["away_win_prob"]
        assert abs(total - 1.0) < 0.05, f"Win probs sum to {total:.4f}"
        assert "home_runs" in d and "histogram" in d["home_runs"]

    check("/api/games/{id}/simulation → 200 envelope with histogram", _api_sim)

    # ── POST /api/predictions/sgp ─────────────────────────────────────────
    def _api_sgp():
        body = {
            "game_id":  77777,
            "bankroll": 1000.0,
            "n_sims":   1000,
            "legs": [
                {"side": "home", "outcome": "moneyline", "odds": -130, "label": "H ML"},
                {"side": "home", "outcome": "over", "line": 8.5,  "odds": -110, "label": "O 8.5"},
            ],
        }
        r = post(base_url, "/api/predictions/sgp", body)
        assert r.status_code == 200, f"status={r.status_code}; body={r.text[:200]}"
        resp = r.json()
        assert "data" in resp
        d = resp["data"]
        assert 0.0 <= d["joint_prob_corr_adjusted"] <= 1.0
        assert 0.0 <= d["joint_prob_naive"] <= 1.0
        # Corr matrix diagonal should be 1
        for i in range(len(d["legs"])):
            assert abs(d["correlation_matrix"][i][i] - 1.0) < 1e-6

    check("POST /api/predictions/sgp → 200, valid probs, diagonal corr=1", _api_sgp)

    # ── /api/backtest/report ─────────────────────────────────────────────
    def _api_bt_report():
        r = get(base_url, "/api/backtest/report")
        assert r.status_code in (200, 404), f"Unexpected status {r.status_code}"
        if r.status_code == 200:
            body = r.json()
            assert "data" in body
            assert "meta" in body
            d = body["data"]
            assert "walk_forward" in d
            assert "calibration" in d
            assert "clv" in d

    check("/api/backtest/report → 200 envelope or 404", _api_bt_report)


# ── entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Validate MLB Quant Engine API endpoints")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Base URL of the running FastAPI server",
    )
    args = parser.parse_args()

    # Quick connectivity check
    try:
        requests.get(args.base_url + "/health", timeout=3)
    except requests.exceptions.ConnectionError:
        print(f"\nERROR: Cannot connect to {args.base_url}")
        print("Start the server first: .venv/bin/python main.py\n")
        sys.exit(1)

    print(f"\n── API validation against {args.base_url} ──")
    run_checks(args.base_url)

    failures = 0
    for name, ok, msg in results:
        icon = PASS if ok else FAIL
        print(f"  {icon}  {name}")
        if not ok:
            print(f"       {msg}")
            failures += 1

    print(f"\n{len(results) - failures}/{len(results)} checks passed.")
    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
