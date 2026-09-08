# MLB Quant Engine — Setup Guide

> **Root of the project:** `mlb-quant-engine/`  
> Every command below assumes you are inside that directory unless noted.

---

## Prerequisites

| Tool | Minimum version | Notes |
|------|-----------------|-------|
| Python | 3.11 | 3.13 used in development |
| Node.js | 20 | Dashboard only |
| npm | 9 | Dashboard only |
| git | any | |

---

## 1 — Clone & Bootstrap

```bash
git clone <repo-url>
cd mlb-quant-engine

# Create the virtualenv (first time only)
python3 -m venv .venv

# Install all Python dependencies
.venv/bin/pip install -r requirements.txt
```

---

## 2 — Environment Variables

```bash
cp .env.example .env
# Then open .env and fill in your keys
```

| Variable | Description | Required for |
|---|---|---|
| `ODDS_API_KEY` | [The-Odds-API](https://the-odds-api.com/) key | Live odds, bet sizing |
| `VISUAL_CROSSING_API_KEY` | [Visual Crossing](https://www.visualcrossing.com/) key | Weather adjustments |
| `MLB_STATS_API_BASE` | Default: `https://statsapi.mlb.com/api/v1` | Schedule, rosters |
| `MLB_PRED_DIR` | Override predictions output dir (used in tests) | Testing only |

The engine runs without any keys — simulations and model inference work
offline; only live odds fetching and weather fetching degrade gracefully.

---

## 3 — Running the Server

```bash
# Start the FastAPI backend (binds to 127.0.0.1:8000)
.venv/bin/python main.py

# With a custom port
.venv/bin/python main.py --port 8080

# With hot reload (development)
.venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Open `http://127.0.0.1:8000/docs` for interactive Swagger UI.

---

## 4 — Running the Dashboard

Open a **second terminal**:

```bash
cd mlb-quant-engine/mlb-dashboard
npm run dev        # Vite dev server at http://127.0.0.1:5173
```

The Vite dev proxy forwards all `/api/*` requests to the FastAPI server
automatically — both must be running at the same time.

For a production build:

```bash
npm run build      # outputs to mlb-dashboard/dist/
```

---

## 5 — Daily Pipeline

The pipeline ingests data, runs simulations, sizes bets, and writes CSVs
to `data/predictions/`.

```bash
# Run the full pipeline for today
.venv/bin/python main.py --pipeline

# Run for a specific date
.venv/bin/python main.py --pipeline --date 2025-04-15

# Tune parameters
.venv/bin/python main.py --pipeline \
    --bankroll 5000 \
    --devig power \
    --n-sims 100000 \
    --output data/predictions
```

### Pipeline Stages

```
1. fetch_schedule()        MLB Stats API → game list
2. fetch_live_odds()       The-Odds-API  → moneyline / totals / spreads
3. feature engineering     bayesian_shrinkage + log5 + park/weather
4. simulate_game()         Monte Carlo (vectorised NumPy, n_sims per game)
5. best_devig()            power / additive / Shin de-vig
6. portfolio_kelly()       covariance-adjusted Kelly sizing
7. apply_caps()            per-bet / team / daily / drawdown guardrails
8. write CSV               data/predictions/recommendations_YYYY-MM-DD.csv
```

---

## 6 — Backtest

```bash
# Run walk-forward backtest (writes to data/predictions/)
.venv/bin/python main.py --backtest

# View the HTML report that was generated
open data/predictions/backtest_report.html   # macOS
```

Outputs written:
- `backtest_summary.json`
- `wf_fold_metrics.json`
- `wf_predictions.parquet`
- `calibration_report.csv`
- `clv_report.parquet`
- `backtest_report.html`

---

## 7 — Recommendations (stdout)

```bash
# Print today's bet recommendations without starting the server
.venv/bin/python main.py --recommend

# For a specific date and bankroll
.venv/bin/python main.py --recommend --date 2025-04-15 --bankroll 2000
```

---

## 8 — Smoke-Test Validation Scripts

Each `scripts/validate_*.py` script tests one layer of the stack
independently — no API keys needed, all external I/O is skipped gracefully.

```bash
# Test ingestion layer (schedule parser, odds structures, Statcast cache)
.venv/bin/python scripts/validate_ingestion.py

# Test feature engineering (shrinkage, log5, park/weather)
.venv/bin/python scripts/validate_features.py

# Test models (Glicko-2, simulator, PA outcome model shapes)
.venv/bin/python scripts/validate_models.py

# Test optimizer (de-vig, Kelly, portfolio caps, drawdown)
.venv/bin/python scripts/validate_optimizer.py

# Test all API endpoints against the live server (server must be running)
.venv/bin/python scripts/validate_api.py [--base-url http://127.0.0.1:8000]
```

All scripts exit `0` on success and print a ✓/✗ summary per check.

---

## 9 — Full Test Suite

```bash
# Run all 328 unit + integration tests
.venv/bin/python -m pytest tests/ -v

# Run a single test file
.venv/bin/python -m pytest tests/test_models.py -v

# Run with coverage
.venv/bin/python -m pytest tests/ --cov=src --cov-report=term-missing
```

---

## 10 — CLI Reference

```
usage: main.py [-h] [--pipeline] [--backtest] [--recommend]
               [--date YYYY-MM-DD] [--bankroll FLOAT]
               [--devig {power,additive,shin}] [--n-sims INT]
               [--output DIR] [--port INT]

Modes (mutually exclusive; default = start server):
  --pipeline    Run full ingestion→simulation→optimisation pipeline and exit
  --backtest    Run walk-forward backtest and write HTML report, then exit
  --recommend   Print today's bet recommendations to stdout and exit

Shared parameters:
  --date        ISO date for pipeline/backtest/recommend  [default: today]
  --bankroll    Bankroll in units for Kelly sizing         [default: 1000]
  --devig       De-vig method: power | additive | shin    [default: power]
  --n-sims      Monte Carlo simulations per game          [default: 50000]
  --output      Output directory for CSV/JSON/Parquet     [default: data/predictions]

Server parameters (only used without --pipeline/--backtest/--recommend):
  --port        FastAPI listen port                        [default: 8000]
```

---

## 11 — Project Layout

```
mlb-quant-engine/
├── data/
│   ├── raw/            Statcast parquet, game logs  (git-ignored)
│   ├── processed/      Engineered feature matrices  (git-ignored)
│   └── predictions/    Daily CSVs + backtest output (git-ignored)
├── models/             Saved model artifacts (*.json, *.pkl)
├── scripts/
│   ├── validate_ingestion.py
│   ├── validate_features.py
│   ├── validate_models.py
│   ├── validate_optimizer.py
│   └── validate_api.py
├── src/
│   ├── api.py          FastAPI application (all endpoints)
│   ├── ingestion/      mlb_stats_api · statcast · weather · live_odds
│   ├── features/       bayesian_shrinkage · matchup_features · park_weather
│   ├── models/         markov_re_matrix · pitcher_elo · pa_outcome_model
│   │                   game_simulator · ensemble
│   ├── optimizer/      vig_removal · kelly · portfolio_cap
│   └── backtest/       walk_forward · calibration · clv_tracker
│                       run_report · run_dispersion_check
├── tests/              328 unit + integration tests
├── mlb-dashboard/      React 19 + Vite + Tailwind v4 frontend
├── docs/
│   ├── QUANT_MATH.md   Mathematical reference (all models)
│   └── SETUP.md        This file
├── main.py             Pipeline runner + FastAPI entry point
├── requirements.txt
└── .env.example
```

---

## 12 — API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check |
| GET | `/games/today` | Today's schedule (legacy) |
| GET | `/predictions/{game_pk}` | Win-prob + totals for one game (legacy) |
| GET | `/recommendations/today` | Bet recommendations with Kelly sizing (legacy) |
| GET | `/backtest/summary` | Latest backtest JSON (legacy) |
| GET | `/calibration/report` | Latest calibration CSV as JSON (legacy) |
| GET | `/api/predictions/daily` | Full slate: sim block + markets + Kelly |
| GET | `/api/games/{game_id}/simulation` | Score distribution histograms |
| POST | `/api/predictions/sgp` | Same-game parlay with correlation-adjusted Kelly |
| GET | `/api/backtest/report` | Full backtest report as structured JSON |

All `/api/` endpoints return `{ data: <payload>, meta: { generated_at, version } }`.

---

## 13 — Adding Data

```bash
# Pull Statcast pitch-level data for a date range
.venv/bin/python -c "
from src.ingestion.statcast import fetch_statcast_range
fetch_statcast_range('2024-04-01', '2024-04-30')
"

# Pull historical schedule
.venv/bin/python -c "
from src.ingestion.mlb_stats_api import get_schedule
games = get_schedule('2025-04-15')
print(f'{len(games)} games')
"
```

---

*Last updated: Phase 9 (complete build including React dashboard)*
