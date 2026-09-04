# MLB Quant Engine — Setup Guide

## Prerequisites

| Tool | Version |
|------|---------|
| Python | ≥ 3.11 |
| Node.js | ≥ 20 (dashboard only) |
| git | any recent |

## Quick Start

```bash
# 1. Clone and enter the repo
git clone <repo-url>
cd mlb-quant-engine

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env — add your ODDS_API_KEY and WEATHER_API_KEY

# 5. Start the API server
uvicorn main:app --host 127.0.0.1 --port 8000 --reload

# 6. Verify
curl http://127.0.0.1:8000/health
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `ODDS_API_KEY` | The-Odds-API key for live odds | Phase 1+ |
| `WEATHER_API_KEY` | WeatherAPI / OpenWeatherMap key | Phase 1+ |
| `MLB_STATS_API_BASE` | MLB Stats API base URL (default provided) | Phase 1+ |

## Running Tests

```bash
pytest tests/ -v
```

## Data Directory

`data/raw/` and `data/processed/` are git-ignored. After Phase 1 you'll
populate them via:

```bash
python -m src.ingestion.statcast   # pulls Statcast → data/raw/
```

## Dashboard (Phase 6)

```bash
cd mlb-dashboard
npm install
npm run dev        # Vite dev server at http://127.0.0.1:5173
```
