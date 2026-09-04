"""
src.api
=======
FastAPI application — exposes model predictions, bet recommendations, and
backtest summaries as a REST API consumed by the mlb-dashboard frontend.

Phase 6 will implement endpoints:
    GET  /health
    GET  /games/today
    GET  /predictions/{game_pk}
    GET  /recommendations/today
    GET  /backtest/summary
"""

from fastapi import FastAPI

app = FastAPI(
    title="MLB Quant Engine",
    description="Statcast-driven MLB prediction and betting optimisation API",
    version="0.1.0",
)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": "0.1.0"}


# TODO (Phase 6): implement remaining endpoints
