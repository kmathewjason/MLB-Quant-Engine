"""
mlb-quant-engine — entry point

Run the FastAPI server:
    uvicorn main:app --host 127.0.0.1 --port 8000 --reload

Run the daily prediction pipeline:
    python main.py --run-pipeline
"""

import argparse

from dotenv import load_dotenv

load_dotenv()


def run_pipeline() -> None:
    """Orchestrate daily ingestion → feature engineering → inference."""
    raise NotImplementedError("Pipeline not yet implemented — see Phase 1+.")


def get_app():
    """Lazy import so uvicorn can import this module without full pipeline deps."""
    from src.api import app  # noqa: PLC0415

    return app


# Expose app at module level for uvicorn
app = get_app()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLB Quant Engine")
    parser.add_argument(
        "--run-pipeline",
        action="store_true",
        help="Execute the daily prediction pipeline",
    )
    args = parser.parse_args()

    if args.run_pipeline:
        run_pipeline()
    else:
        import uvicorn

        uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
