"""
Smoke tests — verify the scaffold imports cleanly and the API health
endpoint responds correctly.
"""

from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_subpackage_imports():
    """All src subpackages must be importable without error."""
    import src.backtest  # noqa: F401
    import src.features  # noqa: F401
    import src.ingestion  # noqa: F401
    import src.models  # noqa: F401
    import src.optimizer  # noqa: F401
