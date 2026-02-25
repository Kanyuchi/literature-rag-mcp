from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
import os
import sys

from fastapi.testclient import TestClient


def _get_app():
    # Ensure config resolves regardless of pytest working directory.
    repo_root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("CONFIG_PATH", str(repo_root / "config" / "literature_config.yaml"))
    os.environ.setdefault("INDICES_PATH", str(repo_root / "indices"))
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from literature_rag import api as api_module

    # Avoid heavy startup during smoke tests.
    @asynccontextmanager
    async def _no_lifespan(_app):
        yield

    api_module.app.router.lifespan_context = _no_lifespan
    return api_module.app


def test_root_endpoint_smoke():
    app = _get_app()
    client = TestClient(app)

    response = client.get("/")
    assert response.status_code == 200
    payload = response.json()

    assert payload.get("name") == "Literature Review RAG API"
    assert payload.get("health") == "/health"


def test_health_endpoint_smoke():
    app = _get_app()
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()

    assert payload.get("status") in {"healthy", "initializing", "unhealthy"}
    assert isinstance(payload.get("ready"), bool)
    assert "stats" in payload
