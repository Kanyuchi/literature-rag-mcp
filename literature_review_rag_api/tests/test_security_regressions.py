from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
import importlib
import sys
import uuid

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient


@pytest.fixture
def api_module(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("CONFIG_PATH", str(repo_root / "config" / "literature_config.yaml"))
    monkeypatch.setenv("INDICES_PATH", str(repo_root / "indices"))
    monkeypatch.setenv("AUTH_REQUIRE_AUTH", "true")
    monkeypatch.setenv("REQUIRE_HTTPS", "false")

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    api = importlib.import_module("literature_rag.api")

    @asynccontextmanager
    async def _no_lifespan(_app):
        yield

    api.app.router.lifespan_context = _no_lifespan
    api.app.dependency_overrides = {}
    yield api
    api.app.dependency_overrides = {}


@pytest.fixture
def client(api_module):
    with TestClient(api_module.app, raise_server_exceptions=False) as test_client:
        yield test_client


def test_security_headers_are_present(client: TestClient):
    response = client.get("/healthz")
    assert response.status_code == 200

    assert response.headers.get("x-content-type-options") == "nosniff"
    assert response.headers.get("x-frame-options") == "DENY"
    assert response.headers.get("referrer-policy") == "strict-origin-when-cross-origin"
    assert "frame-ancestors 'none'" in response.headers.get("content-security-policy", "")
    assert response.headers.get("x-xss-protection") == "0"


def test_validation_errors_are_sanitized(client: TestClient):
    response = client.post("/api/auth/login", json={})
    assert response.status_code == 422

    payload = response.json()
    assert payload == {
        "error": "Invalid request",
        "detail": "One or more request fields are invalid.",
    }


def test_http_exception_5xx_is_sanitized(api_module, client: TestClient):
    route_path = f"/_test_http_exc_{uuid.uuid4().hex}"

    @api_module.app.get(route_path)
    async def _test_route_http_exception():
        raise HTTPException(status_code=500, detail="sensitive internal message")

    response = client.get(route_path)
    assert response.status_code == 500
    payload = response.json()

    assert payload.get("error") == "Internal server error"
    assert "sensitive" not in str(payload).lower()


def test_general_exception_5xx_is_sanitized(api_module, client: TestClient):
    route_path = f"/_test_exc_{uuid.uuid4().hex}"

    @api_module.app.get(route_path)
    async def _test_route_exception():
        raise RuntimeError("sensitive stack details")

    response = client.get(route_path)
    assert response.status_code == 500
    payload = response.json()

    assert payload.get("error") == "Internal server error"
    assert "sensitive" not in str(payload).lower()


def test_health_public_response_is_minimal(api_module, client: TestClient, monkeypatch):
    class DummyRAG:
        def is_ready(self):
            return True

        def get_stats(self):
            return {"total_chunks": 10, "total_papers": 2}

    monkeypatch.setattr(api_module, "rag_system", DummyRAG())

    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()

    assert payload.get("stats", {}).get("scope") == "public"
    assert "total_chunks" not in payload.get("stats", {})


def test_stats_requires_auth_by_default(api_module, client: TestClient, monkeypatch):
    class DummyRAG:
        def is_ready(self):
            return True

        def get_stats(self):
            return {
                "total_papers": 2,
                "total_chunks": 10,
                "papers_by_phase": {},
                "papers_by_topic": {},
                "year_range": {"min": 2020, "max": 2024},
            }

    monkeypatch.setattr(api_module, "rag_system", DummyRAG())

    unauth_response = client.get("/api/stats")
    assert unauth_response.status_code == 401

    api_module.app.dependency_overrides[api_module.get_current_user_optional] = lambda: SimpleNamespace(
        id=1,
        email="verified@example.com",
        is_active=True,
        is_verified=True,
    )

    auth_response = client.get("/api/stats")
    assert auth_response.status_code == 200
    payload = auth_response.json()
    assert payload["total_chunks"] == 10
    assert payload["total_papers"] == 2


def test_unverified_accounts_can_be_blocked_when_policy_enabled(client: TestClient, monkeypatch):
    monkeypatch.setenv("AUTH_REQUIRE_VERIFIED", "true")
    email = f"unverified_{uuid.uuid4().hex}@example.com"

    register_response = client.post(
        "/api/auth/register",
        json={
            "email": email,
            "password": "StrongPass123",
            "name": "Unverified User",
        },
    )
    assert register_response.status_code == 200
    token = register_response.json()["access_token"]

    me_response = client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert me_response.status_code == 403
    assert me_response.json().get("error") == "Email verification required"
