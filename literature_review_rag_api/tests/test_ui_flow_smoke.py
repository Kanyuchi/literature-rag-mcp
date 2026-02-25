from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
import os
import sys
import uuid

from fastapi.testclient import TestClient



def _get_app():
    repo_root = Path(__file__).resolve().parents[1]
    os.environ["CONFIG_PATH"] = str(repo_root / "config" / "literature_config.yaml")
    os.environ["INDICES_PATH"] = str(repo_root / "indices")
    os.environ["AUTH_REQUIRE_AUTH"] = "true"
    os.environ["AUTH_REQUIRE_VERIFIED"] = "false"
    os.environ["REQUIRE_HTTPS"] = "false"

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from literature_rag import api as api_module

    @asynccontextmanager
    async def _no_lifespan(_app):
        yield

    api_module.app.router.lifespan_context = _no_lifespan
    return api_module.app



def _register_and_get_token(client: TestClient) -> str:
    email = f"ui_smoke_{uuid.uuid4().hex}@example.com"
    response = client.post(
        "/api/auth/register",
        json={
            "email": email,
            "password": "StrongPass123",
            "name": "UI Smoke User",
        },
    )
    assert response.status_code == 200
    return response.json()["access_token"]



def test_jobs_create_and_delete_flow_smoke():
    app = _get_app()
    client = TestClient(app)

    token = _register_and_get_token(client)
    headers = {"Authorization": f"Bearer {token}"}

    create_response = client.post(
        "/api/jobs",
        headers=headers,
        json={
            "name": "Smoke Knowledge Base",
            "description": "Created by smoke test",
            "extractor_type": "auto",
        },
    )
    assert create_response.status_code == 200
    job_id = create_response.json()["id"]

    list_response = client.get("/api/jobs", headers=headers)
    assert list_response.status_code == 200
    list_payload = list_response.json()
    assert any(job["id"] == job_id for job in list_payload["jobs"])

    delete_response = client.delete(f"/api/jobs/{job_id}", headers=headers)
    assert delete_response.status_code == 200

    list_after_delete = client.get("/api/jobs", headers=headers)
    assert list_after_delete.status_code == 200
    list_after_payload = list_after_delete.json()
    assert all(job["id"] != job_id for job in list_after_payload["jobs"])



def test_dataset_route_backing_endpoints_return_terminal_state_for_job_kb():
    app = _get_app()
    client = TestClient(app)

    token = _register_and_get_token(client)
    headers = {"Authorization": f"Bearer {token}"}

    create_response = client.post(
        "/api/jobs",
        headers=headers,
        json={"name": "Dataset Smoke KB", "description": "Dataset test", "extractor_type": "auto"},
    )
    assert create_response.status_code == 200
    job_id = create_response.json()["id"]

    docs_response = client.get(f"/api/jobs/{job_id}/documents?limit=500", headers=headers)
    stats_response = client.get(f"/api/jobs/{job_id}/stats", headers=headers)

    assert docs_response.status_code == 200
    assert "documents" in docs_response.json()

    assert stats_response.status_code == 200
    stats_payload = stats_response.json()
    assert "document_count" in stats_payload
    assert "chunk_count" in stats_payload



def test_dataset_route_backing_endpoints_return_terminal_error_for_missing_kb():
    app = _get_app()
    client = TestClient(app)

    token = _register_and_get_token(client)
    headers = {"Authorization": f"Bearer {token}"}

    missing_job_id = 999999999
    docs_response = client.get(f"/api/jobs/{missing_job_id}/documents?limit=500", headers=headers)
    stats_response = client.get(f"/api/jobs/{missing_job_id}/stats", headers=headers)

    assert docs_response.status_code == 404
    assert stats_response.status_code == 404
