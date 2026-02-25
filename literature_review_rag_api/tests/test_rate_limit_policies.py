from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from literature_rag.rate_limiter import RateLimitMiddleware, create_rate_limiter


def _build_app(config: dict) -> FastAPI:
    app = FastAPI()
    limiter = create_rate_limiter(config)
    assert limiter is not None
    app.add_middleware(RateLimitMiddleware, rate_limiter=limiter)

    @app.get("/api/auth/login")
    async def auth_login_probe():
        return {"ok": True}

    @app.get("/api/search")
    async def search_probe():
        return {"ok": True}

    @app.get("/api/plain")
    async def plain_probe():
        return {"ok": True}

    return app


def test_path_specific_limits_apply_and_return_policy_headers():
    app = _build_app(
        {
            "enabled": True,
            "requests": 10,
            "window_seconds": 60,
            "authenticated_multiplier": 1.0,
            "path_limits": [
                {
                    "name": "auth",
                    "prefixes": ["/api/auth/login"],
                    "requests": 2,
                    "window_seconds": 60,
                },
                {
                    "name": "search",
                    "prefixes": ["/api/search"],
                    "requests": 3,
                    "window_seconds": 60,
                },
            ],
        }
    )

    client = TestClient(app)

    # Auth rule: 2 allowed, 3rd throttled
    assert client.get("/api/auth/login").status_code == 200
    assert client.get("/api/auth/login").status_code == 200
    third_auth = client.get("/api/auth/login")
    assert third_auth.status_code == 429
    assert third_auth.headers.get("X-RateLimit-Policy") == "auth"

    # Search rule: 3 allowed, 4th throttled
    assert client.get("/api/search").status_code == 200
    assert client.get("/api/search").status_code == 200
    assert client.get("/api/search").status_code == 200
    fourth_search = client.get("/api/search")
    assert fourth_search.status_code == 429
    assert fourth_search.headers.get("X-RateLimit-Policy") == "search"


def test_authenticated_multiplier_uses_bearer_token_fingerprint_client_identity():
    app = _build_app(
        {
            "enabled": True,
            "requests": 2,
            "window_seconds": 60,
            "authenticated_multiplier": 2.0,
        }
    )

    client = TestClient(app)

    headers_token_a = {"Authorization": "Bearer token-A"}
    headers_token_b = {"Authorization": "Bearer token-B"}

    # Token A gets limit 4 (2 * 2.0)
    for _ in range(4):
        assert client.get("/api/plain", headers=headers_token_a).status_code == 200
    assert client.get("/api/plain", headers=headers_token_a).status_code == 429

    # Token B has independent bucket and should still be allowed initially
    assert client.get("/api/plain", headers=headers_token_b).status_code == 200
