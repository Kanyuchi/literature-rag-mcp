# Authentication Policy Matrix

This document defines the intended authentication behavior for the Retrievo API.

## Accepted Auth Mechanisms

- `Bearer <access_token>` in `Authorization` header
- HttpOnly `access_token` cookie (browser session)
- Refresh endpoint also accepts HttpOnly `refresh_token` cookie (or refresh token in body)

## Endpoint Policy

| Endpoint Group | Auth Required | Notes |
|---|---|---|
| `/healthz` | No | Liveness only, minimal payload |
| `/health` | No (minimal), Yes (detailed stats) | Public callers receive scoped minimal stats |
| `/api/auth/register` | No | Creates user + sets auth cookies |
| `/api/auth/login` | No | Issues session tokens/cookies |
| `/api/auth/refresh` | No (requires valid refresh token) | Refresh rotation required |
| `/api/auth/me` | Yes | Returns current user profile |
| `/api/jobs/**` | Yes | Tenant/user-scoped KB operations |
| `/api/chats/**` | Yes | Session/message endpoints |
| `/api/stats` | Yes | Dashboard data |
| `/api/search`, `/api/chat`, `/api/answer`, `/api/related`, `/api/synthesis` | Yes | Retrieval and generation APIs |
| `/api/upload/**`, `/api/documents/**` | Yes | Upload and document management |
| Legacy core endpoints (`/query`, `/context`, `/synthesis`, `/related`, `/papers`, `/gaps`) | Yes (when auth enabled) | Controlled by `auth.require_auth` |

## Security Invariants

- Auth is enabled by default (`auth.require_auth: true`).
- Optional verification gate can be enabled with `AUTH_REQUIRE_VERIFIED=true`.
- Tokens are not persisted in browser localStorage.
- Cookie sessions use HttpOnly cookies.
- Security and regression tests validate auth defaults and sanitized error behavior.
