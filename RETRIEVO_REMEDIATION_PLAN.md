# Retrievo Security and Reliability Remediation Plan

## Goal
Address all constraints from the audit with measurable outcomes, clear priority, and explicit acceptance criteria before broader production scaling.

## Scope
- Security vulnerabilities (critical + medium)
- Broken UX/routes and non-functional actions
- Feature-completeness cleanup for visible but unfinished sections
- Ongoing validation in CI and monitoring

## Severity Model
- `P0`: Critical security risk or data exposure
- `P1`: High-impact reliability/UX breakage
- `P2`: Product completeness and usability improvements

## Phase Plan

### Phase 0 (P0): Immediate Risk Reduction (Target: 3-5 days)
1. Enforce HTTPS everywhere
- Actions:
  - Configure TLS certs and force HTTP->HTTPS redirect in Nginx.
  - Add HSTS (`Strict-Transport-Security`) after HTTPS verification.
- Success criteria:
  - `http://` requests return 301/308 to `https://`.
  - SSL Labs rating >= A.
  - No API/page accessible over plaintext HTTP.

2. Token and cookie hardening
- Actions:
  - Move access/refresh handling to secure, `HttpOnly`, `Secure`, `SameSite` cookies.
  - Remove JWT persistence from `localStorage`.
  - Keep CSRF defense with server validation; CSRF cookie not readable by JS.
- Success criteria:
  - No auth token in `localStorage` or JS-readable storage.
  - Auth cookies set with `HttpOnly; Secure; SameSite`.
  - CSRF-protected state-changing endpoints reject missing/invalid token.

3. Security headers baseline
- Actions:
  - Add and validate: `Content-Security-Policy`, `X-Frame-Options`, `X-Content-Type-Options`, `Referrer-Policy`, `Permissions-Policy`.
  - Hide server version (`server_tokens off`).
- Success criteria:
  - `curl -I` and browser checks confirm all required headers.
  - `Server` header no longer exposes exact Nginx/OS version.

4. Add rate limiting
- Actions:
  - Apply Nginx/app limits for auth, search, chat, and stats endpoints.
  - Return `429` on thresholds.
- Success criteria:
  - Burst tests reliably trigger throttling on protected endpoints.
  - No endpoint allows unbounded rapid requests.

5. Restrict operational endpoints
- Actions:
  - Limit `/health` detail output (or require admin/internal access).
  - Keep public health minimal if needed for probes.
- Success criteria:
  - Public health response contains no sensitive internals.
  - Detailed diagnostics are authenticated/internal only.

---

### Phase 1 (P1): Auth Consistency + Session Reliability (Target: 1 week)
1. Normalize authentication policy
- Actions:
  - Define explicit auth matrix per endpoint.
  - Align `/api/auth/me`, `/api/jobs`, `/api/stats`, `/api/search`, `/api/chat`, etc.
- Success criteria:
  - Endpoint policy doc exists and is enforced in middleware/tests.
  - No endpoint bypasses intended auth method.

2. Token expiry and refresh flow
- Actions:
  - Implement refresh-token flow with rotation and expiry handling.
  - Frontend interceptor redirects or refreshes instead of white-screen failure.
- Success criteria:
  - Expired access token never causes blank page.
  - User either gets seamless refresh or explicit re-login prompt.

3. Error response hardening
- Actions:
  - Return sanitized user-facing validation errors.
  - Keep detailed stack/type info in server logs only.
- Success criteria:
  - API responses no longer leak internal type/validation implementation details.

4. Account verification policy
- Actions:
  - Enforce and surface verification requirements by role/environment.
- Success criteria:
  - Unverified account behavior is explicit, intentional, and tested.

---

### Phase 2 (P1): Broken UX and Route Stability (Target: 1 week)
1. Fix or hide non-functional routes
- Actions:
  - Implement missing pages or remove links for: knowledge insights/graph/files/docs/settings sections currently blank.
- Success criteria:
  - No reachable route renders blank white/empty main panel.

2. Add robust 404 page
- Actions:
  - Add catch-all route with recovery actions (back/home/search).
- Success criteria:
  - Invalid routes render proper 404 UI with navigation options.

3. Repair broken actions
- Actions:
  - Fix `+ New Knowledge Base` and `+ Create agent` flows.
  - Add visible success/error states.
- Success criteria:
  - Buttons trigger correct modal/navigation/API calls consistently.

4. Dataset page loading reliability
- Actions:
  - Fix loading logic and timeout/fallback handling.
  - Surface loading errors instead of infinite spinner.
- Success criteria:
  - Dataset page reaches terminal state (data or error) within timeout.

5. Destructive action protection
- Actions:
  - Add confirmation modal for clear/delete actions.
- Success criteria:
  - No destructive action executes without explicit user confirmation.

---

### Phase 3 (P2): Product Completeness and Usability (Target: 1-2 weeks)
1. Knowledge base management
- Add rename/edit/description update/export options.
- Success criteria:
  - KB card menu supports edit + delete + export (at minimum).

2. Settings and profile completion
- Implement meaningful content for Profile/Data Sources/Model Providers/MCP/Team/Files.
- Success criteria:
  - Settings pages provide functional controls, not empty placeholders.

3. Chat and document management enhancements
- Clarify chat persistence UX and export behavior.
- Add document sorting/filtering/bulk actions where practical.
- Success criteria:
  - Users can predictably manage sessions and document operations.

## Verification and Quality Gates

### CI/CD Gates
1. Security checks
- Header assertion tests
- Auth policy tests (expected `401/403` vs `200`)
- Rate-limit tests (`429` behavior)

2. Frontend reliability checks
- Route smoke tests (no blank page)
- Action tests for create/edit/delete critical flows

3. Existing CI baseline
- Keep backend smoke + frontend lint/build as required checks.

### Monitoring and Alerting
1. Add structured security/reliability telemetry
- Auth failures, rate-limit hits, token refresh failures, 5xx spikes.
2. Add frontend error monitoring
- Capture white-screen/runtime route failures.
3. Uptime probes
- HTTPS-only availability and API status checks.

## Audit Finding Mapping
- P0 / Phase 0: Findings `1,2,3,5,6,7` and part of `4,8`
- P1 / Phase 1: Findings `4,9,10,11`
- P1 / Phase 2: Findings `12,13,14,15,16,17,19`
- P2 / Phase 3: Findings `18,20,21,22,23,24,25,26,27,28`

## Exit Criteria (Definition of Done)
1. Zero open P0 findings.
2. No blank-route regressions in automated route smoke suite.
3. Auth policy matrix implemented and validated in tests.
4. Rate-limiting and security headers verified in CI and staging.
5. All user-visible nav items are either functional or removed.
6. Release sign-off includes Security + Product + QA checklist completion.

## Suggested Execution Order
1. Complete Phase 0 and deploy to staging first.
2. Run focused security retest.
3. Complete Phase 1 and Phase 2 in parallel where possible.
4. Re-run full audit checklist.
5. Start Phase 3 as iterative product work.
