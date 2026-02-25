#!/usr/bin/env bash
set -euo pipefail

if ! command -v curl >/dev/null 2>&1; then
  echo "ERROR: curl is required." >&2
  exit 1
fi

if ! command -v openssl >/dev/null 2>&1; then
  echo "ERROR: openssl is required." >&2
  exit 1
fi

DOMAIN="${1:-${DOMAIN_NAME:-}}"
if [[ -z "${DOMAIN}" ]]; then
  echo "Usage: $0 <domain>" >&2
  echo "Or set DOMAIN_NAME in the environment." >&2
  exit 1
fi

BASE_HTTP="http://${DOMAIN}"
BASE_HTTPS="https://${DOMAIN}"
CHECK_PATH="${CHECK_PATH:-/api/healthz}"
CURL_MAX_TIME="${CURL_MAX_TIME:-15}"
CURL_RESOLVE_IP="${CURL_RESOLVE_IP:-}"
OPENSSL_CONNECT_HOST="${OPENSSL_CONNECT_HOST:-${DOMAIN}}"
OPENSSL_TIMEOUT="${OPENSSL_TIMEOUT:-15}"

failures=0

pass() {
  echo "PASS: $1"
}

fail() {
  echo "FAIL: $1"
  failures=$((failures + 1))
}

declare -a CURL_COMMON_OPTS
CURL_COMMON_OPTS=(-sS --max-time "${CURL_MAX_TIME}")
if [[ -n "${CURL_RESOLVE_IP}" ]]; then
  CURL_COMMON_OPTS+=(--resolve "${DOMAIN}:80:${CURL_RESOLVE_IP}" --resolve "${DOMAIN}:443:${CURL_RESOLVE_IP}")
fi

check_http_redirect() {
  local status location
  status="$(curl "${CURL_COMMON_OPTS[@]}" -o /dev/null -w '%{http_code}' "${BASE_HTTP}${CHECK_PATH}")"
  location="$(curl "${CURL_COMMON_OPTS[@]}" -I "${BASE_HTTP}${CHECK_PATH}" | awk 'tolower($1)=="location:" {print $2}' | tr -d '\r' | head -n1)"

  if [[ "${status}" =~ ^30[1278]$ ]] && [[ "${location}" == https://* ]]; then
    pass "HTTP ${CHECK_PATH} redirects to HTTPS (${status})"
  else
    fail "HTTP ${CHECK_PATH} should redirect to HTTPS (got status=${status}, location=${location:-<none>})"
  fi
}

check_https_health() {
  local status
  status="$(curl "${CURL_COMMON_OPTS[@]}" -o /dev/null -w '%{http_code}' "${BASE_HTTPS}${CHECK_PATH}")"
  if [[ "${status}" == "200" ]]; then
    pass "HTTPS ${CHECK_PATH} is reachable (200)"
  else
    fail "HTTPS ${CHECK_PATH} expected 200 (got ${status})"
  fi
}

check_tls_certificate() {
  local cert_output not_after
  if command -v timeout >/dev/null 2>&1; then
    cert_output="$(echo | timeout "${OPENSSL_TIMEOUT}" openssl s_client -servername "${DOMAIN}" -connect "${OPENSSL_CONNECT_HOST}:443" 2>/dev/null || true)"
  else
    cert_output="$(echo | openssl s_client -servername "${DOMAIN}" -connect "${OPENSSL_CONNECT_HOST}:443" 2>/dev/null || true)"
  fi
  if [[ -z "${cert_output}" ]]; then
    fail "Could not fetch TLS certificate from ${DOMAIN}:443"
    return
  fi

  not_after="$(echo "${cert_output}" | openssl x509 -noout -enddate 2>/dev/null | cut -d= -f2 || true)"
  if [[ -n "${not_after}" ]]; then
    pass "TLS certificate present (expires: ${not_after})"
  else
    fail "Could not parse TLS certificate expiry"
  fi
}

check_security_headers() {
  local headers
  headers="$(curl "${CURL_COMMON_OPTS[@]}" -I "${BASE_HTTPS}${CHECK_PATH}")"

  grep -qi '^strict-transport-security:' <<<"${headers}" && pass "HSTS header present" || fail "HSTS header missing"
  grep -qi '^x-content-type-options: *nosniff' <<<"${headers}" && pass "X-Content-Type-Options header present" || fail "X-Content-Type-Options missing"
  grep -qi '^x-frame-options: *deny' <<<"${headers}" && pass "X-Frame-Options header present" || fail "X-Frame-Options missing"
  grep -qi '^content-security-policy:' <<<"${headers}" && pass "Content-Security-Policy header present" || fail "Content-Security-Policy missing"
  grep -qi '^referrer-policy:' <<<"${headers}" && pass "Referrer-Policy header present" || fail "Referrer-Policy missing"

  if grep -qi '^server:' <<<"${headers}"; then
    if grep -Eqi '^server:.*nginx/[0-9]' <<<"${headers}"; then
      fail "Server header reveals exact nginx version"
    else
      pass "Server header does not expose exact nginx version"
    fi
  else
    pass "Server header absent"
  fi
}

echo "Running TLS cutover checks for ${DOMAIN}"
check_http_redirect
check_https_health
check_tls_certificate
check_security_headers

if [[ ${failures} -gt 0 ]]; then
  echo "\nTLS cutover verification FAILED (${failures} checks failed)." >&2
  exit 1
fi

echo "\nTLS cutover verification PASSED."
