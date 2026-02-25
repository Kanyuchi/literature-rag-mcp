"""Rate Limiting Module for Literature RAG

Provides sliding window rate limiting with per-tenant support.
Uses in-memory storage by default; can be extended to Redis for distributed deployments.
"""

import hashlib
import time
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock
from typing import Optional, Dict, Tuple, List
from fastapi import Request, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


@dataclass
class PathLimitRule:
    """Path-specific rate limit rule."""
    name: str
    prefixes: List[str] = field(default_factory=list)
    requests: int = 100
    window_seconds: int = 60
    authenticated_multiplier: Optional[float] = None


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    enabled: bool = False
    requests: int = 100  # Max requests per window
    window_seconds: int = 60  # Time window in seconds
    burst_multiplier: float = 1.5  # Allow burst up to requests * multiplier
    authenticated_multiplier: float = 2.0  # Authenticated users get higher limits
    path_limits: List[PathLimitRule] = field(default_factory=list)  # Optional per-path overrides


class SlidingWindowCounter:
    """Sliding window rate limiter using in-memory storage.

    Thread-safe implementation that tracks request counts in time buckets.
    Suitable for single-instance deployments.
    """

    def __init__(self, window_seconds: int = 60, bucket_count: int = 10):
        """Initialize the sliding window counter.

        Args:
            window_seconds: Total window duration
            bucket_count: Number of buckets to divide the window into
        """
        self.window_seconds = window_seconds
        self.bucket_count = bucket_count
        self.bucket_duration = window_seconds / bucket_count

        # {client_id: {bucket_index: count}}
        self._buckets: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self._lock = Lock()

    def _get_bucket_index(self) -> int:
        """Get current bucket index based on time."""
        return int(time.time() / self.bucket_duration) % self.bucket_count

    def _cleanup_old_buckets(self, client_id: str) -> None:
        """Remove buckets outside the current window."""
        current_time = time.time()
        current_bucket = self._get_bucket_index()

        # Calculate which buckets are still valid
        valid_buckets = set()
        for i in range(self.bucket_count):
            bucket_idx = (current_bucket - i) % self.bucket_count
            valid_buckets.add(bucket_idx)

        # Remove invalid buckets
        if client_id in self._buckets:
            invalid = [k for k in self._buckets[client_id] if k not in valid_buckets]
            for k in invalid:
                del self._buckets[client_id][k]

    def get_count(self, client_id: str) -> int:
        """Get total request count for client in current window."""
        with self._lock:
            self._cleanup_old_buckets(client_id)
            return sum(self._buckets[client_id].values())

    def increment(self, client_id: str) -> int:
        """Increment count for client and return new total."""
        with self._lock:
            self._cleanup_old_buckets(client_id)
            bucket = self._get_bucket_index()
            self._buckets[client_id][bucket] += 1
            return sum(self._buckets[client_id].values())

    def reset(self, client_id: str) -> None:
        """Reset counts for a client."""
        with self._lock:
            if client_id in self._buckets:
                del self._buckets[client_id]


class RateLimiter:
    """Rate limiter with per-tenant support."""

    def __init__(self, config: RateLimitConfig):
        """Initialize rate limiter.

        Args:
            config: Rate limit configuration
        """
        self.config = config
        self._default_counter = SlidingWindowCounter(
            window_seconds=config.window_seconds,
            bucket_count=10
        )
        self._rule_counters: Dict[str, SlidingWindowCounter] = {
            rule.name: SlidingWindowCounter(window_seconds=rule.window_seconds, bucket_count=10)
            for rule in config.path_limits
        }
        self._sorted_rules = sorted(
            config.path_limits,
            key=lambda rule: max((len(prefix) for prefix in rule.prefixes), default=0),
            reverse=True
        )

    def _is_authenticated_request(self, request: Request) -> bool:
        """Best-effort authenticated request detection for rate-limit multipliers."""
        user = getattr(request.state, "user", None)
        if user and hasattr(user, "id"):
            return True

        auth_header = request.headers.get("authorization", "")
        if auth_header.lower().startswith("bearer "):
            return True

        return bool(request.cookies.get("access_token"))

    def _select_path_rule(self, path: str) -> Optional[PathLimitRule]:
        """Pick the most specific path rule that matches the request path."""
        for rule in self._sorted_rules:
            for prefix in rule.prefixes:
                if prefix and path.startswith(prefix):
                    return rule
        return None

    def get_client_id(self, request: Request) -> str:
        """Extract client identifier from request.

        Uses user_id if authenticated, then token fingerprint, otherwise IP.
        """
        # Check for authenticated user (set by auth middleware)
        user = getattr(request.state, "user", None)
        if user and hasattr(user, "id"):
            return f"user:{user.id}"

        auth_header = request.headers.get("authorization", "")
        if auth_header.lower().startswith("bearer "):
            token = auth_header[7:].strip()
            if token:
                token_fingerprint = hashlib.sha256(token.encode("utf-8")).hexdigest()[:24]
                return f"bearer:{token_fingerprint}"

        cookie_token = request.cookies.get("access_token")
        if cookie_token:
            token_fingerprint = hashlib.sha256(cookie_token.encode("utf-8")).hexdigest()[:24]
            return f"session:{token_fingerprint}"

        # Fall back to IP address
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            # Take the first IP in the chain (original client)
            ip = forwarded.split(",")[0].strip()
        else:
            ip = request.client.host if request.client else "unknown"

        return f"ip:{ip}"

    def _resolve_context(
        self, request: Request
    ) -> Tuple[str, int, int, SlidingWindowCounter, str]:
        """Resolve client/rule/counter context for this request."""
        client_id = self.get_client_id(request)
        path = request.url.path
        matched_rule = self._select_path_rule(path)

        if matched_rule:
            base_limit = matched_rule.requests
            window_seconds = matched_rule.window_seconds
            auth_multiplier = (
                matched_rule.authenticated_multiplier
                if matched_rule.authenticated_multiplier is not None
                else self.config.authenticated_multiplier
            )
            counter = self._rule_counters[matched_rule.name]
            policy_name = matched_rule.name
        else:
            base_limit = self.config.requests
            window_seconds = self.config.window_seconds
            auth_multiplier = self.config.authenticated_multiplier
            counter = self._default_counter
            policy_name = "default"

        is_authenticated = self._is_authenticated_request(request)
        limit = int(base_limit * auth_multiplier) if is_authenticated else base_limit
        return client_id, limit, window_seconds, counter, policy_name

    def check_rate_limit(self, request: Request) -> Tuple[bool, int, int, int, int, str]:
        """Check if request is within rate limit.

        Args:
            request: FastAPI request object

        Returns:
            Tuple of (allowed, current_count, limit, retry_after_seconds, window_seconds, policy_name)
        """
        if not self.config.enabled:
            return True, 0, 0, 0, 0, "disabled"

        client_id, limit, window_seconds, counter, policy_name = self._resolve_context(request)
        current = counter.get_count(client_id)

        if current >= limit:
            retry_after = window_seconds
            return False, current, limit, retry_after, window_seconds, policy_name

        return True, current, limit, 0, window_seconds, policy_name

    def record_request(self, request: Request) -> Tuple[int, int, int, str]:
        """Record a request and return (count, limit, window_seconds, policy_name)."""
        if not self.config.enabled:
            return 0, 0, 0, "disabled"

        client_id, limit, window_seconds, counter, policy_name = self._resolve_context(request)
        count = counter.increment(client_id)
        return count, limit, window_seconds, policy_name


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting."""

    def __init__(self, app, rate_limiter: RateLimiter):
        """Initialize middleware.

        Args:
            app: FastAPI application
            rate_limiter: RateLimiter instance
        """
        super().__init__(app)
        self.rate_limiter = rate_limiter

    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting."""
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/healthz", "/", "/docs", "/openapi.json"]:
            return await call_next(request)

        # Check rate limit
        allowed, current, limit, retry_after, window_seconds, policy_name = self.rate_limiter.check_rate_limit(request)

        if not allowed:
            client_id = self.rate_limiter.get_client_id(request)
            logger.warning(
                "rate_limit_exceeded",
                extra={
                    "event": "rate_limit_exceeded",
                    "client_id": client_id,
                    "current": current,
                    "limit": limit,
                    "path": request.url.path
                }
            )
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "detail": f"Too many requests. Limit: {limit} per {window_seconds}s",
                    "retry_after": retry_after
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(time.time()) + retry_after),
                    "X-RateLimit-Window": str(window_seconds),
                    "X-RateLimit-Policy": policy_name,
                }
            )

        # Record request and proceed
        new_count, limit, window_seconds, policy_name = self.rate_limiter.record_request(request)
        response = await call_next(request)

        # Add rate limit headers to response
        if self.rate_limiter.config.enabled:
            remaining = max(0, limit - new_count)
            response.headers["X-RateLimit-Limit"] = str(limit)
            response.headers["X-RateLimit-Remaining"] = str(remaining)
            response.headers["X-RateLimit-Reset"] = str(
                int(time.time()) + window_seconds
            )
            response.headers["X-RateLimit-Window"] = str(window_seconds)
            response.headers["X-RateLimit-Policy"] = policy_name

        return response


def create_rate_limiter(config_dict: Dict) -> Optional[RateLimiter]:
    """Create rate limiter from config dictionary.

    Args:
        config_dict: Rate limit config from YAML/env

    Returns:
        RateLimiter instance or None if disabled
    """
    raw_path_limits = config_dict.get("path_limits", []) or []
    parsed_path_limits: List[PathLimitRule] = []

    for index, raw_rule in enumerate(raw_path_limits):
        if not isinstance(raw_rule, dict):
            continue
        prefixes = raw_rule.get("prefixes", []) or []
        prefixes = [prefix for prefix in prefixes if isinstance(prefix, str) and prefix.strip()]
        if not prefixes:
            continue

        requests = int(raw_rule.get("requests", config_dict.get("requests", 100)))
        window_seconds = int(raw_rule.get("window_seconds", config_dict.get("window_seconds", 60)))
        if requests <= 0 or window_seconds <= 0:
            continue

        auth_multiplier_raw = raw_rule.get("authenticated_multiplier")
        auth_multiplier = float(auth_multiplier_raw) if auth_multiplier_raw is not None else None
        rule_name = str(raw_rule.get("name") or f"path_rule_{index}")

        parsed_path_limits.append(
            PathLimitRule(
                name=rule_name,
                prefixes=prefixes,
                requests=requests,
                window_seconds=window_seconds,
                authenticated_multiplier=auth_multiplier,
            )
        )

    config = RateLimitConfig(
        enabled=config_dict.get("enabled", False),
        requests=config_dict.get("requests", 100),
        window_seconds=config_dict.get("window_seconds", 60),
        burst_multiplier=config_dict.get("burst_multiplier", 1.5),
        authenticated_multiplier=config_dict.get("authenticated_multiplier", 2.0),
        path_limits=parsed_path_limits,
    )

    if not config.enabled:
        logger.info("Rate limiting is disabled")
        return None

    logger.info(
        f"Rate limiting enabled: {config.requests} requests per {config.window_seconds}s "
        f"(authenticated: {int(config.requests * config.authenticated_multiplier)})"
    )
    if config.path_limits:
        logger.info(
            "rate_limit_path_overrides",
            extra={
                "event": "rate_limit_path_overrides",
                "count": len(config.path_limits),
                "rules": [
                    {
                        "name": rule.name,
                        "prefixes": rule.prefixes,
                        "requests": rule.requests,
                        "window_seconds": rule.window_seconds,
                        "authenticated_multiplier": rule.authenticated_multiplier,
                    }
                    for rule in config.path_limits
                ],
            }
        )
    return RateLimiter(config)
