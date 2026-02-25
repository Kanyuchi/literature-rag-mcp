"""Rate Limiting Module for Literature RAG

Provides sliding window rate limiting with per-tenant support.
Uses in-memory storage by default; can be extended to Redis for distributed deployments.
"""

import time
import logging
from collections import defaultdict
from dataclasses import dataclass
from threading import Lock
from typing import Optional, Dict, Tuple

from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    enabled: bool = False
    requests: int = 100  # Max requests per window
    window_seconds: int = 60  # Time window in seconds
    burst_multiplier: float = 1.5  # Allow burst up to requests * multiplier
    authenticated_multiplier: float = 2.0  # Authenticated users get higher limits


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
        self._counter = SlidingWindowCounter(
            window_seconds=config.window_seconds,
            bucket_count=10
        )

    def get_client_id(self, request: Request) -> str:
        """Extract client identifier from request.

        Uses user_id if authenticated, otherwise falls back to IP address.
        """
        # Check for authenticated user (set by auth middleware)
        user = getattr(request.state, "user", None)
        if user and hasattr(user, "id"):
            return f"user:{user.id}"

        # Fall back to IP address
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            # Take the first IP in the chain (original client)
            ip = forwarded.split(",")[0].strip()
        else:
            ip = request.client.host if request.client else "unknown"

        return f"ip:{ip}"

    def get_limit(self, request: Request) -> int:
        """Get rate limit for this request.

        Authenticated users get higher limits.
        """
        base_limit = self.config.requests
        user = getattr(request.state, "user", None)

        if user:
            return int(base_limit * self.config.authenticated_multiplier)
        return base_limit

    def check_rate_limit(self, request: Request) -> Tuple[bool, int, int, int]:
        """Check if request is within rate limit.

        Args:
            request: FastAPI request object

        Returns:
            Tuple of (allowed, current_count, limit, retry_after_seconds)
        """
        if not self.config.enabled:
            return True, 0, 0, 0

        client_id = self.get_client_id(request)
        limit = self.get_limit(request)
        current = self._counter.get_count(client_id)

        if current >= limit:
            retry_after = self.config.window_seconds
            return False, current, limit, retry_after

        return True, current, limit, 0

    def record_request(self, request: Request) -> int:
        """Record a request and return new count."""
        if not self.config.enabled:
            return 0

        client_id = self.get_client_id(request)
        return self._counter.increment(client_id)


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
        allowed, current, limit, retry_after = self.rate_limiter.check_rate_limit(request)

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
                    "detail": f"Too many requests. Limit: {limit} per {self.rate_limiter.config.window_seconds}s",
                    "retry_after": retry_after
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(time.time()) + retry_after)
                }
            )

        # Record request and proceed
        new_count = self.rate_limiter.record_request(request)
        response = await call_next(request)

        # Add rate limit headers to response
        if self.rate_limiter.config.enabled:
            remaining = max(0, limit - new_count)
            response.headers["X-RateLimit-Limit"] = str(limit)
            response.headers["X-RateLimit-Remaining"] = str(remaining)
            response.headers["X-RateLimit-Reset"] = str(
                int(time.time()) + self.rate_limiter.config.window_seconds
            )

        return response


def create_rate_limiter(config_dict: Dict) -> Optional[RateLimiter]:
    """Create rate limiter from config dictionary.

    Args:
        config_dict: Rate limit config from YAML/env

    Returns:
        RateLimiter instance or None if disabled
    """
    config = RateLimitConfig(
        enabled=config_dict.get("enabled", False),
        requests=config_dict.get("requests", 100),
        window_seconds=config_dict.get("window_seconds", 60),
        burst_multiplier=config_dict.get("burst_multiplier", 1.5),
        authenticated_multiplier=config_dict.get("authenticated_multiplier", 2.0)
    )

    if not config.enabled:
        logger.info("Rate limiting is disabled")
        return None

    logger.info(
        f"Rate limiting enabled: {config.requests} requests per {config.window_seconds}s "
        f"(authenticated: {int(config.requests * config.authenticated_multiplier)})"
    )
    return RateLimiter(config)
