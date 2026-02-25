"""Authentication Module for Literature RAG

Provides JWT token management, password hashing, and OAuth utilities.
"""

import os
import logging
import hashlib
import secrets
import threading
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any

import bcrypt
from jose import JWTError, jwt
from fastapi import HTTPException, status, Depends, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from .database import get_db, User, UserCRUD, RefreshTokenCRUD

logger = logging.getLogger(__name__)


# ============================================================================
# OAUTH STATE STORE (CSRF Protection)
# ============================================================================

class OAuthStateStore:
    """
    In-memory store for OAuth state tokens with TTL.

    Provides CSRF protection by validating that the state parameter
    returned from OAuth provider matches one we generated.
    """

    def __init__(self, ttl_seconds: int = 600):  # 10 minute TTL
        self._states: Dict[str, datetime] = {}
        self._lock = threading.Lock()
        self._ttl = timedelta(seconds=ttl_seconds)

    def generate_state(self) -> str:
        """Generate a new state token and store it."""
        state = secrets.token_urlsafe(32)
        with self._lock:
            # Clean up expired states first
            self._cleanup_expired()
            self._states[state] = datetime.utcnow()
        return state

    def validate_and_consume(self, state: str) -> bool:
        """
        Validate a state token and remove it from the store.

        Returns True if valid, False if invalid or expired.
        State tokens are single-use (consumed on validation).
        """
        if not state:
            return False

        with self._lock:
            self._cleanup_expired()

            if state not in self._states:
                logger.warning(f"OAuth state validation failed: unknown state")
                return False

            # Consume the state (single-use)
            created_at = self._states.pop(state)
            age = datetime.utcnow() - created_at

            if age > self._ttl:
                logger.warning(f"OAuth state validation failed: state expired (age: {age})")
                return False

            return True

    def _cleanup_expired(self):
        """Remove expired states (call with lock held)."""
        now = datetime.utcnow()
        expired = [
            state for state, created_at in self._states.items()
            if now - created_at > self._ttl
        ]
        for state in expired:
            del self._states[state]


class RedisOAuthStateStore:
    """Redis-backed OAuth state store with TTL for multi-instance deployments."""

    def __init__(self, redis_url: str, ttl_seconds: int = 600):
        try:
            import redis
        except ImportError as e:
            raise RuntimeError("redis package is required for RedisOAuthStateStore") from e

        self._ttl_seconds = ttl_seconds
        self._redis = redis.Redis.from_url(redis_url, decode_responses=True)

    def generate_state(self) -> str:
        state = secrets.token_urlsafe(32)
        key = f"oauth_state:{state}"
        self._redis.setex(key, self._ttl_seconds, "1")
        return state

    def validate_and_consume(self, state: str) -> bool:
        if not state:
            return False
        key = f"oauth_state:{state}"
        value = self._redis.get(key)
        if not value:
            logger.warning("OAuth state validation failed: unknown state")
            return False
        # Consume state (best-effort)
        try:
            self._redis.delete(key)
        except Exception:
            pass
        return True


# Global OAuth state store
_redis_url = os.getenv("REDIS_URL")
if _redis_url:
    try:
        oauth_state_store = RedisOAuthStateStore(_redis_url)
        logger.info("OAuth state store: Redis")
    except Exception as e:
        logger.warning(f"Failed to initialize Redis OAuth state store: {e}. Falling back to in-memory.")
        oauth_state_store = OAuthStateStore()
else:
    oauth_state_store = OAuthStateStore()

# ============================================================================
# CONFIGURATION
# ============================================================================

# JWT Configuration
# Persist a stable secret key so tokens survive server restarts.
# Priority: env var > file-based key > generate-and-save
def _get_or_create_secret_key() -> str:
    """Return a stable JWT secret key that persists across restarts."""
    env_key = os.getenv("JWT_SECRET_KEY")
    if env_key:
        return env_key

    key_file = os.path.join(os.path.dirname(__file__), "..", ".jwt_secret")
    key_file = os.path.abspath(key_file)

    if os.path.exists(key_file):
        with open(key_file, "r") as f:
            return f.read().strip()

    # First run — generate and persist
    new_key = secrets.token_urlsafe(32)
    try:
        with open(key_file, "w") as f:
            f.write(new_key)
        os.chmod(key_file, 0o600)  # Owner-only read/write
        logger.info("Generated and saved new JWT secret key")
    except OSError as e:
        logger.warning(f"Could not persist JWT secret key to file: {e}")
    return new_key

SECRET_KEY = _get_or_create_secret_key()
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))

# OAuth Configuration
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GITHUB_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID", "")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET", "")

# OAuth redirect URLs
OAUTH_REDIRECT_URL = os.getenv("OAUTH_REDIRECT_URL", "http://localhost:5173/auth/callback")

# Security scheme for FastAPI
security = HTTPBearer(auto_error=False)

# Cookie settings (secure-by-default; override for local HTTP development if needed)
AUTH_COOKIE_SECURE = os.getenv("AUTH_COOKIE_SECURE", "true").lower() in ("true", "1", "yes")
AUTH_COOKIE_SAMESITE = os.getenv("AUTH_COOKIE_SAMESITE", "lax")  # "lax", "strict", "none"
AUTH_COOKIE_DOMAIN = os.getenv("AUTH_COOKIE_DOMAIN")  # optional


def require_verified_accounts() -> bool:
    """Return whether verified accounts are required for authenticated access."""
    return os.getenv("AUTH_REQUIRE_VERIFIED", "false").lower() in ("true", "1", "yes")


# ============================================================================
# PASSWORD UTILITIES
# ============================================================================

def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    try:
        return bcrypt.checkpw(
            plain_password.encode('utf-8'),
            hashed_password.encode('utf-8')
        )
    except Exception as e:
        logger.error(f"Password verification error: {e}")
        return False


# ============================================================================
# JWT UTILITIES
# ============================================================================

def create_access_token(
    user_id: int,
    email: str,
    expires_delta: Optional[timedelta] = None
) -> str:
    """Create a JWT access token."""
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode = {
        "sub": str(user_id),
        "email": email,
        "type": "access",
        "exp": expire,
        "iat": datetime.utcnow()
    }

    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def create_refresh_token(
    user_id: int,
    expires_delta: Optional[timedelta] = None
) -> Tuple[str, datetime]:
    """
    Create a refresh token.

    Returns:
        Tuple of (token_string, expiration_datetime)
    """
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

    # Generate a random token
    token = secrets.token_urlsafe(64)

    to_encode = {
        "sub": str(user_id),
        "token": token,
        "type": "refresh",
        "exp": expire,
        "iat": datetime.utcnow()
    }

    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt, expire


def decode_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Decode and validate a JWT token.

    Returns:
        Token payload or None if invalid
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError as e:
        logger.debug(f"Token decode error: {e}")
        return None


def hash_token(token: str) -> str:
    """Create a hash of a token for storage."""
    return hashlib.sha256(token.encode()).hexdigest()


# ============================================================================
# TOKEN PAIR GENERATION
# ============================================================================

def create_token_pair(
    user_id: int,
    email: str,
    db: Session,
    device_info: Optional[str] = None,
    ip_address: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create both access and refresh tokens.

    Returns:
        Dictionary with access_token, refresh_token, and metadata
    """
    # Create access token
    access_token = create_access_token(user_id, email)

    # Create refresh token
    refresh_token, refresh_expires = create_refresh_token(user_id)

    # Store refresh token hash in database
    RefreshTokenCRUD.create(
        db=db,
        user_id=user_id,
        token_hash=hash_token(refresh_token),
        expires_at=refresh_expires,
        device_info=device_info,
        ip_address=ip_address
    )

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60  # seconds
    }


def set_auth_cookies(response: Response, tokens: Dict[str, Any]) -> None:
    """Set HttpOnly auth cookies for access and refresh tokens."""
    access_max_age = ACCESS_TOKEN_EXPIRE_MINUTES * 60
    refresh_max_age = REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60

    response.set_cookie(
        key="access_token",
        value=tokens["access_token"],
        max_age=access_max_age,
        httponly=True,
        secure=AUTH_COOKIE_SECURE,
        samesite=AUTH_COOKIE_SAMESITE,
        domain=AUTH_COOKIE_DOMAIN,
        path="/"
    )
    response.set_cookie(
        key="refresh_token",
        value=tokens["refresh_token"],
        max_age=refresh_max_age,
        httponly=True,
        secure=AUTH_COOKIE_SECURE,
        samesite=AUTH_COOKIE_SAMESITE,
        domain=AUTH_COOKIE_DOMAIN,
        path="/"
    )


def clear_auth_cookies(response: Response) -> None:
    """Clear auth cookies on logout."""
    response.delete_cookie("access_token", path="/", domain=AUTH_COOKIE_DOMAIN)
    response.delete_cookie("refresh_token", path="/", domain=AUTH_COOKIE_DOMAIN)


# ============================================================================
# AUTHENTICATION DEPENDENCIES
# ============================================================================

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
    request: Request = None
) -> User:
    """
    FastAPI dependency to get the current authenticated user.

    Raises:
        HTTPException: If authentication fails
    """
    token = None
    if credentials:
        token = credentials.credentials
    elif request is not None:
        token = request.cookies.get("access_token")

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"}
        )
    payload = decode_token(token)

    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"}
        )

    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type",
            headers={"WWW-Authenticate": "Bearer"}
        )

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"}
        )

    user = UserCRUD.get_by_id(db, int(user_id))
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"}
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled"
        )

    if require_verified_accounts() and not user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email verification required"
        )

    return user


async def get_current_user_optional(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
    request: Request = None
) -> Optional[User]:
    """
    FastAPI dependency to optionally get the current user.
    Returns None if not authenticated (doesn't raise exception).
    """
    if not credentials and request is None:
        return None

    try:
        return await get_current_user(credentials, db, request)
    except HTTPException:
        return None


# ============================================================================
# OAUTH UTILITIES
# ============================================================================

async def get_google_user_info(access_token: str) -> Optional[Dict[str, Any]]:
    """Fetch user info from Google using access token."""
    import httpx

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://www.googleapis.com/oauth2/v2/userinfo",
                headers={"Authorization": f"Bearer {access_token}"}
            )

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Google API error: {response.status_code}")
                return None
    except Exception as e:
        logger.error(f"Error fetching Google user info: {e}")
        return None


async def get_github_user_info(access_token: str) -> Optional[Dict[str, Any]]:
    """Fetch user info from GitHub using access token."""
    import httpx

    try:
        async with httpx.AsyncClient() as client:
            # Get user info
            response = await client.get(
                "https://api.github.com/user",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Accept": "application/vnd.github.v3+json"
                }
            )

            if response.status_code != 200:
                logger.error(f"GitHub API error: {response.status_code}")
                return None

            user_data = response.json()

            # Get primary email if not public
            if not user_data.get("email"):
                email_response = await client.get(
                    "https://api.github.com/user/emails",
                    headers={
                        "Authorization": f"Bearer {access_token}",
                        "Accept": "application/vnd.github.v3+json"
                    }
                )

                if email_response.status_code == 200:
                    emails = email_response.json()
                    primary_email = next(
                        (e["email"] for e in emails if e.get("primary")),
                        emails[0]["email"] if emails else None
                    )
                    user_data["email"] = primary_email

            return user_data
    except Exception as e:
        logger.error(f"Error fetching GitHub user info: {e}")
        return None


async def exchange_google_code(code: str, redirect_uri: str) -> Optional[str]:
    """Exchange Google authorization code for access token."""
    import httpx

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "client_id": GOOGLE_CLIENT_ID,
                    "client_secret": GOOGLE_CLIENT_SECRET,
                    "code": code,
                    "grant_type": "authorization_code",
                    "redirect_uri": redirect_uri
                }
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("access_token")
            else:
                logger.error(f"Google token exchange error: {response.text}")
                return None
    except Exception as e:
        logger.error(f"Error exchanging Google code: {e}")
        return None


async def exchange_github_code(code: str) -> Optional[str]:
    """Exchange GitHub authorization code for access token."""
    import httpx

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://github.com/login/oauth/access_token",
                headers={"Accept": "application/json"},
                data={
                    "client_id": GITHUB_CLIENT_ID,
                    "client_secret": GITHUB_CLIENT_SECRET,
                    "code": code
                }
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("access_token")
            else:
                logger.error(f"GitHub token exchange error: {response.text}")
                return None
    except Exception as e:
        logger.error(f"Error exchanging GitHub code: {e}")
        return None


def get_google_auth_url(state: str, redirect_uri: str) -> str:
    """Generate Google OAuth authorization URL."""
    from urllib.parse import urlencode

    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "access_type": "offline",
        "prompt": "consent"
    }

    return f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}"


def get_github_auth_url(state: str) -> str:
    """Generate GitHub OAuth authorization URL."""
    from urllib.parse import urlencode

    params = {
        "client_id": GITHUB_CLIENT_ID,
        "scope": "user:email",
        "state": state
    }

    return f"https://github.com/login/oauth/authorize?{urlencode(params)}"


def generate_oauth_state(prefix: Optional[str] = None) -> str:
    """Generate and store a new OAuth state token for CSRF protection.

    If prefix is provided (e.g., "google" or "github"), it is prepended
    for client-side provider detection: "{prefix}:{token}".
    """
    token = oauth_state_store.generate_state()
    if prefix:
        return f"{prefix}:{token}"
    return token


def validate_oauth_state(state: str) -> bool:
    """
    Validate an OAuth state token.

    This should be called in OAuth callback endpoints to verify
    that the state parameter matches one we generated.

    Returns True if valid, False otherwise.
    The state is consumed (single-use) upon successful validation.
    """
    if not state:
        return False
    # Allow optional provider prefix (e.g., "google:..." or "github:...")
    if ":" in state:
        _, token = state.split(":", 1)
        return oauth_state_store.validate_and_consume(token)
    return oauth_state_store.validate_and_consume(state)
