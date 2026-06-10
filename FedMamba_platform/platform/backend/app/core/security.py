"""
Core security utilities for the FedMamba-SALT Clinical Platform.

Handles: password hashing (bcrypt), JWT creation/decoding,
API key generation and hashing.
"""
import hashlib
import hmac
import secrets
import uuid
from datetime import datetime, timedelta, timezone

import bcrypt
from jose import JWTError, jwt

from app.config import get_settings

settings = get_settings()


# ---------------------------------------------------------------------------
# Password hashing (bcrypt, work factor 12)
# ---------------------------------------------------------------------------

def hash_password(plain: str) -> str:
    """Hash a plaintext password with bcrypt (work factor 12)."""
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt(rounds=12)).decode()


def verify_password(plain: str, hashed: str) -> bool:
    """Constant-time comparison of plain password against bcrypt hash."""
    return bcrypt.checkpw(plain.encode(), hashed.encode())


# ---------------------------------------------------------------------------
# JWT tokens
# ---------------------------------------------------------------------------

def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    """Create a short-lived JWT access token."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire, "type": "access"})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def create_refresh_token(data: dict) -> tuple[str, str]:
    """Create a long-lived JWT refresh token.

    Returns (token, jti) — jti is the unique token ID stored in Redis
    for rotation and revocation.
    """
    to_encode = data.copy()
    jti = str(uuid.uuid4())
    expire = datetime.now(timezone.utc) + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh", "jti": jti})
    token = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return token, jti


def decode_token(token: str) -> dict | None:
    """Decode and validate a JWT. Returns payload dict or None on failure."""
    try:
        return jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
    except JWTError:
        return None


# ---------------------------------------------------------------------------
# Hospital API key helpers (SHA-256, no HMAC — rotation-safe)
# ---------------------------------------------------------------------------

def generate_api_key() -> str:
    """Generate a new hospital API key.

    Format: sk_hosp_<43-char base64url-encoded random bytes>
    The raw key is shown ONCE at creation. Store only hash_api_key(key) in DB.
    """
    raw = secrets.token_urlsafe(32)
    return f"sk_hosp_{raw}"


def hash_api_key(key: str) -> str:
    """SHA-256 digest of an API key (hex string). Store this in the DB."""
    return hashlib.sha256(key.encode()).hexdigest()


def verify_api_key(presented_key: str, stored_hash: str) -> bool:
    """Constant-time comparison of presented key against stored SHA-256 hash."""
    candidate_hash = hash_api_key(presented_key)
    return hmac.compare_digest(candidate_hash, stored_hash)
