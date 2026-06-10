"""
Authentication business logic. Routers are thin — all logic lives here.
"""
import uuid
from datetime import timedelta

import redis as redis_lib
from fastapi import HTTPException, Request, status
from sqlalchemy.orm import Session

from app.config import get_settings
from app.core.audit import log_action
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)
from app.models.user import User

settings = get_settings()

# Redis key prefixes
_LOCKOUT_PREFIX = "auth:lockout:"
_REFRESH_PREFIX = "auth:refresh:"
_BLOCKLIST_REFRESH = "blocklist:refresh:"

_MAX_ATTEMPTS = 5
_LOCKOUT_SECONDS = 15 * 60  # 15 minutes


def _lockout_key(email: str) -> str:
    return f"{_LOCKOUT_PREFIX}{email}"


def _refresh_key(jti: str) -> str:
    return f"{_REFRESH_PREFIX}{jti}"


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------

def authenticate_user(
    db: Session,
    redis: redis_lib.Redis,
    email: str,
    password: str,
) -> User:
    """Verify credentials and enforce lockout policy.

    Raises HTTPException 401 on bad credentials or 429 on lockout.
    Never reveals which of email/password was wrong.
    """
    lockout_key = _lockout_key(email)
    attempts = redis.get(lockout_key)
    if attempts and int(attempts) >= _MAX_ATTEMPTS:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Account temporarily locked. Try again later.",
        )

    user = db.query(User).filter(User.email == email).first()
    if user is None or not verify_password(password, user.hashed_password):
        # Increment lockout counter
        pipe = redis.pipeline()
        pipe.incr(lockout_key)
        pipe.expire(lockout_key, _LOCKOUT_SECONDS)
        pipe.execute()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Account is inactive",
        )

    # Successful login — reset lockout counter
    redis.delete(lockout_key)
    return user


def _issue_tokens(user: User, redis: redis_lib.Redis) -> tuple[str, str]:
    """Create and store access + refresh token pair."""
    token_data = {
        "sub": str(user.id),
        "role": user.role.value,
        "hospital_id": str(user.hospital_id) if user.hospital_id else None,
    }
    access_token = create_access_token(token_data)
    refresh_token, jti = create_refresh_token(token_data)

    # Store jti in Redis with TTL equal to refresh token lifetime
    ttl = settings.REFRESH_TOKEN_EXPIRE_DAYS * 86400
    redis.set(_refresh_key(jti), str(user.id), ex=ttl)

    return access_token, refresh_token


def login(
    db: Session,
    redis: redis_lib.Redis,
    email: str,
    password: str,
    request: Request | None = None,
) -> tuple[str, str]:
    """Authenticate and return (access_token, refresh_token)."""
    user = authenticate_user(db, redis, email, password)
    access_token, refresh_token = _issue_tokens(user, redis)
    log_action(
        db,
        user_id=user.id,
        action="LOGIN",
        resource_type="User",
        resource_id=str(user.id),
        request=request,
    )
    return access_token, refresh_token


# ---------------------------------------------------------------------------
# Refresh (sliding window with rotation)
# ---------------------------------------------------------------------------

def refresh_tokens(
    redis: redis_lib.Redis,
    db: Session,
    refresh_token: str,
) -> tuple[str, str]:
    """Validate a refresh token, rotate it, return a new pair.

    Raises 401 if token is invalid, expired, blocklisted, or reused.
    Reuse detection: if jti is not in Redis (already rotated/revoked),
    treat as potential theft and raise 401.
    """
    invalid_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired refresh token",
    )

    payload = decode_token(refresh_token)
    if payload is None or payload.get("type") != "refresh":
        raise invalid_exc

    jti: str | None = payload.get("jti")
    user_id: str | None = payload.get("sub")
    if not jti or not user_id:
        raise invalid_exc

    # Check blocklist
    if redis.exists(_BLOCKLIST_REFRESH + jti):
        raise invalid_exc

    # Check jti still in active store (reuse detection)
    if not redis.exists(_refresh_key(jti)):
        raise invalid_exc

    try:
        user_uuid = uuid.UUID(user_id)
    except ValueError:
        raise invalid_exc

    user = db.get(User, user_uuid)
    if user is None or not user.is_active:
        raise invalid_exc

    # Invalidate old jti
    redis.delete(_refresh_key(jti))

    # Issue new pair
    return _issue_tokens(user, redis)


# ---------------------------------------------------------------------------
# Logout
# ---------------------------------------------------------------------------

def logout(redis: redis_lib.Redis, refresh_token: str) -> None:
    """Blocklist the refresh token's jti in Redis so it cannot be reused."""
    payload = decode_token(refresh_token)
    if payload is None:
        return  # Already invalid — no-op

    jti: str | None = payload.get("jti")
    if not jti:
        return

    # Remove from active store
    redis.delete(_refresh_key(jti))

    # Add to blocklist with same TTL as the token's remaining lifetime
    import time
    exp = payload.get("exp", 0)
    remaining = max(int(exp - time.time()), 1)
    redis.set(_BLOCKLIST_REFRESH + jti, "1", ex=remaining)


# ---------------------------------------------------------------------------
# Change password
# ---------------------------------------------------------------------------

def change_password(
    db: Session,
    redis: redis_lib.Redis,
    user: User,
    current_plain: str,
    new_plain: str,
    request: Request | None = None,
) -> None:
    """Verify current password, update hash, clear must_change_password flag."""
    if not verify_password(current_plain, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect",
        )
    if len(new_plain) < 12:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Password must be at least 12 characters",
        )
    if len(new_plain) > 128:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Password must not exceed 128 characters",
        )

    user.hashed_password = hash_password(new_plain)
    user.must_change_password = False
    db.commit()

    log_action(
        db,
        user_id=user.id,
        action="CHANGE_PASSWORD",
        resource_type="User",
        resource_id=str(user.id),
        request=request,
    )
