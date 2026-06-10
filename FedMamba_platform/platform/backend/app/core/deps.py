"""
FastAPI dependency functions for authentication and authorization.
"""
import uuid
from typing import Callable

import redis as redis_lib
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.config import get_settings
from app.core.security import decode_token
from app.database import get_db
from app.models.user import User, UserRole

settings = get_settings()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

_redis_client: redis_lib.Redis | None = None


def get_redis() -> redis_lib.Redis:
    """Return a shared Redis client (lazily initialised)."""
    global _redis_client
    if _redis_client is None:
        _redis_client = redis_lib.from_url(settings.REDIS_URL, decode_responses=True)
    return _redis_client


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
    redis: redis_lib.Redis = Depends(get_redis),
) -> User:
    """Decode and validate the access token; return the active User."""
    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    payload = decode_token(token)
    if payload is None or payload.get("type") != "access":
        raise credentials_exc

    user_id: str | None = payload.get("sub")
    if user_id is None:
        raise credentials_exc

    # Check if token's jti is on the access-token blocklist (set on logout)
    jti = payload.get("jti")
    if jti and redis.exists(f"blocklist:access:{jti}"):
        raise credentials_exc

    try:
        user_uuid = uuid.UUID(user_id)
    except ValueError:
        raise credentials_exc

    user = db.get(User, user_uuid)
    if user is None or not user.is_active:
        raise credentials_exc
    return user


def require_role(*roles: UserRole) -> Callable:
    """Dependency factory — raises 403 if the current user's role is not allowed."""
    def _check(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions",
            )
        return current_user
    return _check


# Convenience shortcuts
def get_current_admin(user: User = Depends(require_role(UserRole.ADMIN))) -> User:
    return user


def get_current_hospital_user(
    user: User = Depends(require_role(UserRole.HOSPITAL_MANAGER))
) -> User:
    return user


def get_current_doctor(user: User = Depends(require_role(UserRole.DOCTOR))) -> User:
    return user
