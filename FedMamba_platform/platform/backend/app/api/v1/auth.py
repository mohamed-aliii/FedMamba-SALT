"""
Authentication endpoints.

Rate limiting: POST /auth/login is limited to 5 requests/minute per IP.
Refresh token rotation: each refresh issues a new token pair and invalidates the old one.
"""
from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

import redis as redis_lib

from app.core.deps import get_current_user, get_redis
from app.database import get_db
from app.models.user import User
from app.schemas.auth import (
    ChangePasswordRequest,
    LoginRequest,
    RefreshRequest,
    TokenResponse,
)
from app.schemas.user import UserRead
from app.services import auth_service

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
def login(
    body: LoginRequest,
    request: Request,
    db: Session = Depends(get_db),
    redis: redis_lib.Redis = Depends(get_redis),
):
    access_token, refresh_token = auth_service.login(
        db, redis, body.email, body.password, request
    )
    return TokenResponse(access_token=access_token, refresh_token=refresh_token)


@router.post("/refresh", response_model=TokenResponse)
def refresh(
    body: RefreshRequest,
    db: Session = Depends(get_db),
    redis: redis_lib.Redis = Depends(get_redis),
):
    access_token, refresh_token = auth_service.refresh_tokens(
        redis, db, body.refresh_token
    )
    return TokenResponse(access_token=access_token, refresh_token=refresh_token)


@router.post("/logout", status_code=204)
def logout(
    body: RefreshRequest,
    redis: redis_lib.Redis = Depends(get_redis),
):
    auth_service.logout(redis, body.refresh_token)


@router.get("/me", response_model=UserRead)
def me(current_user: User = Depends(get_current_user)):
    return current_user


@router.post("/change-password", status_code=204)
def change_password(
    body: ChangePasswordRequest,
    request: Request,
    db: Session = Depends(get_db),
    redis: redis_lib.Redis = Depends(get_redis),
    current_user: User = Depends(get_current_user),
):
    auth_service.change_password(
        db, redis, current_user, body.current_password, body.new_password, request
    )
