"""
Phase 2 — Authentication & RBAC tests.

All tests use an in-memory SQLite database and a fakeredis instance —
no external services required.
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import fakeredis

from app.core.security import hash_password
from app.database import get_db
from app.core.deps import get_redis
from app.main import app
from app.models.base import Base
from app.models.user import User, UserRole

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

TEST_DB_URL = "sqlite://"  # in-memory


@pytest.fixture(scope="function")
def db_session():
    # StaticPool ensures all connections use the same in-memory SQLite
    # connection — without it, post-commit connections see an empty database.
    engine = create_engine(
        TEST_DB_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    import app.models  # noqa: F401 — register all models before create_all
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)


@pytest.fixture(scope="function")
def redis_client():
    return fakeredis.FakeRedis(decode_responses=True)


@pytest.fixture(scope="function")
def client(db_session, redis_client):
    app.dependency_overrides[get_db] = lambda: db_session
    app.dependency_overrides[get_redis] = lambda: redis_client
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def _make_user(db_session, email: str, password: str, role: UserRole) -> User:
    user = User(
        email=email,
        hashed_password=hash_password(password),
        full_name="Test User",
        role=role,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_login_valid(client, db_session):
    _make_user(db_session, "admin@test.com", "securepassword123", UserRole.ADMIN)
    response = client.post(
        "/api/v1/auth/login",
        json={"email": "admin@test.com", "password": "securepassword123"},
    )
    assert response.status_code == 200
    body = response.json()
    assert "access_token" in body
    assert "refresh_token" in body
    assert body["token_type"] == "bearer"


def test_login_wrong_password(client, db_session):
    _make_user(db_session, "user@test.com", "correctpassword123", UserRole.DOCTOR)
    response = client.post(
        "/api/v1/auth/login",
        json={"email": "user@test.com", "password": "wrongpassword"},
    )
    assert response.status_code == 401


def test_protected_endpoint_without_token(client):
    response = client.get("/api/v1/auth/me")
    assert response.status_code == 401


def test_me_with_valid_token(client, db_session):
    _make_user(db_session, "doc@test.com", "securepassword123", UserRole.DOCTOR)
    login = client.post(
        "/api/v1/auth/login",
        json={"email": "doc@test.com", "password": "securepassword123"},
    )
    token = login.json()["access_token"]
    response = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.json()["email"] == "doc@test.com"


def test_doctor_cannot_access_admin_endpoint(client, db_session):
    _make_user(db_session, "doc@test.com", "securepassword123", UserRole.DOCTOR)
    login = client.post(
        "/api/v1/auth/login",
        json={"email": "doc@test.com", "password": "securepassword123"},
    )
    token = login.json()["access_token"]
    response = client.get(
        "/api/v1/users/",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 403


def test_token_refresh(client, db_session):
    _make_user(db_session, "admin@test.com", "securepassword123", UserRole.ADMIN)
    login = client.post(
        "/api/v1/auth/login",
        json={"email": "admin@test.com", "password": "securepassword123"},
    )
    refresh_token = login.json()["refresh_token"]

    response = client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": refresh_token},
    )
    assert response.status_code == 200
    body = response.json()
    assert "access_token" in body
    assert "refresh_token" in body
    # New refresh token must differ from old one (rotation)
    assert body["refresh_token"] != refresh_token


def test_logout_blocks_refresh_token(client, db_session):
    _make_user(db_session, "admin@test.com", "securepassword123", UserRole.ADMIN)
    login = client.post(
        "/api/v1/auth/login",
        json={"email": "admin@test.com", "password": "securepassword123"},
    )
    refresh_token = login.json()["refresh_token"]

    # Logout
    client.post("/api/v1/auth/logout", json={"refresh_token": refresh_token})

    # Second use of same refresh_token must be rejected
    response = client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": refresh_token},
    )
    assert response.status_code == 401


def test_admin_can_create_user(client, db_session):
    _make_user(db_session, "admin@test.com", "securepassword123", UserRole.ADMIN)
    login = client.post(
        "/api/v1/auth/login",
        json={"email": "admin@test.com", "password": "securepassword123"},
    )
    token = login.json()["access_token"]
    response = client.post(
        "/api/v1/users/",
        json={
            "email": "newdoc@hospital.com",
            "password": "doctorpassword123",
            "full_name": "Dr. New",
            "role": "DOCTOR",
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 201
    assert response.json()["must_change_password"] is True
