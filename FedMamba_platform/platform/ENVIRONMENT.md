# FedMamba-SALT Clinical Platform — Environment Specification

> **Scope:** This document is the single source of truth for every dependency, tool, service, configuration, and runtime requirement. Assume nothing exists unless defined here.

---

## 1. Host Machine Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| OS | Windows 10 21H2+ / Ubuntu 22.04 / macOS 13 | Windows 11 / Ubuntu 24.04 |
| CPU | 4 cores | 8+ cores |
| RAM | 8 GB | 16 GB |
| Disk | 20 GB free | 50 GB free |
| Network | Internet access for package downloads | — |

---

## 2. Required Software (Install in This Order)

### 2.1 Python

| Field | Value |
|-------|-------|
| Version | **3.10.x** (tested: 3.10.0) |
| Download | https://www.python.org/downloads/release/python-3100/ |
| Verify | `python --version` → `Python 3.10.x` |
| Note | Must be on system PATH. `pip` bundled. |

### 2.2 Node.js

| Field | Value |
|-------|-------|
| Version | **20.x LTS** |
| Download | https://nodejs.org/en/download/ |
| Verify | `node --version` → `v20.x.x` |
| Bundled | `npm` ≥ 10.x (ships with Node 20) |

### 2.3 Docker Desktop (for containerized deployment)

| Field | Value |
|-------|-------|
| Version | **Docker Engine 24+**, Compose V2 |
| Download | https://www.docker.com/products/docker-desktop/ |
| Verify | `docker --version` and `docker compose version` |
| Windows | Enable WSL 2 backend |

### 2.4 Git

| Field | Value |
|-------|-------|
| Version | 2.40+ |
| Verify | `git --version` |

---

## 3. Infrastructure Services

All services run inside Docker Compose. No manual installation needed if Docker is available.

### 3.1 PostgreSQL (Production Database)

| Field | Value |
|-------|-------|
| Image | `postgres:16-alpine` |
| Port | `5432` |
| Database | `fedmamba` |
| User | `fedmamba` |
| Password | Set via `POSTGRES_PASSWORD` env var |
| Volume | Named volume `pgdata` → `/var/lib/postgresql/data` |
| Health check | `pg_isready -U fedmamba` every 5s, 5 retries |

### 3.2 SQLite (Local Dev Alternative)

| Field | Value |
|-------|-------|
| File | `platform/backend/dev.db` (auto-created) |
| Usage | Set `DATABASE_URL=sqlite:///./dev.db` in `.env` |
| Note | No install needed — bundled with Python |

### 3.3 Redis

| Field | Value |
|-------|-------|
| Image | `redis:7-alpine` |
| Port | `6379` |
| Purpose | Celery broker + result backend |
| Health check | `redis-cli ping` every 5s, 5 retries |

### 3.4 Nginx (Frontend Production Serving)

| Field | Value |
|-------|-------|
| Image | `nginx:alpine` (inside frontend Dockerfile) |
| Port | `80` (mapped to host `3000`) |
| Purpose | Serves built React app, proxies `/api` and `/health` to backend |

---

## 4. Backend Dependencies (Python)

Defined in [pyproject.toml](file:///c:/Users/ZIAD/OneDrive%20-%20Alexandria%20National%20University/FedMamba/platform/backend/pyproject.toml).

### 4.1 Runtime Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `fastapi` | ≥ 0.111.0 | Web framework |
| `uvicorn[standard]` | ≥ 0.30.0 | ASGI server (with `uvloop`, `httptools`, `websockets`) |
| `sqlalchemy` | ≥ 2.0.30 | ORM and database toolkit |
| `pydantic-settings` | ≥ 2.3.0 | Typed configuration from env vars |
| `celery[redis]` | ≥ 5.4.0 | Distributed task queue |
| `httpx` | ≥ 0.27.0 | Async HTTP client |
| `python-dotenv` | ≥ 1.0.0 | `.env` file loading |
| `sse-starlette` | ≥ 2.1.0 | Server-Sent Events for real-time updates |
| `alembic` | ≥ 1.13.0 | Database migrations |
| `psycopg2-binary` | ≥ 2.9.9 | PostgreSQL adapter |
| `aiosqlite` | ≥ 0.20.0 | Async SQLite driver for dev |

### 4.2 Dev / Test Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `pytest` | ≥ 8.2.0 | Test runner |
| `pytest-asyncio` | ≥ 0.23.0 | Async test support |

### 4.3 Research/ML Dependencies (Required by Backend)

| Package | Version | Purpose |
|---------|---------|---------|
| `timm` | == 0.3.2 | Teacher backbone and utilities |
| `einops` | latest | Tensor operations |
| `PyYAML` | latest | Config parsing |
| `matplotlib`, `seaborn` | latest | Visualization |
| `pandas`, `tqdm` | latest | Data processing and progress bars |
| `scikit-image`, `scikit-learn` | latest | Image processing and ML metrics |
| `causal-conv1d` | ≥ 1.4.0 | FedMamba core component (manual install) |
| `mamba-ssm` | latest | FedMamba core component (manual install) |

### 4.4 Install Commands

```bash
# Option A: Conda (Recommended)
cd platform
conda env create -f environment.yml
conda activate fedmamba-salt

# Mamba dependencies MUST be installed manually after env creation
pip install causal-conv1d>=1.4.0 --no-build-isolation
pip install mamba-ssm --no-build-isolation
```
# Option A: editable install (recommended for dev)
cd platform/backend
pip install -e ".[dev]"

# Option B: direct install
pip install fastapi "uvicorn[standard]" sqlalchemy pydantic-settings \
    "celery[redis]" httpx python-dotenv sse-starlette alembic \
    psycopg2-binary aiosqlite pytest pytest-asyncio
```

---

## 5. Frontend Dependencies (Node.js)

Defined in [package.json](file:///c:/Users/ZIAD/OneDrive%20-%20Alexandria%20National%20University/FedMamba/platform/frontend/package.json).

### 5.1 Runtime Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `react` | ^18.3.1 | UI library |
| `react-dom` | ^18.3.1 | React DOM renderer |

### 5.2 Dev Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `@types/react` | ^18.3.3 | React type definitions |
| `@types/react-dom` | ^18.3.0 | ReactDOM type definitions |
| `@vitejs/plugin-react` | ^4.3.1 | Vite React integration (Babel) |
| `typescript` | ^5.5.0 | TypeScript compiler |
| `vite` | ^5.4.0 | Build tool and dev server |

### 5.3 Install Command

```bash
cd platform/frontend
npm install
```

---

## 6. Environment Variables

All variables are **required**. No defaults are hardcoded in application code.

Defined in [.env.example](file:///c:/Users/ZIAD/OneDrive%20-%20Alexandria%20National%20University/FedMamba/platform/.env.example). Copy to `platform/.env` (Docker) or `platform/backend/.env` (local dev).

| Variable | Type | Example (Docker) | Example (Local Dev) | Purpose |
|----------|------|-------------------|---------------------|---------|
| `DATABASE_URL` | string | `postgresql://fedmamba:changeme@postgres:5432/fedmamba` | `sqlite:///./dev.db` | SQLAlchemy connection string |
| `REDIS_URL` | string | `redis://redis:6379/0` | `redis://localhost:6379/0` | Celery broker + result backend |
| `SECRET_KEY` | string | 64-char random string | `dev-secret-key-not-for-production` | JWT signing, session encryption |
| `ALLOWED_ORIGINS` | JSON list | `["http://localhost:3000"]` | `["http://localhost:3000","http://localhost:5173"]` | CORS allowed origins |
| `RESEARCH_ROOT` | path | `/repo` | `c:/Users/.../FedMamba` | Absolute path to repo root |
| `FL_OUTPUT_DIR` | path | `/repo/output` | `c:/Users/.../FedMamba/output` | Where research scripts write outputs |
| `POSTGRES_PASSWORD` | string | `changeme` | — | Used by docker-compose for postgres service |

---

## 7. Network Ports

| Port | Service | Protocol | Context |
|------|---------|----------|---------|
| 3000 | Frontend (Vite dev / Nginx prod) | HTTP | Browser access |
| 5173 | Frontend (Vite dev fallback) | HTTP | Alternative Vite port |
| 8000 | Backend (Uvicorn) | HTTP | API + health check |
| 5432 | PostgreSQL | TCP | Database connections |
| 6379 | Redis | TCP | Celery broker |

---

## 8. Docker Container Specifications

### 8.1 Backend Container

| Field | Value |
|-------|-------|
| Base image | `python:3.10-slim` |
| System packages | `build-essential`, `libpq-dev` |
| Working directory | `/app` |
| Exposed port | `8000` |
| Entrypoint | `uvicorn app.main:app --host 0.0.0.0 --port 8000` |
| Dockerfile | [backend/Dockerfile](file:///c:/Users/ZIAD/OneDrive%20-%20Alexandria%20National%20University/FedMamba/platform/backend/Dockerfile) |

### 8.2 Celery Worker Container

| Field | Value |
|-------|-------|
| Base image | Same build as backend |
| Command override | `celery -A celery_worker.celery_app worker --loglevel=info` |
| Depends on | postgres (healthy), redis (healthy) |

### 8.3 Frontend Container

| Field | Value |
|-------|-------|
| Build stage | `node:20-alpine` |
| Serve stage | `nginx:alpine` |
| Build output | `/app/dist` → `/usr/share/nginx/html` |
| Exposed port | `80` (mapped to host `3000`) |
| Nginx config | Inline heredoc; proxies `/health` and `/api` to `http://backend:8000` |
| Dockerfile | [frontend/Dockerfile](file:///c:/Users/ZIAD/OneDrive%20-%20Alexandria%20National%20University/FedMamba/platform/frontend/Dockerfile) |

---

## 9. Docker Compose Services

Defined in [docker-compose.yml](file:///c:/Users/ZIAD/OneDrive%20-%20Alexandria%20National%20University/FedMamba/platform/docker-compose.yml).

| Service | Image / Build | Ports | Depends On | Volumes |
|---------|---------------|-------|------------|---------|
| `postgres` | `postgres:16-alpine` | 5432:5432 | — | `pgdata:/var/lib/postgresql/data` |
| `redis` | `redis:7-alpine` | 6379:6379 | — | — |
| `backend` | Build `./backend` | 8000:8000 | postgres ✓, redis ✓ | — |
| `celery_worker` | Build `./backend` | — | postgres ✓, redis ✓ | — |
| `frontend` | Build `./frontend` | 3000:80 | backend | — |

Dev override ([docker-compose.dev.yml](file:///c:/Users/ZIAD/OneDrive%20-%20Alexandria%20National%20University/FedMamba/platform/docker-compose.dev.yml)) adds:
- Backend: `--reload` flag, source volume mount
- Frontend: `npm run dev --host 0.0.0.0`, source volume mount (excludes `node_modules`)

---

## 10. Runtime Configurations

### 10.1 Vite Dev Server

| Field | Value |
|-------|-------|
| Port | 3000 |
| Proxy `/health` | → `http://localhost:8000` |
| Proxy `/api` | → `http://localhost:8000` |
| Config file | [vite.config.ts](file:///c:/Users/ZIAD/OneDrive%20-%20Alexandria%20National%20University/FedMamba/platform/frontend/vite.config.ts) |

### 10.2 TypeScript Compiler

| Field | Value |
|-------|-------|
| Target | ES2020 |
| Module | ESNext |
| JSX | react-jsx |
| Strict mode | Enabled |
| Config file | [tsconfig.json](file:///c:/Users/ZIAD/OneDrive%20-%20Alexandria%20National%20University/FedMamba/platform/frontend/tsconfig.json) |

### 10.3 Pytest

| Field | Value |
|-------|-------|
| Async mode | `auto` (no explicit `@pytest.mark.asyncio` needed) |
| Config location | `[tool.pytest.ini_options]` in `pyproject.toml` |

---

## 11. Startup Commands

### Local Development (no Docker)

```bash
# Terminal 1 — Backend
cd platform/backend
pip install -e ".[dev]"          # first time only
uvicorn app.main:app --reload --port 8000

# Terminal 2 — Frontend
cd platform/frontend
npm install                      # first time only
npm run dev

# Terminal 3 — Celery Worker (requires Redis running)
cd platform/backend
celery -A celery_worker.celery_app worker --loglevel=info
```

### Docker Compose (Production-like)

```bash
cd platform
cp .env.example .env             # edit values
docker compose up --build -d

# Dev mode (hot reload)
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

### Run Tests

```bash
cd platform/backend
python -m pytest tests/ -v
```

---

## 12. Verification Checklist

| # | Check | Command | Expected |
|---|-------|---------|----------|
| 1 | Python version | `python --version` | `Python 3.10.x` |
| 2 | Node version | `node --version` | `v20.x.x` |
| 3 | Docker version | `docker --version` | `Docker version 24+` |
| 4 | Backend health | `curl http://localhost:8000/health` | `{"status":"ok","version":"0.1.0"}` |
| 5 | Smoke test | `cd platform/backend && python -m pytest tests/ -v` | `1 passed` |
| 6 | Frontend loads | Browser → `http://localhost:3000` | Displays "Status: ok" |
| 7 | Postgres connects | `docker compose exec postgres pg_isready -U fedmamba` | `accepting connections` |
| 8 | Redis responds | `docker compose exec redis redis-cli ping` | `PONG` |

---

## 13. File Layout Reference

```
platform/
├── .env.example                      # All env vars documented
├── docker-compose.yml                # Production service definitions
├── docker-compose.dev.yml            # Hot-reload overrides
├── README.md                         # Setup instructions
├── ENVIRONMENT.md                    # This file
├── backend/
│   ├── .env                          # Local dev env vars (git-ignored)
│   ├── pyproject.toml                # Python project + dependencies
│   ├── Dockerfile                    # Python 3.10-slim container
│   ├── celery_worker.py              # Celery app entry point
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                   # FastAPI app factory + /health
│   │   ├── config.py                 # Pydantic BaseSettings
│   │   ├── database.py               # SQLAlchemy engine + session
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   └── v1/
│   │   │       ├── __init__.py
│   │   │       └── router.py         # V1 API router
│   │   ├── core/__init__.py
│   │   ├── models/__init__.py        # SQLAlchemy ORM models
│   │   ├── schemas/__init__.py       # Pydantic schemas
│   │   ├── services/__init__.py      # Business logic
│   │   └── tasks/__init__.py         # Celery task definitions
│   └── tests/
│       └── test_health.py            # Smoke test
└── frontend/
    ├── package.json                  # Node.js dependencies
    ├── vite.config.ts                # Vite + proxy config
    ├── tsconfig.json                 # TypeScript config
    ├── index.html                    # HTML shell
    ├── Dockerfile                    # Node build + nginx serve
    ├── public/
    └── src/
        ├── main.tsx                  # React entry point
        ├── App.tsx                   # Health check display
        ├── vite-env.d.ts             # Vite type declarations
        ├── pages/
        ├── components/
        ├── hooks/
        ├── stores/
        └── api/
```
