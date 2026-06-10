# FedMamba-SALT Clinical Platform

A production web application that wraps the FedMamba-SALT federated learning research codebase.

## Architecture

```
platform/
├── backend/     FastAPI + Celery (Python)
├── frontend/    React + Vite (TypeScript)
├── docker-compose.yml
└── .env.example
```

The platform communicates with the research codebase through:
- **Subprocess/task queue** — research scripts are CLI tools invoked via Celery tasks
- **Config files** — YAML configs the research scripts already support
- **Results watcher** — polling output directories for checkpoints and CSVs

> **Critical:** Research files (`models/`, `augmentations/`, `objectives/`, `eval/`, `utils/`, `train_*.py`) are never modified by the platform.

## Quick Start (Conda — Recommended)

```bash
cd platform
conda env create -f environment.yml
conda activate fedmamba-salt
```

This installs Python 3.10, Node.js 20, PostgreSQL 16, Redis 7, and all pip dependencies in one step.

### Post-Install: Mamba Dependencies

The FedMamba model requires specific order and isolation settings for its core SSM dependencies. **You must run these manually after creating the conda environment:**

```bash
# Install order matters: causal-conv1d before mamba-ssm.
# --no-build-isolation lets the build see the existing PyTorch/CUDA.
pip install causal-conv1d>=1.4.0 --no-build-isolation
pip install mamba-ssm --no-build-isolation
```

Then start services:

```bash
# Terminal 1 — Backend
cd platform/backend
uvicorn app.main:app --reload --port 8000

# Terminal 2 — Frontend
cd platform/frontend
npm install    # first time only
npm run dev

# Terminal 3 — Celery (optional, needs Redis running)
cd platform/backend
celery -A celery_worker.celery_app worker --loglevel=info
```

## Quick Start (Docker)

```bash
cp .env.example .env
# Edit .env with your values

docker-compose up --build -d

# Verify
curl http://localhost:8000/health
# Open http://localhost:3000
```

## Quick Start (Local Dev, no Docker)

### Backend

```bash
cd platform/backend

# Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Create a .env file for local dev
cp ../.env.example .env
# Edit .env: set DATABASE_URL=sqlite:///./dev.db, REDIS_URL, etc.

pip install -e ".[dev]"
uvicorn app.main:app --reload --port 8000
```

### Frontend (requires Node.js)

```bash
cd platform/frontend
npm install
npm run dev
```

## Running Tests

```bash
cd platform/backend
pytest tests/ -v
```
