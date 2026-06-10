# FedMamba-SALT Clinical Platform — AI Agent Build Decomposition
## Master Prompt Engineering Document

> **Purpose:** This document provides a phase-by-phase decomposition of building the FedMamba-SALT Clinical Platform into discrete, size-appropriate tasks for an AI coding agent. Each phase includes a self-contained execution prompt, explicit human-in-the-loop checkpoints, and clear handoff criteria before the next phase begins.

> **Architecture Principle:** The web platform wraps the existing research codebase via a **thin adapter layer**. The research code (`train_centralized.py`, `train_fedavg.py`, `train_fed_finetune.py`, `eval/`, `models/`, `objectives/`, `augmentations/`, `utils/`) is **never modified**. All platform logic calls into it through stable interfaces. This decoupling means research changes do not break the platform and platform changes do not corrupt research results.

---

## Pre-Phase: Context Brief for Every Agent Session

> **Paste this block at the top of EVERY agent prompt in this document.**

```
SYSTEM CONTEXT (read before acting):
You are building the FedMamba-SALT Clinical Platform — a production web application
that wraps an existing federated learning research codebase.

CRITICAL CONSTRAINTS:
1. NEVER modify any file in: augmentations/, models/, objectives/, utils/, eval/,
   train_centralized.py, train_fedavg.py, train_fed_finetune.py
   These are research files under active development. The platform adapts to them,
   not the reverse.

2. The federated learning backend is GENERALIZED. The platform must not hardcode
   assumptions about the number of clients, the dataset name, the number of classes,
   or the specific FL algorithm. All of these are runtime-configurable.

3. The platform communicates with the research backend through:
   - A subprocess/task queue interface (the research scripts are CLI tools)
   - A config file layer (YAML configs that the research scripts already support)
   - A results watcher (polling output directories for checkpoint files and CSVs)

4. Treat every research script as a black box with a stable CLI interface.
   Parse its stdout/stderr and output files; never reach into its internals.

5. Current repo structure:
   augmentations/medical_aug.py    — dual-view augmentation pipelines
   augmentations/retina_dataset.py — SSL-FL image dataset loader
   models/inception_mamba.py       — student encoder (InceptionMamba)
   models/vit_teacher.py           — frozen ViT-B/16 teacher
   objectives/salt_loss.py         — SALT loss (centered & standardised MSE)
   eval/linear_probe.py            — linear probe + full fine-tune evaluation
   eval/eval_tta.py                — TTA evaluation post-training
   train_centralized.py            — centralized pre-training entry point
   train_fedavg.py                 — federated pre-training (FedAvg/FedProx)
   train_fed_finetune.py           — federated fine-tuning evaluation
   utils/fedavg.py                 — FedAvg aggregation utilities
   utils/scaffold.py               — SCAFFOLD algorithm
   utils/data_splits.py            — client split CSV discovery
   utils/ckpt_compat.py            — checkpoint loading compatibility
   utils/teacher_stats.py          — teacher embedding statistics
```

---

## Phase Map Overview

```
Phase 0  — Project Scaffold & Monorepo Setup              [~2 h]   HUMAN REVIEW GATE
Phase 1  — Database Schema & Core Data Models             [~3 h]   HUMAN REVIEW GATE
Phase 2  — Authentication & Role System                   [~2 h]   HUMAN REVIEW GATE
Phase 3  — FL Backend Adapter Layer                       [~4 h]   HUMAN REVIEW GATE
Phase 4  — Admin Portal: Experiment Management UI         [~4 h]   HUMAN REVIEW GATE
Phase 5  — Hospital Client Portal                         [~3 h]   HUMAN REVIEW GATE
Phase 6  — Doctor Clinical Portal                         [~3 h]   HUMAN REVIEW GATE
Phase 7  — Real-Time Dashboard & Monitoring               [~3 h]   HUMAN REVIEW GATE
Phase 8  — Explainability & Report Generation             [~3 h]   HUMAN REVIEW GATE
Phase 9  — Security Hardening & Audit Logging             [~2 h]   HUMAN REVIEW GATE
Phase 10 — Integration Testing & Deployment Config        [~2 h]   FINAL HUMAN SIGN-OFF
```

---

---

# PHASE 0 — Project Scaffold & Monorepo Setup

## Metadata
| Field | Value |
|---|---|
| Estimated agent time | ~2 hours |
| Human gate type | Architecture review |
| Depends on | Nothing (first phase) |
| Produces | Runnable empty shell with working dev server |

## Human Input Required Before Starting

> ⚠️ **HUMAN DECISION REQUIRED — answer these before giving the Phase 0 prompt to an agent:**

1. **Technology stack choice:**
   - Backend: FastAPI (Python, stays in the same ecosystem as the research code) — *recommended*
   - Frontend: React + Vite (or Next.js if SSR is needed)
   - Database: PostgreSQL (production) + SQLite (local dev)
   - Task queue: Celery + Redis (for long-running FL jobs)
   - Real-time: WebSockets via FastAPI or Server-Sent Events

2. **Deployment target:** Docker Compose (local) → Kubernetes (production) or bare-metal hospital servers?

3. **Repository layout:** Monorepo (platform/ lives alongside research code) or separate repo?
   - *Recommended: monorepo, with `platform/` directory at root*

4. **Python version:** Match the research environment (confirm with `python --version` in the research environment).

---

## Phase 0 Agent Prompt

```
PHASE 0: Project Scaffold & Monorepo Setup

GOAL: Create the complete directory skeleton and configuration files for the
FedMamba-SALT Clinical Platform. No business logic yet — only structure,
tooling, and a verified "hello world" for every service.

[Paste the SYSTEM CONTEXT block from the top of this document here]

STACK (confirmed by human before this prompt):
- Backend: FastAPI + Python 3.11
- Frontend: React 18 + Vite 5 + TypeScript
- Database: PostgreSQL (prod), SQLite (dev)
- Task queue: Celery 5 + Redis 7
- Real-time: Server-Sent Events (SSE) via FastAPI
- Containerization: Docker Compose

DIRECTORY STRUCTURE TO CREATE (inside the existing repo root):
platform/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                  (FastAPI app factory)
│   │   ├── config.py                (Pydantic settings, reads .env)
│   │   ├── database.py              (SQLAlchemy engine + session)
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   └── v1/
│   │   │       ├── __init__.py
│   │   │       └── router.py        (mounts all sub-routers)
│   │   ├── core/
│   │   │   └── __init__.py
│   │   ├── models/                  (SQLAlchemy ORM models — empty for now)
│   │   │   └── __init__.py
│   │   ├── schemas/                 (Pydantic schemas — empty for now)
│   │   │   └── __init__.py
│   │   ├── services/                (business logic — empty for now)
│   │   │   └── __init__.py
│   │   └── tasks/                   (Celery task definitions — empty for now)
│   │       └── __init__.py
│   ├── tests/
│   │   └── test_health.py           (single smoke test: GET /health returns 200)
│   ├── pyproject.toml               (or requirements.txt)
│   ├── Dockerfile
│   └── celery_worker.py             (Celery app entry point)
├── frontend/
│   ├── src/
│   │   ├── main.tsx
│   │   ├── App.tsx
│   │   ├── pages/                   (empty)
│   │   ├── components/              (empty)
│   │   ├── hooks/                   (empty)
│   │   ├── stores/                  (empty — Zustand or Jotai)
│   │   └── api/                     (empty — typed API client)
│   ├── public/
│   ├── index.html
│   ├── vite.config.ts
│   ├── tsconfig.json
│   ├── package.json
│   └── Dockerfile
├── docker-compose.yml               (postgres, redis, backend, celery, frontend)
├── docker-compose.dev.yml           (override: hot-reload volumes)
├── .env.example                     (all required env vars documented)
└── README.md                        (setup instructions)

TASKS:
1. Create every file and directory listed above with minimal but correct
   boilerplate (imports, no placeholder comments like "TODO: implement").

2. backend/app/main.py must:
   - Create FastAPI app with title "FedMamba-SALT Clinical Platform"
   - Include a GET /health endpoint returning {"status": "ok", "version": "0.1.0"}
   - Include CORS middleware configured from environment variables
   - Mount /api/v1 router

3. backend/app/config.py must expose a Settings class (Pydantic BaseSettings) with:
   - DATABASE_URL: str
   - REDIS_URL: str
   - SECRET_KEY: str
   - ALLOWED_ORIGINS: list[str]
   - RESEARCH_ROOT: str  # absolute path to the repo root (parent of platform/)
   - FL_OUTPUT_DIR: str  # where research scripts write checkpoints
   Do NOT hardcode any values; all come from environment variables.

4. docker-compose.yml must define services: postgres, redis, backend, celery_worker,
   frontend. Use named volumes for postgres data. Backend depends_on postgres + redis.

5. backend/celery_worker.py must create a Celery app connected to Redis and
   auto-discover tasks in app/tasks/.

6. frontend/src/App.tsx must fetch GET /health and display the status on screen.
   This verifies the frontend can reach the backend.

7. Write backend/tests/test_health.py with a pytest test using httpx AsyncClient
   that calls GET /health and asserts status 200 and body {"status": "ok"}.

VERIFICATION STEPS (agent must run these and confirm they pass):
- `cd platform && docker-compose up --build -d`
- `curl http://localhost:8000/health` → must return {"status": "ok", ...}
- `cd backend && pytest tests/test_health.py -v` → must pass
- Frontend at http://localhost:3000 must display "Status: ok"

OUTPUT: List every file created with its path. Report test results.
Do not proceed if any verification step fails.
```
---