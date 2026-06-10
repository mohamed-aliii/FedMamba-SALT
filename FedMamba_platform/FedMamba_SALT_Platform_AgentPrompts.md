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

## Phase 0 Human Review Gate

Before moving to Phase 1, a human must verify:

- [ ] `docker-compose up` starts all services without errors
- [ ] `/health` endpoint returns 200
- [ ] Frontend renders and shows the health status
- [ ] Directory structure matches the spec exactly
- [ ] `.env.example` documents every required variable
- [ ] `RESEARCH_ROOT` correctly points to the repo root

**Human sign-off required before Phase 1 begins.**

---

---

# PHASE 1 — Database Schema & Core Data Models

## Metadata
| Field | Value |
|---|---|
| Estimated agent time | ~3 hours |
| Human gate type | Schema review + data governance sign-off |
| Depends on | Phase 0 complete |
| Produces | All ORM models, Alembic migrations, Pydantic schemas |

## Human Input Required Before Starting

> ⚠️ **HUMAN DECISION REQUIRED:**

1. **Patient data storage policy:** Does any patient-identifiable information get stored in the platform database, or only anonymized case IDs? *This is a regulatory/compliance decision — not a technical one.*

2. **Multi-tenancy model:** Are hospitals completely isolated schemas, or rows in shared tables protected by hospital_id foreign keys? *Recommended: row-level with hospital_id for simplicity, full schema isolation for strict compliance.*

3. **Audit retention policy:** How long must audit log entries be retained? (Affects table partitioning design.)

---

## Phase 1 Agent Prompt

```
PHASE 1: Database Schema & Core Data Models

GOAL: Define all SQLAlchemy ORM models, write Alembic migrations, and create
matching Pydantic schemas (request/response) for every entity in the system.

[Paste the SYSTEM CONTEXT block here]

ENTITIES TO MODEL:

1. User
   Fields: id (uuid), email, hashed_password, full_name, role (enum: ADMIN,
   HOSPITAL_MANAGER, DOCTOR), hospital_id (FK, nullable for ADMIN),
   is_active (bool), created_at, updated_at

2. Hospital
   Fields: id (uuid), name, description, api_key_hash, is_active,
   contact_email, created_at, updated_at

3. FLExperiment
   Fields: id (uuid), name, description, status (enum: PENDING, RUNNING,
   COMPLETED, FAILED, PAUSED), phase (enum: PRETRAIN, FINETUNE),
   algorithm (enum: FEDAVG, FEDPROX, SCAFFOLD, CENTRALIZED),
   config_yaml (text, the YAML passed to research scripts),
   output_dir (str), created_by_id (FK→User), created_at, updated_at,
   started_at, completed_at

4. FLRound
   Fields: id (uuid), experiment_id (FK→FLExperiment), round_number (int),
   status (enum: RUNNING, COMPLETED, FAILED), avg_loss (float, nullable),
   val_acc (float, nullable), val_auc (float, nullable), metrics_json (text),
   created_at, completed_at

5. HospitalParticipation
   Fields: id (uuid), experiment_id (FK→FLExperiment), hospital_id (FK→Hospital),
   client_index (int), dataset_size (int, nullable), is_active (bool),
   joined_at

6. ModelCheckpoint
   Fields: id (uuid), experiment_id (FK→FLExperiment), round_number (int,
   nullable), checkpoint_type (enum: LATEST, BEST, PERIODIC),
   file_path (str), val_acc (float, nullable), val_auc (float, nullable),
   is_deployed (bool), created_at

7. PatientCase
   Fields: id (uuid), case_uid (str, anonymized ID provided by hospital),
   hospital_id (FK→Hospital), uploaded_by_id (FK→User), image_path (str),
   metadata_json (text, non-PHI metadata only), created_at

8. DiagnosticResult
   Fields: id (uuid), case_id (FK→PatientCase), checkpoint_id
   (FK→ModelCheckpoint), prediction_class (int), confidence (float),
   probabilities_json (text), gradcam_path (str, nullable),
   processing_time_ms (int), created_at

9. AuditLog
   Fields: id (uuid), user_id (FK→User, nullable), action (str),
   resource_type (str), resource_id (str, nullable), ip_address (str,
   nullable), details_json (text), created_at

TASKS:

1. Create platform/backend/app/models/ files:
   - base.py: Base = declarative_base(), with id (uuid default), created_at,
     updated_at as TimestampMixin
   - user.py, hospital.py, experiment.py, checkpoint.py,
     patient.py, diagnostic.py, audit.py
   Each file imports Base from base.py and defines one or more models.

2. Create platform/backend/app/models/__init__.py that imports all models
   so Alembic can discover them.

3. Set up Alembic:
   - Run `alembic init platform/backend/alembic`
   - Configure alembic/env.py to use app.config.Settings().DATABASE_URL
     and import all models via app.models
   - Generate the initial migration: `alembic revision --autogenerate -m "initial_schema"`
   - Verify migration SQL looks correct (check for all 9 tables)

4. Create platform/backend/app/schemas/ files:
   - One file per entity group (user.py, hospital.py, experiment.py, etc.)
   - Each file has: Create schema (input), Read schema (output, includes id +
     timestamps), Update schema (all fields Optional)
   - Use Pydantic v2 model_config = ConfigDict(from_attributes=True)

5. Create platform/backend/app/database.py with:
   - async SQLAlchemy engine (asyncpg for postgres, aiosqlite for SQLite dev)
   - AsyncSessionLocal factory
   - get_db() async dependency for FastAPI

6. Write tests in platform/backend/tests/test_models.py:
   - Test that all 9 tables are created in a fresh SQLite test database
   - Test that a User + Hospital can be inserted and retrieved
   - Test that FLExperiment with FLRound relationship works

IMPORTANT CONSTRAINTS:
- PatientCase.image_path stores a relative path inside FL_OUTPUT_DIR only.
  Absolute paths or paths outside this directory must be rejected.
- AuditLog must NEVER be deletable via application code (no DELETE endpoint
  for audit logs). Add a comment in the model confirming this.
- All UUID primary keys must be server-generated (default=uuid4), never
  supplied by the client.
- The config_yaml field in FLExperiment stores the raw YAML text that will
  be passed verbatim to the research scripts. The platform must not parse or
  validate its contents — it is opaque to the platform layer.

OUTPUT: List all files created. Show the output of `alembic upgrade head`
(should complete without errors). Show test results.
```

---

## Phase 1 Human Review Gate

Before moving to Phase 2, a human must verify:

- [ ] All 9 tables created correctly in database
- [ ] Data governance officer has reviewed which fields are stored (patient data policy)
- [ ] `alembic upgrade head` runs clean on both SQLite and PostgreSQL
- [ ] Schema review: no PHI in database per organizational policy
- [ ] Audit log confirmed as insert-only in application code
- [ ] UUID strategy confirmed (server-generated only)

**Human sign-off required before Phase 2 begins.**

---

---

# PHASE 2 — Authentication & Role System

## Metadata
| Field | Value |
|---|---|
| Estimated agent time | ~2 hours |
| Human gate type | Security review |
| Depends on | Phase 1 complete |
| Produces | Working JWT auth, RBAC middleware, login/logout endpoints |

## Human Input Required Before Starting

> ⚠️ **HUMAN DECISION REQUIRED:**

1. **Token expiry policy:** How long should access tokens last? (e.g., 15 min access + 7-day refresh) — depends on hospital security policy.
2. **Hospital API key mechanism:** Hospitals authenticate their local training nodes via API key. Confirm: HMAC-SHA256 of key stored in DB, never the raw key.
3. **Password policy:** Minimum requirements (length, complexity) — set according to healthcare compliance requirements.

---

## Phase 2 Agent Prompt

```
PHASE 2: Authentication & Role System

GOAL: Implement JWT-based authentication with role-based access control (RBAC).
Three roles exist: ADMIN, HOSPITAL_MANAGER, DOCTOR. Define what each can access.

[Paste the SYSTEM CONTEXT block here]

ROLE PERMISSIONS MATRIX:
- ADMIN: full access to all endpoints
- HOSPITAL_MANAGER: read/write access to their own hospital's data and
  experiments they are participating in; cannot create new hospitals or
  manage other hospitals
- DOCTOR: read-only access to model results within their hospital; can
  upload PatientCases and request DiagnosticResults; cannot see FL training
  internals

TASKS:

1. Create platform/backend/app/core/security.py with:
   - hash_password(plain: str) → str  (bcrypt)
   - verify_password(plain: str, hashed: str) → bool
   - create_access_token(data: dict, expires_delta: timedelta) → str  (JWT)
   - create_refresh_token(data: dict) → str  (JWT, longer expiry)
   - decode_token(token: str) → dict | None
   - hash_api_key(key: str) → str  (HMAC-SHA256 with SECRET_KEY)
   - verify_api_key(key: str, stored_hash: str) → bool

2. Create platform/backend/app/core/deps.py with FastAPI dependencies:
   - get_current_user(token: str = Depends(oauth2_scheme), db: AsyncSession)
     → User  (raises 401 if invalid)
   - require_role(*roles: Role) → Callable  (dependency factory, raises 403
     if user role not in allowed roles)
   - get_current_hospital_user()  (shortcut: require_role(HOSPITAL_MANAGER))
   - get_current_doctor()  (shortcut: require_role(DOCTOR))
   - get_current_admin()  (shortcut: require_role(ADMIN))

3. Create platform/backend/app/api/v1/auth.py with endpoints:
   - POST /auth/login  (email + password → access_token + refresh_token)
   - POST /auth/refresh  (refresh_token → new access_token)
   - POST /auth/logout  (invalidate refresh token — add to a Redis blocklist)
   - GET  /auth/me  (returns current user profile)
   - POST /auth/change-password  (requires current password)

4. Create platform/backend/app/api/v1/users.py with endpoints:
   - POST /users/  (ADMIN only — create any user)
   - GET  /users/  (ADMIN only — list all users)
   - GET  /users/{user_id}  (ADMIN or self)
   - PATCH /users/{user_id}  (ADMIN or self — cannot change own role)
   - DELETE /users/{user_id}  (ADMIN only — soft delete: set is_active=False)

5. Create platform/backend/app/services/auth_service.py with the business
   logic called by auth.py routers (keeps routers thin).

6. Create platform/backend/app/core/audit.py with:
   - log_action(db, user_id, action, resource_type, resource_id, details,
     request) — writes to AuditLog table
   - Call this from every state-changing endpoint (POST, PATCH, DELETE)
   - Include the client IP from the request object

7. Write tests in platform/backend/tests/test_auth.py:
   - Test login with valid credentials returns 200 + tokens
   - Test login with wrong password returns 401
   - Test accessing a protected endpoint without token returns 401
   - Test that DOCTOR cannot access ADMIN-only endpoint (403)
   - Test token refresh works
   - Test logout invalidates token (second use of same refresh_token returns 401)

SECURITY REQUIREMENTS:
- Tokens must include: sub (user_id), role, hospital_id (nullable), iat, exp
- Refresh tokens must be stored in Redis with TTL equal to their expiry
- Logout must add the refresh token jti to a Redis blocklist
- Never log passwords, tokens, or API keys in any log output
- All password hashing must use bcrypt with work factor >= 12
- Rate-limit /auth/login to 5 attempts per minute per IP (use slowapi)

OUTPUT: List all files. Show test results (all must pass).
```

---

## Phase 2 Human Review Gate

Before Phase 3:

- [ ] Security team has reviewed the JWT implementation
- [ ] RBAC matrix confirmed with stakeholders (admin, hospital manager, doctor)
- [ ] Rate limiting confirmed as active on login endpoint
- [ ] Token expiry values set to organization policy
- [ ] API key hashing strategy confirmed (HMAC-SHA256 with SECRET_KEY)

**Human sign-off required before Phase 3 begins.**

---

---

# PHASE 3 — FL Backend Adapter Layer

## Metadata
| Field | Value |
|---|---|
| Estimated agent time | ~4 hours |
| Human gate type | Technical integration review |
| Depends on | Phase 2 complete |
| Produces | The bridge between the platform and research scripts |

## Human Input Required Before Starting

> ⚠️ **HUMAN DECISION REQUIRED:**

1. **Execution environment:** Do the research scripts run on the same machine as the platform, or on a remote GPU server? If remote: SSH-based execution or job scheduler (SLURM)?
2. **GPU availability:** Will the platform ever run CPU-only? (Affects default config generation.)
3. **Output directory strategy:** Confirm the path convention for `FL_OUTPUT_DIR`. Where do checkpoints land relative to the research root?

---

## Phase 3 Agent Prompt

```
PHASE 3: FL Backend Adapter Layer

GOAL: Build the complete adapter that translates platform database state into
research script CLI calls, monitors their progress, and parses their output
back into the database. This is the most critical phase — it is the only
place the platform touches the research codebase.

[Paste the SYSTEM CONTEXT block here]

ARCHITECTURE:
- Platform never imports research modules directly (no `from models import ...`)
- Platform launches research scripts as subprocesses via Celery tasks
- Platform monitors output directories for CSV metrics and checkpoint files
- Platform parses stdout/stderr for real-time log forwarding
- All communication is one-way: platform → subprocess → filesystem → platform

IMPORTANT: The research scripts accept YAML configs (--config flag) for all
hyperparameters. The platform generates these YAML files. The config_yaml
field in FLExperiment stores the YAML text. The platform must generate
valid YAML but must not validate its semantic correctness — that is the
research team's responsibility.

TASKS:

1. Create platform/backend/app/core/fl_config.py with:

   class FLConfigGenerator:
       """
       Generates YAML config files for research scripts.
       The platform provides a set of KNOWN parameters it can configure.
       Any additional parameters can be passed through an 'extra_config'
       JSON blob on the FLExperiment, giving the research team full control
       over parameters the platform doesn't know about.
       """

       def generate_pretrain_config(
           self,
           experiment: FLExperiment,
           output_dir: str,
           data_path: str,
           teacher_ckpt: str,
           federated: bool,
           **extra,
       ) -> str:  # returns YAML string

       def generate_finetune_config(
           self,
           experiment: FLExperiment,
           output_dir: str,
           encoder_ckpt: str,
           data_path: str,
           **extra,
       ) -> str:  # returns YAML string

   Both methods merge the experiment's stored config_yaml (if any) with the
   generated parameters. The stored config_yaml always wins for overlapping
   keys (research team override).

2. Create platform/backend/app/core/fl_runner.py with:

   class FLScriptRunner:
       """Manages subprocess execution of research scripts."""

       SCRIPT_MAP = {
           ("PRETRAIN", "CENTRALIZED"): "train_centralized.py",
           ("PRETRAIN", "FEDAVG"):      "train_fedavg.py",
           ("PRETRAIN", "FEDPROX"):     "train_fedavg.py",
           ("PRETRAIN", "SCAFFOLD"):    "train_fedavg.py",
           ("FINETUNE", "FEDAVG"):      "train_fed_finetune.py",
           ("FINETUNE", "FEDPROX"):     "train_fed_finetune.py",
           ("FINETUNE", "SCAFFOLD"):    "train_fed_finetune.py",
           ("FINETUNE", "CENTRALIZED"): "eval/linear_probe.py",
       }

       def build_command(self, script: str, config_path: str) -> list[str]:
           """Returns the full command list for subprocess.Popen"""

       def launch(self, command: list[str], experiment_id: str) -> subprocess.Popen:
           """Launch subprocess, capture stdout+stderr, return handle"""

       def terminate(self, process: subprocess.Popen) -> None:
           """Gracefully stop a running experiment"""

3. Create platform/backend/app/core/fl_monitor.py with:

   class FLOutputMonitor:
       """Polls the experiment output directory and parses results."""

       def find_metrics_csv(self, output_dir: str) -> str | None:
           """
           Looks for known CSV filenames:
           - training_metrics.csv (train_centralized.py)
           - federated_metrics.csv (train_fedavg.py)
           - fed_finetune_metrics.csv (train_fed_finetune.py)
           Returns path to first found file, or None.
           """

       def parse_latest_round(self, csv_path: str) -> dict | None:
           """
           Parse the last row of any metrics CSV.
           Returns a dict with normalized keys: round, loss, val_acc,
           val_auc, lr, time_s — mapping from the CSV column names
           that vary by script.

           Column name mapping (source → normalized):
           train_centralized.py:   epoch → round, loss → loss
           train_fedavg.py:        round → round, avg_loss → loss
           train_fed_finetune.py:  round → round, val_acc → val_acc
           """

       def find_checkpoints(self, output_dir: str) -> list[dict]:
           """
           Scans for ckpt_*.pth files.
           Returns list of {filename, path, type: LATEST|BEST|PERIODIC,
           round_number (from filename if parseable)}.
           """

       def parse_loss_from_stdout_line(self, line: str) -> dict | None:
           """
           Parse a stdout line from any research script.
           Returns {round, loss, enc_std, lr} if the line matches
           the known log format, else None.
           Used for streaming real-time updates.
           """

4. Create platform/backend/app/tasks/fl_tasks.py with Celery tasks:

   @celery_app.task(bind=True, max_retries=0)
   def run_fl_experiment(self, experiment_id: str) -> None:
       """
       Main Celery task for running an FL experiment.
       Steps:
       1. Load experiment from DB
       2. Generate YAML config file to output_dir/config.yaml
       3. Build subprocess command
       4. Update experiment status to RUNNING in DB
       5. Launch subprocess
       6. Poll every 10 seconds:
          a. Check if process is still alive
          b. Parse latest metrics CSV row
          c. Write FLRound record to DB with latest metrics
          d. Emit SSE event via Redis pub/sub
       7. On process exit:
          a. Check return code
          b. Scan for checkpoints → write ModelCheckpoint records
          c. Update experiment status to COMPLETED or FAILED
       """

   @celery_app.task
   def stop_fl_experiment(experiment_id: str) -> None:
       """Terminate a running experiment gracefully."""

   @celery_app.task
   def sync_experiment_checkpoints(experiment_id: str) -> None:
       """Re-scan output directory and sync checkpoint records to DB.
       Used for recovery after a crash."""

5. Create platform/backend/app/api/v1/experiments.py with endpoints:
   - POST /experiments/       (ADMIN — create experiment, schedule task)
   - GET  /experiments/       (ADMIN — list all; HOSPITAL_MANAGER — list theirs)
   - GET  /experiments/{id}   (role-filtered)
   - PATCH /experiments/{id}  (ADMIN — update config before start)
   - POST /experiments/{id}/start   (ADMIN — enqueue run_fl_experiment)
   - POST /experiments/{id}/stop    (ADMIN — enqueue stop_fl_experiment)
   - GET  /experiments/{id}/rounds  (paginated list of FLRound records)
   - GET  /experiments/{id}/checkpoints  (list ModelCheckpoint records)
   - GET  /experiments/{id}/logs    (SSE stream of stdout lines from Redis)

6. Create platform/backend/app/api/v1/sse.py with:
   - GET /sse/experiments/{id}  (SSE endpoint, streams round updates)
   Uses Redis pub/sub: Celery task publishes to channel experiment:{id},
   SSE endpoint subscribes and forwards to browser.

7. Write tests in platform/backend/tests/test_fl_adapter.py:
   - Test FLConfigGenerator produces valid YAML (parseable with yaml.safe_load)
   - Test FLOutputMonitor.parse_latest_round handles all 3 CSV formats correctly
     (use fixture CSV files matching the exact column names from the research scripts)
   - Test FLOutputMonitor.find_checkpoints finds ckpt_latest.pth, ckpt_best.pth,
     and ckpt_round_0010.pth and classifies them correctly
   - Test that run_fl_experiment task updates experiment status to RUNNING
     (mock subprocess, mock DB)

FIXTURE CSV ROWS FOR TESTS (exact column names from research scripts):
train_centralized.py row:
  epoch,loss,student_std,teacher_std,salt_norm_mode,...,lr,epoch_time_s,...

train_fedavg.py row:
  round,avg_loss,avg_enc_std,avg_teacher_std,salt_norm_mode,...,lr,round_time_s,...

train_fed_finetune.py row:
  round,val_acc,val_loss_weighted,...,auc,...,enc_lr,cls_lr,round_time_s,...

CRITICAL: The column mappings above must be read from the actual CSV headers
that the research scripts produce. Do not hardcode column indices — always
locate columns by header name.

OUTPUT: List all files. Show test results.
```

---

## Phase 3 Human Review Gate

Before Phase 4:

- [ ] Run the adapter against a real (small, local) experiment end-to-end
- [ ] Verify the metrics CSV parser handles all 3 script output formats
- [ ] Confirm checkpoint file detection works on actual output directories
- [ ] Research team reviews that no research script was modified
- [ ] Verify SSE stream delivers round updates to a browser tab
- [ ] Celery worker confirmed running and processing tasks

**Human sign-off required before Phase 4 begins.**

---

---

# PHASE 4 — Admin Portal: Experiment Management UI

## Metadata
| Field | Value |
|---|---|
| Estimated agent time | ~4 hours |
| Human gate type | UX review + functional testing |
| Depends on | Phase 3 complete |
| Produces | Full admin web UI for managing hospitals, experiments, and models |

## Human Input Required Before Starting

> ⚠️ **HUMAN DECISION REQUIRED:**

1. **Design system:** Use a component library (Mantine, shadcn/ui, Ant Design) or build from scratch? Confirm before this phase.
2. **YAML editor:** Should the admin be able to edit the raw YAML config in-browser? If yes, confirm it is clearly labeled "Advanced — for research team use only."

---

## Phase 4 Agent Prompt

```
PHASE 4: Admin Portal — Experiment Management UI

GOAL: Build the complete React frontend for the Admin role. The admin can
manage hospitals, create and monitor FL experiments, manage model checkpoints,
and deploy models for clinical use.

[Paste the SYSTEM CONTEXT block here]

PAGES TO BUILD:

1. /admin/dashboard
   - Summary cards: active experiments, total hospitals, deployed models,
     recent alerts
   - Last 5 experiment status updates (live via SSE)

2. /admin/hospitals
   - Table of all hospitals with status, client count, last activity
   - Create hospital modal: name, contact email → generates API key
     (display once, never again)
   - Deactivate hospital button (confirmation dialog)

3. /admin/experiments
   - Table with columns: name, phase, algorithm, status, current round,
     best val_acc, created_at
   - Status badges: color-coded (PENDING=gray, RUNNING=blue, COMPLETED=green,
     FAILED=red, PAUSED=yellow)
   - Create experiment flow (multi-step wizard):
     Step 1: Basic info (name, description, phase, algorithm)
     Step 2: Participating hospitals (multi-select with client index assignment)
     Step 3: Training config (form fields for known parameters)
     Step 4: Advanced YAML override (code editor, labeled "Research Team")
     Step 5: Review and confirm

4. /admin/experiments/{id}
   - Experiment detail: status, config summary, participating hospitals
   - Start / Stop / Pause buttons (state-dependent)
   - Live training chart: loss over rounds (recharts LineChart, SSE-fed)
   - Rounds table: round number, loss, val_acc, val_auc, time
   - Checkpoints panel: list with accuracy, type (BEST/LATEST/PERIODIC),
     Deploy button (sets is_deployed=true, only one can be deployed at a time)
   - Log stream panel: scrolling terminal-style view of stdout (SSE)

5. /admin/models
   - Table of all deployed ModelCheckpoints across all experiments
   - Columns: experiment name, round, val_acc, val_auc, deployed_at, actions
   - Undeploy button (requires confirmation)

TECHNICAL REQUIREMENTS:

1. Create platform/frontend/src/api/client.ts:
   - Axios instance with base URL from env
   - Request interceptor: attach Bearer token from localStorage
   - Response interceptor: on 401 → attempt token refresh → retry → logout

2. Create platform/frontend/src/api/ typed modules:
   - experiments.ts, hospitals.ts, auth.ts, checkpoints.ts
   Each module exports typed async functions mirroring the backend endpoints.

3. Create platform/frontend/src/stores/ (Zustand):
   - authStore.ts: user, tokens, login(), logout(), refreshToken()
   - experimentStore.ts: experiments list, current experiment, rounds

4. Create platform/frontend/src/hooks/useSSE.ts:
   - Subscribes to /api/v1/sse/experiments/{id}
   - Calls a callback on each event
   - Handles reconnection on disconnect
   - Cleans up EventSource on unmount

5. Create platform/frontend/src/components/charts/TrainingChart.tsx:
   - Recharts LineChart with two Y-axes: loss (left) and val_acc (right)
   - Updates in real-time as SSE events arrive
   - Shows current round number in chart title

6. Create platform/frontend/src/components/ExperimentWizard/:
   - Multi-step form as described above
   - Step 3 must render form fields for these known parameters ONLY:
     n_clients, E_epoch, batch_size, lr, max_rounds (or epochs), mu,
     mask_ratio, dataset, num_classes, label_fraction, algo
   - All other parameters go to the YAML override in Step 4
   - The YAML editor must show a read-only base YAML generated from Step 3
     values, with a user-editable override section at the bottom

7. Protected routing:
   - All /admin/* routes require role=ADMIN
   - Redirect to /login if not authenticated
   - Show "Access Denied" if wrong role

OUTPUT: Describe every component file created. Verify the create experiment
wizard reaches the backend (POST /experiments/ returns 201). Verify SSE
chart updates when a fake event is published to Redis.
```

---

## Phase 4 Human Review Gate

Before Phase 5:

- [ ] Admin can create a hospital and see the generated API key (displayed once)
- [ ] Experiment wizard completes all 5 steps and creates a DB record
- [ ] Start experiment button triggers the Celery task
- [ ] Live chart updates as rounds complete
- [ ] Log stream shows real stdout from the research script
- [ ] Deploy checkpoint button sets one model as deployed
- [ ] UX review: research team confirms YAML override is clearly labeled and usable

**Human sign-off required before Phase 5 begins.**

---

---

# PHASE 5 — Hospital Client Portal

## Metadata
| Field | Value |
|---|---|
| Estimated agent time | ~3 hours |
| Human gate type | Hospital IT team review |
| Depends on | Phase 4 complete |
| Produces | Hospital manager UI + local training node API endpoints |

## Human Input Required Before Starting

> ⚠️ **HUMAN DECISION REQUIRED:**

1. **Local training node:** Will hospitals run training on their own hardware (requiring a local agent that calls back to the platform) or will all training be orchestrated centrally (platform SSHes into hospital servers)? This is an infrastructure/security decision.
2. **Data upload policy:** Does the platform receive any data from hospitals (even anonymized), or do hospitals only report training results? *Confirm with data governance.*

---

## Phase 5 Agent Prompt

```
PHASE 5: Hospital Client Portal

GOAL: Build the Hospital Manager portal and the hospital node registration
API. Hospital managers can see their participation in experiments, monitor
local training status, and manage their registered training nodes.

[Paste the SYSTEM CONTEXT block here]

NOTE ON TRAINING ARCHITECTURE:
The research scripts run on a central machine managed by the platform.
Hospitals do not run training locally in this version. Hospital managers
use this portal to:
1. View which experiments their hospital is participating in
2. Monitor their contribution metrics (client loss, dataset size)
3. Manage their hospital's registered doctors
4. See their hospital's model access

This design is intentional: it keeps patient data on hospital premises
(referenced by path on the central secure server) while the platform
manages compute. The hospital's participation is tracked via the
HospitalParticipation table.

PAGES TO BUILD (accessible to HOSPITAL_MANAGER role):

1. /hospital/dashboard
   - Summary: active experiments, my hospital's total training rounds,
     doctors registered, deployed models accessible to my hospital
   - Recent experiment activity

2. /hospital/experiments
   - Table of experiments this hospital is participating in
   - Per-experiment: my client index, my dataset size, my contribution
     loss (from FLRound metrics_json), current round
   - Read-only view (cannot start/stop experiments)

3. /hospital/experiments/{id}
   - Experiment detail (read-only subset of admin view)
   - Chart: my client's loss over rounds vs global average
   - My client's metrics extracted from rounds_json

4. /hospital/doctors
   - List of doctors registered at this hospital
   - Invite doctor: send invite by email (creates User with DOCTOR role,
     hospital_id = this hospital, temporary password)
   - Deactivate doctor (soft delete)

5. /hospital/models
   - Deployed models accessible to this hospital
   - Model info: accuracy, AUC, training experiment
   - "Allowed to use" status (hospital must be a participant in the
     experiment that produced the model to access it)

API ENDPOINTS TO BUILD:

1. GET  /hospitals/me  — returns the current user's hospital details
2. GET  /hospitals/me/participation  — list of HospitalParticipation records
3. GET  /hospitals/me/doctors  — list doctors in my hospital
4. POST /hospitals/me/doctors  — create a doctor account in my hospital
   (HOSPITAL_MANAGER only, cannot create ADMIN or other HOSPITAL_MANAGER)
5. PATCH /hospitals/me/doctors/{user_id}  — update doctor (active status)

ACCESS CONTROL RULE:
A HOSPITAL_MANAGER can only see data where hospital_id = their hospital_id.
Implement this as a FastAPI dependency: get_my_hospital_or_403() that
reads the current user's hospital_id and injects it into every query.
Never rely on the client sending their hospital_id in the request body.

MODEL ACCESS RULE:
A hospital may use a deployed model ONLY if they participated in the
experiment that produced the checkpoint. Enforce this check in the
diagnostic endpoint (Phase 6) — not here. Here only display the allowed
models.

TESTS:
- Test that a HOSPITAL_MANAGER cannot access another hospital's data
  (should return 403, not 404)
- Test that doctor creation correctly assigns the manager's hospital_id
- Test that the participation endpoint correctly filters by hospital

OUTPUT: List all files. Show test results.
```

---

## Phase 5 Human Review Gate

Before Phase 6:

- [ ] Hospital manager can see their experiments and contribution metrics
- [ ] Hospital manager can invite/manage doctors (within their hospital only)
- [ ] Hospital manager CANNOT see other hospitals' data (confirmed by test)
- [ ] Model access list correctly shows only models from participated experiments
- [ ] Hospital IT representative has reviewed the portal for usability

**Human sign-off required before Phase 6 begins.**

---

---

# PHASE 6 — Doctor Clinical Portal

## Metadata
| Field | Value |
|---|---|
| Estimated agent time | ~3 hours |
| Human gate type | Clinical usability review + ethics/IRB check |
| Depends on | Phase 5 complete |
| Produces | Doctor-facing diagnostic workflow |

## Human Input Required Before Starting

> ⚠️ **HUMAN DECISION REQUIRED:**

1. **Clinical disclaimer:** Every diagnostic result must display a mandatory disclaimer ("AI-assisted, not a clinical diagnosis. Must be reviewed by a qualified clinician."). Confirm exact disclaimer text with clinical/legal team.
2. **Image format policy:** Which image formats will doctors upload? (DICOM, PNG, JPEG, NPY?) DICOM requires special handling.
3. **Inference execution:** Is inference run on CPU (as per original design) or GPU? Where does inference run — platform server or hospital workstation?
4. **Report format:** Does the generated clinical report need to match a specific template from the hospital's EMR system?

---

## Phase 6 Agent Prompt

```
PHASE 6: Doctor Clinical Portal

GOAL: Build the doctor-facing interface for uploading patient cases and
receiving AI-assisted diagnostic results. The portal must be clinically
appropriate: clear, fast, explainable, and clearly labeled as AI-assisted.

[Paste the SYSTEM CONTEXT block here]

CRITICAL CLINICAL REQUIREMENTS (non-negotiable):
1. Every diagnostic result page MUST display this disclaimer prominently:
   [INSERT APPROVED DISCLAIMER TEXT FROM HUMAN REVIEW]
   in a visually distinct warning box before the result.
2. Confidence scores must be shown with uncertainty framing:
   "The model is 87% confident in this classification" — not just "87%"
3. The doctor must explicitly acknowledge the disclaimer before viewing
   results (checkbox: "I understand this is AI-assisted, not a diagnosis")

PAGES TO BUILD (DOCTOR role):

1. /clinical/dashboard
   - Recent cases (last 10)
   - Quick upload button
   - Deployed model info (which model is active, its accuracy/AUC)

2. /clinical/upload
   - File upload dropzone (drag-and-drop)
   - Case UID field (anonymized ID from hospital's records — doctor provides)
   - Submit button
   - On submit: creates PatientCase record, enqueues inference task
   - Redirects to /clinical/cases/{id}/results

3. /clinical/cases
   - Table of all cases uploaded by this doctor
   - Columns: case UID, uploaded at, status (PENDING/COMPLETE/FAILED),
     predicted class, confidence
   - Link to case detail

4. /clinical/cases/{id}/results
   - Disclaimer acknowledgment gate (must check box to proceed)
   - Prediction card:
     - Predicted class name (e.g., "Diabetic Retinopathy Detected")
     - Confidence bar (visual, 0-100%)
     - All class probabilities table
   - Grad-CAM heatmap (if available) shown as overlay on original image
   - Processing time displayed
   - "Generate Report" button → PDF download
   - "Submit for Second Opinion" button (flags case for clinical review)

API ENDPOINTS TO BUILD:

1. POST /cases/
   - Accepts multipart/form-data: file + case_uid
   - Validates: file extension, file size limit (configurable, default 50MB)
   - Saves file to FL_OUTPUT_DIR/cases/{hospital_id}/{uuid}.{ext}
   - Creates PatientCase record
   - Enqueues run_inference Celery task
   - Returns: {case_id, status: "PENDING"}

2. GET  /cases/  — list doctor's own cases (paginated)
3. GET  /cases/{id}  — case detail + diagnostic result (if available)
4. GET  /cases/{id}/image  — serve the uploaded image (auth-gated)
5. GET  /cases/{id}/gradcam  — serve the Grad-CAM overlay image (if available)
6. POST /cases/{id}/report  — generate PDF report, return download URL

INFERENCE CELERY TASK (platform/backend/app/tasks/inference_tasks.py):

@celery_app.task
def run_inference(case_id: str) -> None:
    """
    Run model inference on a patient case using the deployed checkpoint.

    Steps:
    1. Load PatientCase from DB
    2. Find the deployed ModelCheckpoint (is_deployed=True)
    3. Verify the doctor's hospital participated in that checkpoint's experiment
    4. Build inference command:
       python eval/eval_tta.py
         --ckpt {checkpoint.file_path}
         --data_path {temp_dir_with_single_image}
         --num_classes {n_classes}
         --n_tta 4
         --output_dir {output_dir}
    5. Run subprocess, wait for completion
    6. Parse output JSON from output_dir/predictions.json
       (the eval script must be extended — see note below)
    7. Write DiagnosticResult record
    8. Optionally generate Grad-CAM (see Phase 8)
    """

IMPORTANT NOTE ON EVAL SCRIPT:
eval/eval_tta.py does not currently output a JSON predictions file. The
inference task should NOT modify eval_tta.py. Instead, create a new script:
platform/backend/app/core/single_image_inference.py
that loads the checkpoint using the same load_encoder/build_model pattern
as eval_tta.py (by importing from eval.linear_probe and eval.eval_tta —
READING from the research code is allowed, only MODIFYING is forbidden)
and performs single-image inference, outputting a JSON file.

This script runs as a subprocess, exactly like the other research scripts.

TESTS:
- Test POST /cases/ with a valid image file returns 201 + case_id
- Test GET /cases/{id} before inference returns status=PENDING
- Test access control: doctor cannot access another hospital's case (403)
- Test inference task produces DiagnosticResult record (mock subprocess)

OUTPUT: List all files. Show test results. Confirm disclaimer text is visible
on the results page before and during result display.
```

---

## Phase 6 Human Review Gate

Before Phase 7:

- [ ] Clinical team has reviewed the disclaimer text and placement
- [ ] IRB/ethics check: patient case handling meets regulatory requirements
- [ ] Doctor can upload an image and receive a result end-to-end (integration test)
- [ ] Disclaimer acknowledgment gate is functional and required
- [ ] Confidence framing language reviewed by clinical team
- [ ] Image file size limits and accepted formats confirmed

**Human sign-off required before Phase 7 begins.**

---

---

# PHASE 7 — Real-Time Dashboard & Monitoring

## Metadata
| Field | Value |
|---|---|
| Estimated agent time | ~3 hours |
| Human gate type | Operational review |
| Depends on | Phase 6 complete |
| Produces | Live monitoring dashboards for all roles |

## Human Input Required Before Starting

> ⚠️ **HUMAN DECISION REQUIRED:**

1. **Alert thresholds:** What loss plateau or accuracy drop values should trigger an alert? (Research team decision — they know the expected training behavior.)
2. **Notification channels:** Should alerts go to email, Slack, or just in-app? (IT/admin decision.)

---

## Phase 7 Agent Prompt

```
PHASE 7: Real-Time Dashboard & Monitoring

GOAL: Build comprehensive real-time dashboards for all three user roles,
plus a system monitoring backend that detects training anomalies.

[Paste the SYSTEM CONTEXT block here]

BACKEND TASKS:

1. Create platform/backend/app/core/fl_monitor_alerts.py with:

   class TrainingAnomalyDetector:
       """
       Detects training problems by analyzing FLRound records.
       Called after every round is written to DB.
       """
       def check_loss_plateau(self, rounds: list[FLRound], patience: int = 20) -> bool
       def check_val_acc_drop(self, rounds: list[FLRound], threshold: float = 5.0) -> bool
       def check_client_divergence(self, round: FLRound) -> bool
           # Parses metrics_json for per-client losses and flags if any
           # client's loss is >3x the average
       def check_nan_loss(self, round: FLRound) -> bool

   @celery_app.task
   def check_experiment_health(experiment_id: str) -> None:
       """Called after each round is written. Publishes alerts to Redis."""

2. Add GET /experiments/{id}/metrics endpoint:
   - Returns all FLRound records for an experiment as JSON
   - Supports query params: ?from_round=N&limit=M
   - Used for chart initial load (SSE handles incremental updates)

3. Add GET /system/stats endpoint (ADMIN only):
   - Active Celery tasks count
   - Total experiments by status
   - Total cases processed today
   - Total model predictions today

FRONTEND — ADMIN MONITORING VIEW (/admin/monitoring):

1. System health panel:
   - Active tasks (Celery queue depth from Redis)
   - DB connection pool status
   - Disk usage for FL_OUTPUT_DIR
   - Last updated timestamp (auto-refreshes every 30s)

2. Active experiments panel:
   - Card per running experiment
   - Live progress bar: current_round / max_rounds
   - Live loss chart (mini, sparkline style)
   - Last update time
   - Alert badges (plateau, divergence, NaN)

3. Alerts panel:
   - List of active anomaly alerts
   - Each alert: experiment name, alert type, round number, description
   - Dismiss button (ADMIN only)

FRONTEND — HOSPITAL MANAGER MONITORING (/hospital/monitoring):

1. My experiment contributions:
   - Per experiment: my client loss vs global loss chart
   - Rounds where my client diverged from global average (highlighted)
   - My client participation rate (rounds active / total rounds)

FRONTEND — DOCTOR MONITORING (/clinical/my-stats):

1. My case history:
   - Cases this week / this month
   - Average confidence score distribution (histogram)
   - Cases by predicted class (bar chart)

ALL CHARTS must use Recharts and receive data from:
- Initial GET /experiments/{id}/metrics (full history)
- SSE stream for incremental updates (append new data points)
Never re-fetch the full dataset on each update.

TESTS:
- Test TrainingAnomalyDetector.check_loss_plateau with fixture round data
- Test check_nan_loss correctly identifies NaN loss records
- Test GET /system/stats requires ADMIN role
- Test SSE stream delivers an alert event when anomaly is detected

OUTPUT: List all files. Show test results.
```

---

## Phase 7 Human Review Gate

Before Phase 8:

- [ ] Alert thresholds set to research-team-approved values
- [ ] Live chart updates verified in browser (actual SSE test)
- [ ] System stats endpoint returns correct counts
- [ ] Anomaly detector correctly flags a loss plateau (manual test with fake data)
- [ ] Operations team has reviewed the monitoring dashboard

**Human sign-off required before Phase 8 begins.**

---

---

# PHASE 8 — Explainability & Report Generation

## Metadata
| Field | Value |
|---|---|
| Estimated agent time | ~3 hours |
| Human gate type | Clinical + research review |
| Depends on | Phase 7 complete |
| Produces | Grad-CAM visualization + PDF clinical report |

## Human Input Required Before Starting

> ⚠️ **HUMAN DECISION REQUIRED:**

1. **Report template:** Provide the clinical report template (header, footer, required fields) approved by the clinical team. This affects the PDF layout code.
2. **Grad-CAM layer:** Which layer in the InceptionMambaEncoder should Grad-CAM target? (Research team decision — depends on model architecture understanding.)
3. **Hospital logo in reports:** Should the generated PDF include the hospital's logo? (IT decision — logo file path/URL needed.)

---

## Phase 8 Agent Prompt

```
PHASE 8: Explainability & Report Generation

GOAL: Add Grad-CAM visualization to diagnostic results and generate
downloadable PDF clinical reports. Both run as subprocesses to avoid
importing research models into the platform Python process.

[Paste the SYSTEM CONTEXT block here]

GRAD-CAM IMPLEMENTATION:

Create platform/backend/app/core/gradcam_runner.py — a standalone Python
script (not imported, run as subprocess) that:

1. Accepts CLI args: --ckpt, --image, --output_dir, --target_layer (optional),
   --num_classes, --device
2. Loads the model using IDENTICAL code to eval/eval_tta.py's build_model()
   function (copy the loading logic, do not import from eval — this script
   runs as a subprocess with its own Python process so the PYTHONPATH must
   include the repo root)
3. Computes Grad-CAM using pytorch-grad-cam library:
   from pytorch_grad_cam import GradCAM
   from pytorch_grad_cam.utils.image import show_cam_on_image
4. Saves: {output_dir}/gradcam_overlay.png (heatmap overlaid on original)
            {output_dir}/gradcam_raw.npy (raw CAM values for later use)
            {output_dir}/gradcam_meta.json (layer name, class idx, confidence)
5. Exits 0 on success, 1 on failure

Celery task platform/backend/app/tasks/inference_tasks.py:
After run_inference completes successfully, enqueue run_gradcam task:

@celery_app.task
def run_gradcam(case_id: str) -> None:
    """Runs gradcam_runner.py as subprocess, updates DiagnosticResult.gradcam_path"""

PDF REPORT GENERATION:

Create platform/backend/app/core/report_generator.py — generates PDF using
ReportLab or WeasyPrint (not subprocess — this is pure Python data formatting,
no ML code):

class ClinicalReportGenerator:

    def generate(self, case: PatientCase, result: DiagnosticResult,
                 hospital: Hospital, doctor: User,
                 gradcam_path: str | None) -> bytes:
        """
        Returns PDF bytes for the clinical report.

        Report sections:
        1. Header: Hospital name, logo (if configured), report date/time,
           report ID (DiagnosticResult.id)
        2. Patient information: case_uid, uploaded_by (doctor name), date
        3. AI Analysis Summary:
           - Predicted class (large, clear)
           - Confidence with disclaimer
           - All class probabilities table
        4. Visual Evidence (if Grad-CAM available):
           - Original image thumbnail
           - Grad-CAM overlay
           - Caption: "Highlighted regions contributed most to the prediction"
        5. Model Information:
           - Model version (checkpoint experiment name + round)
           - Model accuracy (val_acc) and AUC on training data
           - Training dataset description
        6. MANDATORY DISCLAIMER (full text, bold, boxed):
           [INSERT APPROVED DISCLAIMER TEXT]
        7. Signature line: "Reviewed by: _______________  Date: ___________"
        8. Footer: Report ID, generated timestamp, "Confidential — Patient Record"
        """

POST /cases/{id}/report endpoint:
- Calls ClinicalReportGenerator.generate()
- Returns the PDF bytes with Content-Type: application/pdf
- Logs the report generation in AuditLog

FRONTEND UPDATES:

1. Update /clinical/cases/{id}/results to show Grad-CAM overlay:
   - Display original image and Grad-CAM side by side
   - Show "Generating explanation..." spinner while gradcam task runs
   - Poll GET /cases/{id} every 5s until gradcam_path is populated
   - Once populated, load /cases/{id}/gradcam image

2. Add GradCamViewer component:
   - Two panels: original image | heatmap overlay
   - Slider to blend between original and heatmap (opacity control)
   - Caption explaining what Grad-CAM shows

TESTS:
- Test ClinicalReportGenerator.generate() produces valid PDF bytes
  (check that output starts with b'%PDF')
- Test that report includes disclaimer text
- Test gradcam_runner.py script exists and is runnable (dry run with --help)
- Test POST /cases/{id}/report returns 200 with content-type application/pdf

IMPORTANT: The Grad-CAM target layer for InceptionMambaEncoder is the
last convolutional layer before the final projection. Based on the model:
  Default target: encoder.stages[-1][-1].inception.branch1.conv
  (The human review gate will confirm the correct layer name.)
If --target_layer is not provided, gradcam_runner.py should default to
this layer. If the layer name is wrong, Grad-CAM will fail gracefully
with a clear error message — do not silently fall back to random layers.

OUTPUT: List all files. Show test results. Show a sample report outline
(text description of what the PDF would contain for a mock case).
```

---

## Phase 8 Human Review Gate

Before Phase 9:

- [ ] Clinical team has reviewed a sample generated PDF report
- [ ] Disclaimer text appears correctly in the PDF
- [ ] Grad-CAM target layer confirmed by research team
- [ ] Generated heatmap visually makes sense on a sample image (manual check)
- [ ] Report template matches hospital's required format
- [ ] PDF contains no PHI beyond the anonymized case UID

**Human sign-off required before Phase 9 begins.**

---

---

# PHASE 9 — Security Hardening & Audit Logging

## Metadata
| Field | Value |
|---|---|
| Estimated agent time | ~2 hours |
| Human gate type | Security audit |
| Depends on | Phase 8 complete |
| Produces | Production-ready security layer |

## Human Input Required Before Starting

> ⚠️ **HUMAN DECISION REQUIRED:**

1. **Penetration test:** Will an external security firm conduct a pen test before go-live? (Recommended — the platform handles patient-adjacent data.)
2. **TLS certificate source:** Self-signed (dev/staging) or CA-signed (production)?
3. **Compliance standard:** Is HIPAA, GDPR, or another standard specifically required? This affects audit log retention and encryption requirements.

---

## Phase 9 Agent Prompt

```
PHASE 9: Security Hardening & Audit Logging

GOAL: Harden all security surfaces, complete audit logging coverage, and
add input validation to every endpoint that touches the filesystem.

[Paste the SYSTEM CONTEXT block here]

TASKS:

1. PATH TRAVERSAL PREVENTION (highest priority):
   Create platform/backend/app/core/path_validator.py:

   class SafePathValidator:
       def __init__(self, allowed_root: str):
           self.allowed_root = Path(allowed_root).resolve()

       def validate_and_resolve(self, user_path: str) -> Path:
           """
           Resolves the path and raises ValueError if it is outside
           allowed_root. Use for ANY path constructed from user input.
           Called before: image saves, checkpoint loads, result reads.
           """

       def is_safe_filename(self, filename: str) -> bool:
           """
           Returns False if filename contains: ../, .., \, null bytes,
           shell metacharacters. Use for uploaded filenames.
           """

   Apply SafePathValidator to:
   - POST /cases/ (image upload path)
   - GET /cases/{id}/image (serve image path)
   - GET /cases/{id}/gradcam (serve Grad-CAM path)
   - POST /cases/{id}/report (checkpoint path passed to report generator)
   - Any endpoint that constructs a filesystem path

2. AUDIT LOG COMPLETENESS:
   Verify every state-changing endpoint calls audit.log_action().
   Add a FastAPI middleware that logs all requests with:
   - method, path, status_code, user_id (if authenticated), ip_address,
     response_time_ms
   Write to a separate access_log table (not AuditLog — different purpose).

3. SECURITY HEADERS MIDDLEWARE:
   Add FastAPI middleware that sets on every response:
   - X-Content-Type-Options: nosniff
   - X-Frame-Options: DENY
   - X-XSS-Protection: 1; mode=block
   - Referrer-Policy: strict-origin-when-cross-origin
   - Content-Security-Policy: (appropriate for the app's static assets)
   - Strict-Transport-Security: max-age=31536000 (HTTPS only)

4. INPUT VALIDATION HARDENING:
   Review all Pydantic schemas and add:
   - max_length constraints on all string fields
   - min/max on all numeric fields (e.g., num_classes: int, min=2, max=1000)
   - email validator on User.email
   - Regex validator on Hospital.name (alphanumeric + spaces + hyphens only)
   - File size validator in POST /cases/ (max 50MB, configurable)
   - File extension whitelist in POST /cases/ (configurable, default: png, jpg,
     jpeg, npy)

5. RATE LIMITING:
   Using slowapi, apply limits:
   - POST /auth/login: 5/minute per IP
   - POST /cases/: 20/minute per user
   - POST /cases/{id}/report: 10/minute per user
   - GET /cases/{id}/image: 60/minute per user

6. SUBPROCESS INJECTION PREVENTION:
   In fl_runner.py and all places where subprocess is called:
   - NEVER use shell=True
   - NEVER interpolate user strings into command lists
   - All paths passed to subprocesses must be validated by SafePathValidator
   - The YAML config file is written to disk and passed as --config path,
     never as inline shell arguments

7. ENCRYPTION AT REST:
   Document (do not implement yet — requires infrastructure decision) the
   recommended encryption strategy for:
   - Images in FL_OUTPUT_DIR
   - Checkpoint files
   - Database fields containing sensitive values
   Write this as platform/SECURITY_NOTES.md

8. Write security tests in platform/backend/tests/test_security.py:
   - Test path traversal: POST /cases/ with filename "../../etc/passwd" → 422
   - Test path traversal: case_id with embedded "../" → 400
   - Test security headers present on all responses
   - Test rate limit: 6th login attempt returns 429
   - Test that shell metacharacters in hospital name are rejected (422)
   - Test that a 100MB file upload is rejected (413)

OUTPUT: List all files modified or created. Show security test results
(all must pass). List every endpoint that calls audit.log_action().
```

---

## Phase 9 Human Review Gate

Before Phase 10:

- [ ] Security team has reviewed path traversal protection
- [ ] All audit log tests pass
- [ ] Rate limiting confirmed active (test with curl)
- [ ] Security headers present in browser dev tools
- [ ] Subprocess injection prevention code reviewed
- [ ] SECURITY_NOTES.md reviewed by IT team for completeness

**Human sign-off required before Phase 10 begins.**

---

---

# PHASE 10 — Integration Testing & Deployment Configuration

## Metadata
| Field | Value |
|---|---|
| Estimated agent time | ~2 hours |
| Human gate type | Final sign-off before production deployment |
| Depends on | All previous phases complete |
| Produces | End-to-end test suite + production Docker configuration |

## Human Input Required Before Starting

> ⚠️ **HUMAN DECISION REQUIRED:**

1. **Production environment details:** Server specs, domain name, TLS certificate paths.
2. **Backup strategy:** Database and checkpoint backup schedule/destination.
3. **Go-live approval:** All clinical, legal, security, and IT stakeholders must sign off before this phase's output is deployed to production.

---

## Phase 10 Agent Prompt

```
PHASE 10: Integration Testing & Deployment Configuration

GOAL: Write the end-to-end integration test suite that validates the full
user journey for all three roles, and produce production-ready deployment
configuration.

[Paste the SYSTEM CONTEXT block here]

END-TO-END TEST SCENARIOS:

Write platform/backend/tests/test_e2e.py covering these full user journeys:

Scenario 1 — Admin creates and runs an experiment:
1. POST /auth/login as admin → get tokens
2. POST /hospitals/ → create Hospital A
3. POST /users/ → create hospital_manager for Hospital A
4. POST /experiments/ → create experiment (PRETRAIN, FEDAVG, n_clients=2)
5. POST /experiments/{id}/start → enqueue task (mock Celery, don't run real script)
6. Assert experiment status = RUNNING in DB
7. Simulate 3 rounds: directly write FLRound records to DB, publish SSE events
8. Assert GET /experiments/{id}/rounds returns 3 rounds
9. Write ModelCheckpoint record (simulate script completion)
10. POST /experiments/{id}/stop
11. Assert experiment status = COMPLETED

Scenario 2 — Hospital manager monitors participation:
1. POST /auth/login as hospital_manager
2. GET /hospitals/me/participation → assert experiment visible
3. GET /experiments/{id} → assert can see (is participant)
4. GET /experiments/{other_id} → assert 403 (not participant)
5. POST /hospitals/me/doctors → create doctor
6. POST /auth/login as new doctor → assert success

Scenario 3 — Doctor uploads case and gets diagnosis:
1. POST /auth/login as doctor
2. POST /cases/ with test image → get case_id, status=PENDING
3. Mock inference task: write DiagnosticResult to DB directly
4. GET /cases/{case_id} → assert status=COMPLETE, prediction visible
5. POST /cases/{case_id}/report → assert 200, content-type=application/pdf
6. Assert AuditLog has entries for upload and report generation

Scenario 4 — Access control matrix:
Test every (role, endpoint) pair in the access matrix and confirm
the correct HTTP status code (200, 403, 404, 401).
Generate a coverage table in the test output.

PRODUCTION DEPLOYMENT:

1. Update platform/docker-compose.yml for production:
   - Remove all development volumes
   - Add restart: always to all services
   - Add resource limits (memory/cpu) to all services
   - Remove exposed ports except 80/443 on nginx

2. Create platform/nginx/nginx.conf:
   - Reverse proxy to FastAPI backend
   - Serve React frontend static files
   - TLS configuration (certificate paths from env)
   - Rate limiting at nginx level (complement to app-level)
   - Gzip compression for static assets
   - Cache-Control headers for static assets

3. Create platform/docker-compose.prod.yml override:
   - Uses gunicorn with uvicorn workers for FastAPI
   - Sets WORKERS = (2 * CPU_COUNT) + 1
   - Disables debug mode
   - Uses production SECRET_KEY from env (not default)

4. Create platform/backend/app/core/healthcheck.py:
   GET /health/ready endpoint that checks:
   - DB connection (simple SELECT 1)
   - Redis connection (PING)
   - FL_OUTPUT_DIR exists and is writable
   - At least one deployed ModelCheckpoint exists (warn, not fail)
   Returns 200 if all critical checks pass, 503 if any critical check fails.

5. Create platform/DEPLOYMENT.md documenting:
   - Required environment variables (with descriptions)
   - Database setup steps (alembic upgrade head)
   - First-time admin user creation (CLI command)
   - How to add the MAE teacher checkpoint
   - How to register the first hospital
   - Backup procedure
   - Log file locations

6. Create platform/backend/app/cli.py (Click CLI):
   - `python -m app.cli create-admin --email X --password Y`
     Creates the first admin user, bypassing the normal registration flow.
     Required for initial system setup.
   - `python -m app.cli sync-checkpoints --experiment-id X`
     Calls the sync_experiment_checkpoints Celery task.
   - `python -m app.cli health-check`
     Runs the same checks as /health/ready and prints a report.

FINAL CHECKLIST (agent verifies each item):
- [ ] All 227+ tests pass (count from all phases)
- [ ] `docker-compose -f docker-compose.yml -f docker-compose.prod.yml up --build`
      completes without errors
- [ ] GET /health/ready returns 200 in production config
- [ ] create-admin CLI command creates a working admin account
- [ ] E2E Scenario 1-4 all pass
- [ ] No hardcoded secrets anywhere in the codebase
       (`grep -r "SECRET\|PASSWORD\|API_KEY" platform/ --include="*.py"
        | grep -v ".env\|config.py\|test_\|#"` returns nothing)

OUTPUT: Final test summary with pass/fail counts per phase.
List any tests that could not be written without real research script
execution (document as "requires manual integration testing").
```

---

## Phase 10 Human Review Gate (Final Sign-Off)

> ⚠️ **ALL of the following humans must sign off before production deployment:**

- [ ] **Clinical Lead:** Clinical portal meets patient safety requirements; disclaimer text approved
- [ ] **Data Governance Officer:** Patient data handling compliant with policy; no PHI in unexpected places
- [ ] **Security Officer:** Security hardening reviewed; pen test scheduled or completed
- [ ] **Research Team:** Adapter layer does not modify research code; YAML config generation is correct
- [ ] **Hospital IT Representatives:** Portal usable by hospital managers; API key distribution process agreed
- [ ] **Platform Administrator:** Deployment docs followed; admin account created; health check passes

---

---

## Appendix A: Agent Session Management Guidelines

When running any phase prompt with an AI coding agent, follow these practices:

**At session start:**
1. Always paste the full SYSTEM CONTEXT block
2. Paste the phase prompt in full
3. Provide the output of the previous phase's file listing as confirmation of starting state

**During the session:**
- If the agent asks about a research script's behavior, point it to the source file — do not allow it to guess
- If the agent proposes modifying a research file, stop it immediately and remind it of Constraint #1
- If the agent's response exceeds its context limit, split the phase into two sessions at a logical boundary

**At session end:**
- Ask the agent to produce a "phase completion report": files created, tests passed, open questions
- Record any open questions for the human review gate
- Do not start the next phase until the human review gate is complete

---

## Appendix B: Research Code Change Management

Since the research codebase is under active development, the following process protects the platform:

1. **When a research script's CLI changes** (new flag, removed flag, renamed flag):
   - The change affects `fl_runner.py`'s `build_command()` method only
   - Update the command builder; no other platform code changes needed

2. **When a research script's output CSV format changes** (new column, renamed column):
   - The change affects `fl_monitor.py`'s `parse_latest_round()` method only
   - Update the column mapping dict; no other platform code changes needed

3. **When a new research script is added** (e.g., a new FL algorithm):
   - Add a new entry to `fl_runner.py`'s `SCRIPT_MAP`
   - Add the new algorithm enum value to `FLExperiment.algorithm`
   - Write a new Alembic migration for the schema change
   - No other platform code changes needed

4. **When model architecture changes** (new InceptionMambaEncoder config):
   - `single_image_inference.py` and `gradcam_runner.py` import `load_encoder()`
     from `eval.linear_probe` — they automatically pick up architecture changes
   - No platform code changes needed unless the inference API changes

This isolation is the core value of the adapter architecture.

---

## Appendix C: Phase Dependency Graph

```
Phase 0 (Scaffold)
    └── Phase 1 (Database)
            └── Phase 2 (Auth)
                    └── Phase 3 (FL Adapter)  ← most critical
                            ├── Phase 4 (Admin UI)
                            ├── Phase 5 (Hospital UI)
                            └── Phase 6 (Doctor UI)
                                    └── Phase 7 (Monitoring)
                                            └── Phase 8 (XAI + Reports)
                                                    └── Phase 9 (Security)
                                                            └── Phase 10 (Deploy)
```

Phases 4, 5, and 6 can be parallelized once Phase 3 is complete and reviewed.

---

*Document version: 1.0 | Generated for FedMamba-SALT Clinical Platform*
*Research codebase: FedMamba-SALT (active development — platform must remain decoupled)*
