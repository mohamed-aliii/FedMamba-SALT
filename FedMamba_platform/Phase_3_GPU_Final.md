# PHASE 3 — FL Backend Adapter Layer (GPU Edition)

## Decisions — Locked In

| Decision | Value |
|---|---|
| Execution environment | Same machine as platform (no SSH, no SLURM) |
| GPU availability | **CUDA GPU — required** |
| Output directory | Auto-discovered by scanning RESEARCH_ROOT |

---

## Pre-Phase Migration Note

The previous CPU version of this prompt hardcoded `device: cpu` everywhere.
**All of those are now reversed.** Every place the old prompt said "CPU lock"
or "override to cpu" is now a **CUDA availability pre-flight check** instead.
If `torch.cuda.is_available()` returns False at launch time, the experiment
must fail immediately with a clear error — never silently fall back to CPU,
because GPU training assumptions (batch size, workers, memory) will produce
wrong behaviour on CPU.

---

## SYSTEM CONTEXT

```
SYSTEM CONTEXT (paste at top of every agent session):
You are building the FedMamba-SALT Clinical Platform — a web app that wraps
an existing federated learning research codebase.

CRITICAL CONSTRAINTS:
1. NEVER modify files in: augmentations/, models/, objectives/, utils/, eval/,
   train_centralized.py, train_fedavg.py, train_fed_finetune.py
2. The platform launches research scripts as subprocesses. It reads their
   output files and stdout. It never imports their modules.
3. All FL configs are generated as YAML files passed via --config flag.
4. Training runs on CUDA GPU. If CUDA is unavailable at launch, fail fast.
5. Research root layout:
   train_centralized.py, train_fedavg.py, train_fed_finetune.py (root)
   eval/linear_probe.py, eval/eval_tta.py
   models/, augmentations/, objectives/, utils/
```

---

## PHASE 3 AGENT PROMPT

```
PHASE 3: FL Backend Adapter Layer

GOAL: Bridge the platform to the research scripts. Same machine. CUDA GPU.
Output directories auto-discovered from RESEARCH_ROOT.

[Paste SYSTEM CONTEXT here]

LOCKED:
- subprocess.Popen, no SSH
- device: cuda (pre-flight check; fail hard if unavailable)
- Output dirs: scan RESEARCH_ROOT for marker files

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — DB Migration (add to FLExperiment via Alembic)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
New columns:
  pid:        Integer, nullable   (subprocess PID)
  output_dir: String,  nullable   (absolute path, set at launch)

These were omitted from Phase 1. Create the migration now.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2 — platform/backend/app/core/fl_config.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GPU_DEFAULTS = {
    "device":      "cuda",
    "num_workers": 4,      # parallel data loading (GPU needs fed fast)
    "pin_memory":  True,   # faster host→device transfer
    "batch_size":  128,    # standard GPU batch
}

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

class FLConfigGenerator:

    def __init__(self, research_root: str):
        self.research_root = Path(research_root).resolve()

    def get_script_path(self, phase: str, algorithm: str) -> Path:
        """Resolve absolute path to research script. Raise FileNotFoundError
        with helpful message if missing (wrong RESEARCH_ROOT)."""

    def generate_output_dir(self, experiment_id: str) -> Path:
        """
        Returns: {RESEARCH_ROOT}/outputs/platform/{experiment_id}/
        Creates directory immediately. Raises if not writable.
        """

    def generate_config(
        self,
        experiment: FLExperiment,
        output_dir: Path,
        data_path: str,
        extra_paths: dict,  # {"teacher_ckpt": "...", "encoder_ckpt": "..."}
    ) -> Path:
        """
        Write config.yaml to output_dir. Return its path.

        Merge order (later wins):
          1. GPU_DEFAULTS
          2. _experiment_to_params(experiment)
          3. extra_paths
          4. experiment.config_yaml  (research team — always wins)

        DEVICE RULE:
          device is always "cuda". If experiment.config_yaml contains
          "device: cpu", log a WARNING and override back to "cuda".
          Rationale: CPU fallback silently corrupts GPU-tuned hyperparams
          (batch_size=128, pin_memory=True make no sense on CPU).
          The admin must explicitly reconfigure for CPU — not silently degrade.
        """

    def _experiment_to_params(self, experiment: FLExperiment) -> dict:
        """
        Map experiment fields → YAML keys:
          algorithm SCAFFOLD  → algo: scaffold
          algorithm FEDPROX   → mu: <from experiment config>
          phase FINETUNE      → mode: federated_finetune
        Unknown fields stay in experiment.config_yaml.
        """

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3 — platform/backend/app/core/output_discovery.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MARKER_FILES = {
    "ckpt_latest.pth":          "has_checkpoint",
    "ckpt_best.pth":            "has_best_checkpoint",
    "ckpt_best_finetune.pth":   "has_best_finetune_checkpoint",
    "training_metrics.csv":     "centralized_metrics",
    "federated_metrics.csv":    "fedavg_metrics",
    "fed_finetune_metrics.csv": "finetune_metrics",
    "config.yaml":              "platform_managed",
}

CSV_FOR_SCRIPT = {
    "train_centralized.py":  "training_metrics.csv",
    "train_fedavg.py":       "federated_metrics.csv",
    "train_fed_finetune.py": "fed_finetune_metrics.csv",
}

class OutputDirectoryScanner:

    def __init__(self, research_root: str, max_depth: int = 4):
        self.research_root = Path(research_root).resolve()
        self.max_depth = max_depth

    def scan(self) -> list[dict]:
        """
        Walk RESEARCH_ROOT up to max_depth. Return dirs that contain
        at least one MARKER_FILE.

        Each result:
        {
          "path": str,                  # absolute
          "markers": list[str],
          "is_platform_managed": bool,  # config.yaml present
          "metrics_csv": str | None,
          "checkpoints": list[str],
        }

        EXCLUDE: platform/, __pycache__, .git, node_modules, .venv
        """

    def classify_script(self, directory_info: dict) -> str | None:
        """
        "centralized_metrics" in markers → train_centralized.py
        "fedavg_metrics" in markers      → train_fedavg.py
        "finetune_metrics" in markers    → train_fed_finetune.py
        Else: read config.yaml for 'algo'/'mode' field. Else None.
        """

    def find_experiment_output(self, experiment_id: str) -> dict | None:
        """Find output dir for a platform-managed experiment by id in path
        or inside config.yaml."""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 4 — platform/backend/app/core/fl_runner.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# GPU is fast — use tighter values than CPU equivalents
POLLING_INTERVAL_SECONDS  = 10    # GPU rounds complete in seconds/minutes
PROCESS_TIMEOUT_HOURS     = 24    # GPU runs rarely exceed 24h
STALLED_THRESHOLD_MINUTES = 20    # no output for 20 min on GPU = problem
LOG_BUFFER_LINES          = 500

class FLScriptRunner:

    def __init__(self, research_root: str):
        self.research_root = Path(research_root).resolve()
        self.python_executable = sys.executable
        # Same python as platform → same venv → torch, mamba_ssm available

    def build_command(self, script_path: Path, config_path: Path) -> list[str]:
        """
        Return: [sys.executable, str(script_path), "--config", str(config_path)]
        NEVER shell=True. NEVER string interpolation. List only.
        Both paths validated by SafePathValidator before this is called.
        """

    def launch(self, command: list[str], experiment_id: str,
               output_dir: Path) -> subprocess.Popen:
        """
        subprocess.Popen config:
          stdout=PIPE, stderr=STDOUT  (merged stream)
          cwd=self.research_root      (CRITICAL: scripts use relative paths)
          env: inherit + PYTHONPATH=research_root
          start_new_session=True      (kill entire process group incl. workers)
          bufsize=1, text=True        (line-buffered)

        After launch:
          Store PID → Redis key: experiment:{id}:pid
          Background thread: read stdout line-by-line →
            push to Redis list experiment:{id}:logs (trim to LOG_BUFFER_LINES)
        """

    def terminate(self, experiment_id: str) -> None:
        """
        1. Get PID from Redis experiment:{id}:pid
        2. os.killpg(os.getpgid(pid), SIGTERM)  ← kills DataLoader workers too
        3. Wait 10s for clean shutdown
        4. If alive: os.killpg(..., SIGKILL)
        5. Clean Redis keys for this experiment
        """

    def is_alive(self, experiment_id: str) -> bool:
        """PID from Redis. os.kill(pid, 0) — no signal, just liveness check.
        Return False if PID missing or OSError."""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 5 — platform/backend/app/core/fl_monitor.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Verified column names from the actual research script source files.
CSV_COLUMN_MAPS = {
    "training_metrics.csv": {
        "round":   "epoch",
        "loss":    "loss",
        "val_acc": None,       # pretrain has no val_acc
        "val_auc": None,
        "lr":      "lr",
        "time_s":  "epoch_time_s",
        "enc_std": "student_std",
    },
    "federated_metrics.csv": {
        "round":   "round",
        "loss":    "avg_loss",
        "val_acc": None,
        "val_auc": None,
        "lr":      "lr",
        "time_s":  "round_time_s",
        "enc_std": "avg_enc_std",
    },
    "fed_finetune_metrics.csv": {
        "round":   "round",
        "loss":    "val_loss_weighted",
        "val_acc": "val_acc",
        "val_auc": "auc",
        "lr":      "cls_lr",
        "time_s":  "round_time_s",
        "enc_std": None,
    },
}

class FLOutputMonitor:

    def __init__(self, scanner: OutputDirectoryScanner):
        self.scanner = scanner

    def find_metrics_csv(self, output_dir: str) -> tuple[str, str] | None:
        """Check in order (most specific first):
        fed_finetune_metrics.csv → federated_metrics.csv → training_metrics.csv
        Return (absolute_path, csv_filename) or None."""

    def parse_latest_round(self, csv_path: str) -> dict | None:
        """
        Read ONLY the last row (memory-efficient):
            with open(csv_path) as f:
                *_, last = csv.DictReader(f)
        Apply CSV_COLUMN_MAPS. Return normalized dict:
        {round, loss, val_acc, val_auc, lr, time_s, enc_std, raw}
        All numeric values cast to float. None for unmapped columns.
        """

    def parse_all_rounds(self, csv_path: str) -> list[dict]:
        """All rows normalized. Called once on chart initial load.
        SSE handles incremental updates after that."""

    def find_checkpoints(self, output_dir: str) -> list[dict]:
        """
        Scan for ckpt_*.pth files. Classify by filename:
          ckpt_latest.pth         → LATEST,   round=None
          ckpt_best.pth           → BEST,     round=None
          ckpt_best_finetune.pth  → BEST,     round=None
          ckpt_epoch_NNNN.pth     → PERIODIC, round=NNNN
          ckpt_round_NNNN.pth     → PERIODIC, round=NNNN

        Return list of {filename, path, type, round_number, size_mb, modified_at}
        """

    def parse_stdout_line(self, line: str) -> dict | None:
        """
        Real-time metric extraction from stdout.

        Known log formats (from actual research script f-strings):

        train_centralized.py (~line 315):
          "  Epoch [ 10/100]  loss=0.0423  align=0.0401  var=0.0022
           enc_std=0.2341  ...  lr=5.00e-04  time=12.3s  gpu=1024MB"
          NOTE: gpu field will now show real MB usage (not 0MB as on CPU)

        train_fedavg.py (~line 420):
          "  Round [ 10/200]  loss=0.0512  enc_std=0.1923  ...
           lr=5.00e-04  time=45.2s  c1=0.0521  c2=0.0498 ..."

        train_fed_finetune.py (~line 680):
          "  Round [ 10/ 50]  train_acc=72.45%  val=78.23%
           auc=0.8341  ...  enc_lr=1.0e-04  cls_lr=1.0e-03  time=23.1s"

        Parse with regex. Return None if no format matches.
        Return same keys as parse_latest_round() plus "source_script".
        """

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 6 — platform/backend/app/tasks/fl_tasks.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@celery_app.task(bind=True, max_retries=0, name="fl.run_experiment")
def run_fl_experiment(self, experiment_id: str) -> None:
    """
    PRE-FLIGHT CHECKS (fail fast — set status=FAILED if any fails):
    1. Experiment exists in DB with status=PENDING
    2. CUDA available: torch.cuda.is_available() must be True
       Error message: "CUDA GPU required but not available. Check drivers."
    3. Script exists: FLConfigGenerator.get_script_path()
    4. data_path exists and contains labels.csv
    5. For FINETUNE: encoder_ckpt exists
    6. RESEARCH_ROOT contains train_centralized.py (sanity check)

    LAUNCH:
    7.  Generate output_dir
    8.  Write config.yaml (device: cuda enforced)
    9.  Build command list
    10. Set experiment: status=RUNNING, output_dir=..., started_at=now()
    11. Launch subprocess (cwd=research_root)
    12. Set experiment: pid=process.pid

    MONITORING LOOP (poll every POLLING_INTERVAL_SECONDS=10):
    while process is alive:
        find + parse latest metrics CSV row
        if new round found:
            write FLRound to DB
            publish SSE event to Redis channel experiment:{id}
        if time_since_last_update > STALLED_THRESHOLD_MINUTES * 60:
            publish_alert(experiment_id, "STALLED")
        if elapsed > PROCESS_TIMEOUT_HOURS * 3600:
            runner.terminate(experiment_id)
            set_failed("Exceeded maximum runtime")
            return

    POST-PROCESS:
    exit_code = process.wait()
    scan output_dir for checkpoints → upsert ModelCheckpoint records
    experiment.status = COMPLETED if exit_code == 0 else FAILED
    if FAILED: store last 50 log lines from Redis as error_message
    experiment.completed_at = now()
    publish SSE: {"type": "experiment_complete", "status": "..."}
    """

@celery_app.task(name="fl.stop_experiment")
def stop_fl_experiment(experiment_id: str) -> None:
    """runner.terminate(experiment_id). Set status=PAUSED (not FAILED)."""

@celery_app.task(name="fl.sync_checkpoints")
def sync_experiment_checkpoints(experiment_id: str) -> None:
    """Re-scan output_dir and sync ModelCheckpoint records. Used for
    recovery and for manually-run experiments."""

@celery_app.task(name="fl.startup_recovery")
def recover_orphaned_experiments() -> None:
    """
    Runs on Celery worker startup (via worker_ready signal).

    1. validate_environment() — fail hard if research scripts missing
    2. Find all experiments with status=RUNNING
       If pid is dead (os.kill(pid,0) raises OSError): set status=FAILED
    3. Run OutputDirectoryScanner.scan() — import any manually-run
       experiment directories not yet in the DB as status=IMPORTED
    """

# Register in celery_worker.py:
# from celery.signals import worker_ready
# @worker_ready.connect
# def on_worker_ready(**kwargs):
#     recover_orphaned_experiments.delay()

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 7 — platform/backend/app/api/v1/experiments.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Endpoints (unchanged from master document):
  POST   /experiments/              ADMIN — create, enqueue task
  GET    /experiments/              ADMIN=all; HOSPITAL_MANAGER=own
  GET    /experiments/{id}          role-filtered
  PATCH  /experiments/{id}          ADMIN — update config before start
  POST   /experiments/{id}/start    ADMIN — enqueue run_fl_experiment
  POST   /experiments/{id}/stop     ADMIN — enqueue stop_fl_experiment
  GET    /experiments/{id}/rounds   paginated FLRound list
  GET    /experiments/{id}/checkpoints
  GET    /experiments/{id}/logs     SSE stream from Redis list

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 8 — platform/backend/app/api/v1/sse.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  GET /sse/experiments/{id}
  Subscribe to Redis channel experiment:{id}.
  Forward each message as SSE event to browser.
  Handle reconnect. Clean up on disconnect.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ENVIRONMENT VALIDATION (inside recover_orphaned_experiments)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def validate_environment() -> None:
    checks = {
        "research_root_exists":  Path(RESEARCH_ROOT).exists(),
        "train_centralized":     (RESEARCH_ROOT / "train_centralized.py").exists(),
        "train_fedavg":          (RESEARCH_ROOT / "train_fedavg.py").exists(),
        "train_fed_finetune":    (RESEARCH_ROOT / "train_fed_finetune.py").exists(),
        "torch_importable":      _can_import("torch"),
        "cuda_available":        torch.cuda.is_available(),      # HARD FAIL if False
        "mamba_ssm_available":   _can_import("mamba_ssm"),       # EXPECTED True on GPU
        "output_dir_writable":   os.access(FL_OUTPUT_DIR, os.W_OK),
    }
    # HARD FAIL: research scripts missing, CUDA unavailable, output not writable
    # WARN only: nothing (all GPU checks are hard fails)
    # Note: mamba_ssm is expected to be installed and working on a CUDA machine.
    #   If it fails to import, log ERROR (not warning) — it means the venv
    #   is misconfigured and the model will use a linear mock silently.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STDOUT PARSING NOTE — GPU-specific
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
On GPU, research scripts print real GPU memory usage:
  "gpu=1024MB" (train_centralized.py)
The parse_stdout_line regex must capture this field and surface it
in the SSE event as "gpu_mb": int so the monitoring dashboard can
display live GPU memory usage.

On CPU this field was always "gpu=0MB" — meaningless. On GPU it is
a real diagnostic value. Include it in the FLRound.metrics_json blob.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TESTS — platform/backend/tests/test_fl_adapter.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Test 1: device is always cuda in generated config
  - Create experiment with no config_yaml override
  - Call generate_config()
  - Assert YAML has device: cuda

Test 2: device: cpu in config_yaml is overridden to cuda with warning
  - Create experiment with config_yaml: "device: cpu\nlr: 0.001"
  - Call generate_config()
  - Assert resulting YAML has device: cuda
  - Assert WARNING was logged mentioning the override

Test 3: Research team override wins for all fields except device
  - config_yaml: "lr: 0.001\nbatch_size: 64"
  - GPU_DEFAULTS has batch_size: 128
  - Assert batch_size: 64 (override wins)
  - Assert device: cuda (device lock always wins)

Test 4: get_script_path raises FileNotFoundError with helpful message
  - Set RESEARCH_ROOT to empty temp dir
  - Assert error message contains the attempted path and RESEARCH_ROOT value

Test 5: OutputDirectoryScanner finds and excludes correctly
  Create temp structure:
    tmp/outputs/exp1/ckpt_latest.pth       ← must be found
    tmp/outputs/exp1/training_metrics.csv  ← must be found
    tmp/eval_results/run1/fed_finetune_metrics.csv ← must be found
    tmp/platform/something.py              ← must be EXCLUDED
    tmp/__pycache__/cache.pyc              ← must be EXCLUDED
  Assert scan() returns exactly 2 results (exp1, run1).

Test 6: CSV column mapping for all 3 scripts
  Fixture CSV rows — EXACT headers from research source files:

  training_metrics.csv header:
    epoch,loss,student_std,teacher_std,salt_norm_mode,
    salt_teacher_std_mean,salt_teacher_std_min,salt_teacher_std_max,
    salt_teacher_target_finite,salt_student_centered_finite,
    lr,epoch_time_s,gpu_mem_allocated_mb,gpu_mem_reserved_mb,gpu_mem_peak_mb

  federated_metrics.csv header:
    round,avg_loss,avg_enc_std,avg_teacher_std,salt_norm_mode,
    salt_teacher_std_mean,salt_teacher_std_min,salt_teacher_std_max,
    salt_teacher_target_finite,salt_student_centered_finite,
    student_update_norms,projector_update_norms,lr,round_time_s,gpu_mb,
    client_1_loss,...

  fed_finetune_metrics.csv header:
    round,val_acc,val_loss_weighted,val_loss_unweighted,balanced_acc,auc,
    prediction_hist,per_class_recall,per_class_f1,per_class_support,
    feature_norm_mean,feature_std_mean,head_weight_norms,head_biases,
    encoder_update_norms,classifier_update_norms,
    enc_lr,cls_lr,round_time_s,gpu_mb,client_1_loss,...

  Assert parse_latest_round() returns correctly normalized dict for each.

Test 7: Checkpoint classification
  ckpt_latest.pth         → LATEST,   round=None
  ckpt_best.pth           → BEST,     round=None
  ckpt_best_finetune.pth  → BEST,     round=None
  ckpt_epoch_0010.pth     → PERIODIC, round=10
  ckpt_round_0050.pth     → PERIODIC, round=50

Test 8: FLScriptRunner.build_command structure
  - Assert type is list, not str
  - Assert no shell=True used anywhere in the class
  - Assert command[0] == sys.executable
  - Assert "--config" present followed by a path string

Test 9: recover_orphaned_experiments marks dead processes as FAILED
  - Insert FLExperiment: status=RUNNING, pid=99999 (guaranteed non-existent)
  - Run recover_orphaned_experiments()
  - Assert experiment.status == FAILED
  - Assert experiment.error_message contains "Process terminated unexpectedly"

Test 10: pre-flight CUDA check fails fast when CUDA unavailable
  - Mock torch.cuda.is_available() to return False
  - Call run_fl_experiment (synchronously, not via Celery)
  - Assert experiment.status == FAILED
  - Assert error message contains "CUDA GPU required"
  - Assert no subprocess was launched

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT REQUIREMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. List all files created with paths.
2. Show all 10 tests passing.
3. Run OutputDirectoryScanner.scan() against the REAL research root
   and print what it finds.
4. Print torch.cuda.is_available() and torch.cuda.get_device_name(0)
   from sys.executable to confirm GPU is visible to the subprocess venv.
5. Confirm sys.executable path.
```

---

## Human Review Gate — Phase 3

Before moving to Phase 4, verify:

- [ ] All 10 tests pass
- [ ] `torch.cuda.is_available()` returns True in the platform's venv
- [ ] `mamba_ssm` imports successfully (not falling back to linear mock)
- [ ] Scanner correctly finds existing research output directories
- [ ] A real small experiment runs end-to-end: create → start → monitor → complete
- [ ] SSE stream delivers round updates to a browser tab during that test run
- [ ] GPU memory values appear in the SSE events (not 0MB)
- [ ] No research script file was modified (confirm with `git diff`)

---

## What Changed from CPU Version — Summary

| Item | CPU version | GPU version |
|---|---|---|
| `device` default | `"cpu"` | `"cuda"` |
| `num_workers` | `0` | `4` |
| `pin_memory` | `False` | `True` |
| `batch_size` default | `16` | `128` |
| Device override rule | Lock to cpu, warn on cuda | Lock to cuda, warn on cpu |
| CUDA pre-flight check | Not present | Hard fail if unavailable |
| `POLLING_INTERVAL_SECONDS` | `30` | `10` |
| `PROCESS_TIMEOUT_HOURS` | `48` | `24` |
| `STALLED_THRESHOLD_MINUTES` | `60` | `20` |
| `mamba_ssm` missing | Warning (expected on CPU) | ERROR (unexpected on GPU) |
| `gpu=NNNmb` in stdout | Always `0MB`, ignored | Real value, capture + surface |
| Test 1 assertion | `device == cpu` | `device == cuda` |
| Test 2 assertion | `cuda` overridden to `cpu` | `cpu` overridden to `cuda` |
| New test | — | Test 10: CUDA unavailable → fail fast |
