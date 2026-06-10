"""
Celery tasks for FL experiment lifecycle management.

CRITICAL: This module NEVER imports from the research codebase.
Research scripts are launched as subprocesses only.
"""
import json
import logging
import os
import time
from datetime import datetime, timezone

import redis as redis_lib

# Celery is an optional runtime dependency — not required for unit tests.
# If unavailable, task functions are plain callables (no broker needed).
try:
    from celery_worker import celery_app
except ImportError:  # pragma: no cover
    class _MockCeleryApp:
        """Passthrough mock so @celery_app.task works without celery installed."""
        def task(self, *args, **kwargs):
            def decorator(fn):
                fn.delay = fn  # tests can call .delay() which is just the fn
                return fn
            return decorator
    celery_app = _MockCeleryApp()  # type: ignore[assignment]
from app.config import get_settings
from app.core.fl_config import FLConfigGenerator
from app.core.fl_monitor import FLOutputMonitor
from app.core.fl_runner import (
    FLScriptRunner,
    POLLING_INTERVAL_SECONDS,
    PROCESS_TIMEOUT_HOURS,
    STALLED_THRESHOLD_MINUTES,
    LOG_BUFFER_LINES,
)
from app.core.output_discovery import OutputDirectoryScanner
from app.core.safe_path import SafePathValidator
from app.database import SessionLocal
from app.models.experiment import ExperimentStatus, FLExperiment
from app.models.round import FLRound, RoundStatus
from app.models.checkpoint import ModelCheckpoint

# Import torch at module level so it can be patched in tests.
# Use try/except because torch is not required at import time (only at task execution).
try:
    import torch as torch  # noqa: PLC0414
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)
settings = get_settings()


def _get_redis() -> redis_lib.Redis:
    return redis_lib.from_url(settings.REDIS_URL, decode_responses=True)


def _set_failed(db, experiment: FLExperiment, message: str) -> None:
    experiment.status = ExperimentStatus.FAILED
    experiment.error_message = message
    experiment.completed_at = datetime.now(timezone.utc)
    db.commit()


def _publish(redis: redis_lib.Redis, experiment_id: str, event: dict) -> None:
    redis.publish(f"experiment:{experiment_id}", json.dumps(event))


# ---------------------------------------------------------------------------
# Main experiment task
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, max_retries=0, name="fl.run_experiment")
def run_fl_experiment(self, experiment_id: str) -> None:
    """Full FL experiment lifecycle: pre-flight → launch → monitor → post-process."""
    db = SessionLocal()
    redis = _get_redis()
    try:
        _run_experiment_impl(experiment_id, db, redis)
    finally:
        db.close()


def _run_experiment_impl(experiment_id: str, db, redis: redis_lib.Redis) -> None:
    import uuid as _uuid
    # ── Pre-flight checks ──────────────────────────────────────────────────
    experiment = db.query(FLExperiment).filter(
        FLExperiment.id == _uuid.UUID(experiment_id)
    ).first()
    if experiment is None:
        logger.error("Experiment %s not found", experiment_id)
        return
    if experiment.status != ExperimentStatus.PENDING:
        logger.warning(
            "Experiment %s is not PENDING (status=%s); skipping",
            experiment_id, experiment.status,
        )
        return

    # 1. CUDA availability — hard fail, never silent CPU fallback
    # Uses module-level `torch` import (allows patching in tests).
    if torch is None:
        _set_failed(db, experiment, "torch not importable — venv misconfigured.")
        return
    if not torch.cuda.is_available():
        _set_failed(
            db, experiment,
            "CUDA GPU required but not available. Check drivers and "
            "ensure the machine has a CUDA-capable GPU."
        )
        _publish(redis, experiment_id, {
            "type": "experiment_failed",
            "reason": "CUDA unavailable",
        })
        return

    # 2. Script exists
    config_gen = FLConfigGenerator(settings.RESEARCH_ROOT)
    try:
        script_path = config_gen.get_script_path(
            experiment.phase.value, experiment.algorithm.value
        )
    except (FileNotFoundError, ValueError) as e:
        _set_failed(db, experiment, str(e))
        return

    # 3. Research root sanity check
    import pathlib
    if not (pathlib.Path(settings.RESEARCH_ROOT) / "train_centralized.py").exists():
        _set_failed(
            db, experiment,
            f"RESEARCH_ROOT does not contain train_centralized.py: "
            f"'{settings.RESEARCH_ROOT}'"
        )
        return

    # 4. data_path (stored in config_yaml or we skip this check)
    data_path = settings.FL_OUTPUT_DIR  # default data location

    # ── Launch ────────────────────────────────────────────────────────────
    # 5. Generate output directory
    output_dir = config_gen.generate_output_dir(str(experiment.id))

    # 6. Write config.yaml (device: cuda enforced inside generate_config)
    config_path = config_gen.generate_config(
        experiment=experiment,
        output_dir=output_dir,
        data_path=data_path,
    )

    # 7. Validate paths (security)
    SafePathValidator.validate(script_path, settings.RESEARCH_ROOT)
    SafePathValidator.validate(config_path, settings.RESEARCH_ROOT)

    runner = FLScriptRunner(settings.RESEARCH_ROOT, redis)
    command = runner.build_command(script_path, config_path)

    # 8. Update DB: RUNNING, output_dir, started_at
    experiment.status = ExperimentStatus.RUNNING
    experiment.output_dir = str(output_dir)
    experiment.started_at = datetime.now(timezone.utc)
    db.commit()

    # 9. Launch subprocess
    process = runner.launch(command, str(experiment.id), output_dir)

    # 10. Store PID in DB
    experiment.pid = process.pid
    db.commit()

    _publish(redis, experiment_id, {
        "type": "experiment_started",
        "pid": process.pid,
        "output_dir": str(output_dir),
    })

    # ── Monitoring loop ───────────────────────────────────────────────────
    monitor = FLOutputMonitor(settings.RESEARCH_ROOT)
    last_round_seen = -1
    last_update_time = time.monotonic()
    start_time = time.monotonic()

    while process.poll() is None:
        time.sleep(POLLING_INTERVAL_SECONDS)

        elapsed = time.monotonic() - start_time

        # Timeout check
        if elapsed > PROCESS_TIMEOUT_HOURS * 3600:
            runner.terminate(str(experiment.id))
            _set_failed(
                db, experiment,
                f"Exceeded maximum runtime of {PROCESS_TIMEOUT_HOURS} hours."
            )
            _publish(redis, experiment_id, {"type": "experiment_timeout"})
            return

        # Parse latest metrics CSV
        csv_result = monitor.find_metrics_csv(str(output_dir))
        if csv_result:
            csv_path, csv_filename = csv_result
            metrics = monitor.parse_latest_round(csv_path)
            if metrics and metrics.get("round") is not None:
                current_round = int(metrics["round"])
                if current_round > last_round_seen:
                    last_round_seen = current_round
                    last_update_time = time.monotonic()

                    # Write FLRound to DB
                    fl_round = FLRound(
                        experiment_id=experiment.id,
                        round_number=current_round,
                        status=RoundStatus.COMPLETED,
                        avg_loss=metrics.get("loss"),
                        val_acc=metrics.get("val_acc"),
                        val_auc=metrics.get("val_auc"),
                        metrics_json=json.dumps(metrics.get("raw", {})),
                        completed_at=datetime.now(timezone.utc),
                    )
                    db.add(fl_round)
                    db.commit()

                    # Publish SSE event
                    _publish(redis, experiment_id, {
                        "type": "round_complete",
                        "round": current_round,
                        "loss": metrics.get("loss"),
                        "val_acc": metrics.get("val_acc"),
                        "val_auc": metrics.get("val_auc"),
                        "lr": metrics.get("lr"),
                        "time_s": metrics.get("time_s"),
                        "gpu_mb": metrics.get("gpu_mb"),
                    })

        # Stall detection
        idle_seconds = time.monotonic() - last_update_time
        if idle_seconds > STALLED_THRESHOLD_MINUTES * 60:
            logger.warning(
                "Experiment %s stalled — no output for %.0f minutes",
                experiment_id, idle_seconds / 60,
            )
            _publish(redis, experiment_id, {
                "type": "alert",
                "level": "STALLED",
                "idle_minutes": round(idle_seconds / 60, 1),
            })
            last_update_time = time.monotonic()  # reset so we don't spam

    # ── Post-process ──────────────────────────────────────────────────────
    exit_code = process.wait()

    # Sync checkpoints
    checkpoints = monitor.find_checkpoints(str(output_dir))
    for ckpt in checkpoints:
        existing = db.query(ModelCheckpoint).filter(
            ModelCheckpoint.experiment_id == experiment.id,
            ModelCheckpoint.filename == ckpt["filename"],
        ).first()
        if not existing:
            db.add(ModelCheckpoint(
                experiment_id=experiment.id,
                filename=ckpt["filename"],
                path=ckpt["path"],
                checkpoint_type=ckpt["type"],
                round_number=ckpt.get("round_number"),
                size_mb=ckpt.get("size_mb"),
            ))
    db.commit()

    if exit_code == 0:
        experiment.status = ExperimentStatus.COMPLETED
    else:
        experiment.status = ExperimentStatus.FAILED
        # Store last 50 log lines as error context
        recent_logs = runner.get_recent_logs(str(experiment.id), n=50)
        experiment.error_message = "\n".join(recent_logs)

    experiment.completed_at = datetime.now(timezone.utc)
    db.commit()

    _publish(redis, experiment_id, {
        "type": "experiment_complete",
        "status": experiment.status.value,
        "exit_code": exit_code,
    })


# ---------------------------------------------------------------------------
# Stop experiment
# ---------------------------------------------------------------------------

@celery_app.task(name="fl.stop_experiment")
def stop_fl_experiment(experiment_id: str) -> None:
    """Terminate a running experiment. Set status=PAUSED (not FAILED)."""
    redis = _get_redis()
    runner = FLScriptRunner(settings.RESEARCH_ROOT, redis)
    runner.terminate(experiment_id)

    db = SessionLocal()
    try:
        experiment = db.query(FLExperiment).filter(
            FLExperiment.id == experiment_id
        ).first()
        if experiment and experiment.status == ExperimentStatus.RUNNING:
            experiment.status = ExperimentStatus.PAUSED
            experiment.completed_at = datetime.now(timezone.utc)
            db.commit()
        _publish(redis, experiment_id, {"type": "experiment_stopped"})
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Checkpoint sync
# ---------------------------------------------------------------------------

@celery_app.task(name="fl.sync_checkpoints")
def sync_experiment_checkpoints(experiment_id: str) -> None:
    """Re-scan output_dir and sync ModelCheckpoint records."""
    db = SessionLocal()
    try:
        experiment = db.query(FLExperiment).filter(
            FLExperiment.id == experiment_id
        ).first()
        if not experiment or not experiment.output_dir:
            return

        monitor = FLOutputMonitor(settings.RESEARCH_ROOT)
        checkpoints = monitor.find_checkpoints(experiment.output_dir)
        for ckpt in checkpoints:
            existing = db.query(ModelCheckpoint).filter(
                ModelCheckpoint.experiment_id == experiment.id,
                ModelCheckpoint.filename == ckpt["filename"],
            ).first()
            if not existing:
                db.add(ModelCheckpoint(
                    experiment_id=experiment.id,
                    filename=ckpt["filename"],
                    path=ckpt["path"],
                    checkpoint_type=ckpt["type"],
                    round_number=ckpt.get("round_number"),
                    size_mb=ckpt.get("size_mb"),
                ))
        db.commit()
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Startup recovery
# ---------------------------------------------------------------------------

@celery_app.task(name="fl.startup_recovery")
def recover_orphaned_experiments() -> None:
    """Runs on Celery worker startup via worker_ready signal.

    1. validate_environment() — fail hard if research scripts missing or
       CUDA unavailable
    2. Find all experiments with status=RUNNING
       If PID is dead: set status=FAILED with error message
    3. Run OutputDirectoryScanner — import manually-run experiments
       not yet in DB as status=IMPORTED
    """
    _validate_environment()

    db = SessionLocal()
    try:
        running = db.query(FLExperiment).filter(
            FLExperiment.status == ExperimentStatus.RUNNING
        ).all()

        for experiment in running:
            if experiment.pid is None:
                _set_failed(db, experiment, "Process terminated unexpectedly (no PID recorded).")
                continue
            try:
                os.kill(experiment.pid, 0)
                # Process still alive — leave as-is
            except OSError:
                # PID is dead
                _set_failed(
                    db, experiment,
                    f"Process terminated unexpectedly (PID {experiment.pid} not found)."
                )
                logger.warning(
                    "Experiment %s marked FAILED — PID %s is dead",
                    experiment.id, experiment.pid,
                )

        db.commit()

        # Scan for manually-run experiments
        scanner = OutputDirectoryScanner(settings.RESEARCH_ROOT)
        found_dirs = scanner.scan()
        for dir_info in found_dirs:
            if not dir_info["is_platform_managed"]:
                # Check if already in DB by path
                existing = db.query(FLExperiment).filter(
                    FLExperiment.output_dir == dir_info["path"]
                ).first()
                if not existing:
                    logger.info(
                        "Discovered manually-run output dir: %s (not importing — no experiment record)",
                        dir_info["path"],
                    )
    finally:
        db.close()


def _validate_environment() -> None:
    """Validate the environment on Celery worker startup.

    Hard fails: research scripts missing, CUDA unavailable, output not writable.
    """
    import pathlib
    research_root = pathlib.Path(settings.RESEARCH_ROOT)
    errors = []

    if not research_root.exists():
        errors.append(f"RESEARCH_ROOT does not exist: '{research_root}'")
    else:
        for script in ("train_centralized.py", "train_fedavg.py", "train_fed_finetune.py"):
            if not (research_root / script).exists():
                errors.append(f"Research script missing: '{research_root / script}'")

    if torch is None:
        errors.append("torch not importable — venv is misconfigured.")
    elif not torch.cuda.is_available():
        errors.append(
            "CUDA GPU required but not available. "
            "Check NVIDIA drivers and cuda installation."
        )
    else:
        logger.info(
            "GPU detected: %s (CUDA %s)",
            torch.cuda.get_device_name(0),
            torch.version.cuda,
        )

    try:
        import mamba_ssm  # noqa: F401
        logger.info("mamba_ssm imported successfully.")
    except ImportError:
        logger.error(
            "mamba_ssm could not be imported — the model may use a linear mock silently. "
            "Run: pip install mamba-ssm --no-build-isolation"
        )

    fl_output = pathlib.Path(settings.FL_OUTPUT_DIR)
    if fl_output.exists() and not os.access(fl_output, os.W_OK):
        errors.append(f"FL_OUTPUT_DIR is not writable: '{fl_output}'")

    if errors:
        msg = "Environment validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        logger.critical(msg)
        raise RuntimeError(msg)

    logger.info("Environment validation passed.")
