"""
FL Script Runner — manages subprocess execution of research scripts.

SECURITY: build_command() NEVER uses shell=True. NEVER string interpolation.
Commands are always list[str]. Paths are validated by SafePathValidator first.

GPU NOTE: start_new_session=True is critical — it kills the entire process
group including DataLoader worker processes when we SIGTERM.
"""
import logging
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

import redis as redis_lib

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GPU-tuned constants
# ---------------------------------------------------------------------------
POLLING_INTERVAL_SECONDS   = 10    # GPU rounds complete in seconds/minutes
PROCESS_TIMEOUT_HOURS      = 24    # GPU runs rarely exceed 24h
STALLED_THRESHOLD_MINUTES  = 20    # no output for 20 min on GPU = problem
LOG_BUFFER_LINES           = 500   # max stdout lines kept in Redis

_PID_KEY    = "experiment:{id}:pid"
_LOGS_KEY   = "experiment:{id}:logs"
_PUBSUB_CH  = "experiment:{id}"


def _pid_key(experiment_id: str) -> str:
    return f"experiment:{experiment_id}:pid"


def _logs_key(experiment_id: str) -> str:
    return f"experiment:{experiment_id}:logs"


def _pubsub_channel(experiment_id: str) -> str:
    return f"experiment:{experiment_id}"


class FLScriptRunner:
    """Manages subprocess execution of FL research scripts."""

    def __init__(self, research_root: str, redis_client: redis_lib.Redis) -> None:
        self.research_root = Path(research_root).resolve()
        self.python_executable = sys.executable
        # Same Python as platform → same venv → torch, mamba_ssm available
        self.redis = redis_client

    def build_command(self, script_path: Path, config_path: Path) -> list[str]:
        """Build the subprocess command list.

        Returns [sys.executable, str(script_path), "--config", str(config_path)].
        NEVER shell=True. NEVER string interpolation. List only.
        Both paths must be validated by SafePathValidator before calling this.
        """
        return [
            self.python_executable,
            str(script_path),
            "--config",
            str(config_path),
        ]

    def launch(
        self,
        command: list[str],
        experiment_id: str,
        output_dir: Path,
    ) -> subprocess.Popen:
        """Launch a research script as a subprocess.

        Configuration:
          stdout=PIPE, stderr=STDOUT  — merged stream
          cwd=research_root           — CRITICAL: scripts use relative paths
          env: inherit + PYTHONPATH=research_root
          start_new_session=True      — kill entire process group incl. workers
          bufsize=1, text=True        — line-buffered

        After launch:
          - PID stored in Redis experiment:{id}:pid with 25h TTL
          - Background thread reads stdout line-by-line → pushes to Redis
            list experiment:{id}:logs (trimmed to LOG_BUFFER_LINES)
        """
        env = os.environ.copy()
        # Ensure the research codebase is importable by the subprocess
        existing_pythonpath = env.get("PYTHONPATH", "")
        if existing_pythonpath:
            env["PYTHONPATH"] = f"{self.research_root}{os.pathsep}{existing_pythonpath}"
        else:
            env["PYTHONPATH"] = str(self.research_root)

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(self.research_root),
            env=env,
            start_new_session=True,  # own process group → SIGTERM kills workers
            bufsize=1,
            text=True,
        )

        # Store PID in Redis (TTL: process timeout + 1h buffer)
        pid_ttl = (PROCESS_TIMEOUT_HOURS + 1) * 3600
        self.redis.set(_pid_key(experiment_id), str(process.pid), ex=pid_ttl)

        # Background thread: stream stdout → Redis log buffer
        thread = threading.Thread(
            target=self._stream_logs,
            args=(process, experiment_id),
            daemon=True,
            name=f"log-stream-{experiment_id[:8]}",
        )
        thread.start()

        logger.info(
            "Launched experiment %s: PID=%s command=%s",
            experiment_id, process.pid, command,
        )
        return process

    def _stream_logs(self, process: subprocess.Popen, experiment_id: str) -> None:
        """Background thread: read stdout line-by-line → Redis list."""
        logs_key = _logs_key(experiment_id)
        try:
            for line in process.stdout:  # type: ignore[union-attr]
                stripped = line.rstrip("\n")
                pipe = self.redis.pipeline()
                pipe.rpush(logs_key, stripped)
                pipe.ltrim(logs_key, -LOG_BUFFER_LINES, -1)
                pipe.execute()
        except Exception as e:
            logger.warning("Log stream for %s terminated: %s", experiment_id, e)

    def terminate(self, experiment_id: str) -> None:
        """Terminate a running experiment and its entire process group.

        Steps:
          1. Get PID from Redis
          2. os.killpg(SIGTERM) → kills DataLoader workers too
          3. Wait up to 10s for clean shutdown
          4. If still alive: os.killpg(SIGKILL)
          5. Clean Redis keys for this experiment
        """
        pid_str = self.redis.get(_pid_key(experiment_id))
        if not pid_str:
            logger.warning("No PID found in Redis for experiment %s", experiment_id)
            return

        pid = int(pid_str)

        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGTERM)
            logger.info("Sent SIGTERM to process group %s (PID %s)", pgid, pid)
        except OSError as e:
            logger.info("Process %s already gone: %s", pid, e)
            self._cleanup_redis(experiment_id)
            return

        # Wait up to 10s for clean shutdown
        import time
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if not self.is_alive(experiment_id):
                break
            time.sleep(0.5)
        else:
            # Force kill
            try:
                pgid = os.getpgid(pid)
                os.killpg(pgid, signal.SIGKILL)
                logger.warning("Sent SIGKILL to process group %s", pgid)
            except OSError:
                pass

        self._cleanup_redis(experiment_id)

    def _cleanup_redis(self, experiment_id: str) -> None:
        self.redis.delete(_pid_key(experiment_id))
        # Keep logs for post-mortem — don't delete them here

    def is_alive(self, experiment_id: str) -> bool:
        """Check if the subprocess is still running.

        Uses os.kill(pid, 0) — sends no signal, just checks existence.
        Returns False if PID missing from Redis or process is gone.
        """
        pid_str = self.redis.get(_pid_key(experiment_id))
        if not pid_str:
            return False
        pid = int(pid_str)
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def get_recent_logs(self, experiment_id: str, n: int = LOG_BUFFER_LINES) -> list[str]:
        """Return the most recent n log lines from Redis."""
        return self.redis.lrange(_logs_key(experiment_id), -n, -1)

    def publish_event(self, experiment_id: str, event: dict) -> None:
        """Publish an SSE event to the Redis pub/sub channel."""
        import json
        self.redis.publish(_pubsub_channel(experiment_id), json.dumps(event))
