"""
Output directory auto-discovery for the FL research codebase.

Scans RESEARCH_ROOT for directories that contain known marker files
(checkpoints, metrics CSVs). Used for:
  1. Recovering experiments lost due to platform restart
  2. Importing manually-run experiments not created via the platform
  3. Finding output dirs for running experiments
"""
import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Marker files that identify a directory as an FL output directory
# ---------------------------------------------------------------------------
MARKER_FILES: dict[str, str] = {
    "ckpt_latest.pth":          "has_checkpoint",
    "ckpt_best.pth":            "has_best_checkpoint",
    "ckpt_best_finetune.pth":   "has_best_finetune_checkpoint",
    "training_metrics.csv":     "centralized_metrics",
    "federated_metrics.csv":    "fedavg_metrics",
    "fed_finetune_metrics.csv": "finetune_metrics",
    "config.yaml":              "platform_managed",
}

# Expected CSV filename for each research script
CSV_FOR_SCRIPT: dict[str, str] = {
    "train_centralized.py":  "training_metrics.csv",
    "train_fedavg.py":       "federated_metrics.csv",
    "train_fed_finetune.py": "fed_finetune_metrics.csv",
}

# Directories to always exclude from scanning
_EXCLUDED_DIRS: set[str] = {
    "platform", "__pycache__", ".git", "node_modules", ".venv",
    "venv", ".tox", "dist", "build", ".eggs",
}


class OutputDirectoryScanner:
    """Walks RESEARCH_ROOT and identifies FL output directories."""

    def __init__(self, research_root: str, max_depth: int = 4) -> None:
        self.research_root = Path(research_root).resolve()
        self.max_depth = max_depth

    def scan(self) -> list[dict]:
        """Walk RESEARCH_ROOT up to max_depth and return FL output directories.

        Returns dirs that contain at least one MARKER_FILE.
        Excludes: platform/, __pycache__, .git, node_modules, .venv

        Each result dict:
        {
          "path": str,                  # absolute path
          "markers": list[str],         # semantic labels found
          "is_platform_managed": bool,  # config.yaml present
          "metrics_csv": str | None,    # absolute path to metrics CSV
          "checkpoints": list[str],     # absolute paths to .pth files
        }
        """
        results = []
        self._walk(self.research_root, 0, results)
        return results

    def _walk(self, current: Path, depth: int, results: list) -> None:
        if depth > self.max_depth:
            return
        try:
            entries = list(current.iterdir())
        except PermissionError:
            return

        # Check if this directory is a marker directory
        found_markers = []
        found_checkpoints = []
        metrics_csv = None

        for entry in entries:
            name = entry.name
            if name in MARKER_FILES:
                found_markers.append(MARKER_FILES[name])
                if name.endswith("_metrics.csv"):
                    metrics_csv = str(entry)
            if name.startswith("ckpt_") and name.endswith(".pth"):
                found_checkpoints.append(str(entry))

        if found_markers:
            results.append({
                "path": str(current),
                "markers": found_markers,
                "is_platform_managed": "platform_managed" in found_markers,
                "metrics_csv": metrics_csv,
                "checkpoints": found_checkpoints,
            })

        # Recurse into subdirectories that are not excluded
        for entry in entries:
            if entry.is_dir() and entry.name not in _EXCLUDED_DIRS:
                self._walk(entry, depth + 1, results)

    def classify_script(self, directory_info: dict) -> str | None:
        """Determine which research script produced this output directory.

        Uses marker presence first, then reads config.yaml as fallback.
        """
        markers = directory_info.get("markers", [])

        if "finetune_metrics" in markers:
            return "train_fed_finetune.py"
        if "fedavg_metrics" in markers:
            return "train_fedavg.py"
        if "centralized_metrics" in markers:
            return "train_centralized.py"

        # Fallback: read config.yaml for algo/mode field
        config_path = Path(directory_info["path"]) / "config.yaml"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    cfg = yaml.safe_load(f) or {}
                mode = cfg.get("mode", "")
                algo = cfg.get("algo", "")
                if mode == "federated_finetune":
                    return "train_fed_finetune.py"
                if algo in ("fedavg", "fedprox", "scaffold"):
                    return "train_fedavg.py"
                if mode == "centralized":
                    return "train_centralized.py"
            except Exception as e:
                logger.debug("Could not read config.yaml at %s: %s", config_path, e)

        return None

    def find_experiment_output(self, experiment_id: str) -> dict | None:
        """Find the output directory for a platform-managed experiment.

        Matches by experiment_id appearing in the directory path OR
        inside config.yaml as the 'experiment_id' key.
        """
        for info in self.scan():
            # Check path component
            if experiment_id in info["path"]:
                return info

            # Check config.yaml
            config_path = Path(info["path"]) / "config.yaml"
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        cfg = yaml.safe_load(f) or {}
                    if cfg.get("experiment_id") == experiment_id:
                        return info
                except Exception:
                    pass

        return None
