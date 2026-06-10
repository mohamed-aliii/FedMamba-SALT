"""
FL Config Generator — produces YAML config files for research scripts.

CRITICAL CONSTRAINT: This module NEVER imports from the research codebase.
All generated configs are passed as files to subprocesses.

GPU RULE: device is always "cuda". If config_yaml contains "device: cpu",
log a WARNING and override. CPU fallback silently corrupts GPU-tuned
hyperparams (batch_size=128, pin_memory=True are meaningless on CPU).
"""
import logging
import os
from pathlib import Path

import yaml

from app.models.experiment import FLAlgorithm, FLExperiment, ExperimentPhase

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GPU Defaults — all tuned for CUDA training
# ---------------------------------------------------------------------------
GPU_DEFAULTS: dict = {
    "device": "cuda",
    "num_workers": 4,      # parallel data loading (GPU needs fed fast)
    "pin_memory": True,    # faster host→device transfer
    "batch_size": 128,     # standard GPU batch
}

# ---------------------------------------------------------------------------
# Script map — (phase, algorithm) → script filename relative to RESEARCH_ROOT
# ---------------------------------------------------------------------------
SCRIPT_MAP: dict[tuple[str, str], str] = {
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
    """Generates YAML config files for research scripts.

    The platform provides a set of KNOWN parameters it can configure.
    Any additional parameters can be passed through the experiment's
    config_yaml field, giving the research team full control over
    parameters the platform doesn't know about.
    """

    def __init__(self, research_root: str) -> None:
        self.research_root = Path(research_root).resolve()

    def get_script_path(self, phase: str, algorithm: str) -> Path:
        """Resolve the absolute path to a research script.

        Raises:
            FileNotFoundError: With helpful message including the attempted
                path and the configured RESEARCH_ROOT.
        """
        key = (phase.upper(), algorithm.upper())
        script_rel = SCRIPT_MAP.get(key)
        if script_rel is None:
            raise ValueError(
                f"No script defined for phase={phase}, algorithm={algorithm}. "
                f"Valid combinations: {list(SCRIPT_MAP.keys())}"
            )
        script_path = self.research_root / script_rel
        if not script_path.exists():
            raise FileNotFoundError(
                f"Research script not found: '{script_path}'. "
                f"Configured RESEARCH_ROOT='{self.research_root}'. "
                f"Ensure RESEARCH_ROOT points to the directory containing "
                f"train_centralized.py and train_fedavg.py."
            )
        return script_path

    def generate_output_dir(self, experiment_id: str) -> Path:
        """Create and return {RESEARCH_ROOT}/outputs/platform/{experiment_id}/.

        Raises:
            OSError: If the directory cannot be created (permissions, etc.).
        """
        output_dir = self.research_root / "outputs" / "platform" / experiment_id
        output_dir.mkdir(parents=True, exist_ok=True)
        if not os.access(output_dir, os.W_OK):
            raise OSError(
                f"Output directory is not writable: '{output_dir}'. "
                f"Check filesystem permissions."
            )
        return output_dir

    def generate_config(
        self,
        experiment: FLExperiment,
        output_dir: Path,
        data_path: str,
        extra_paths: dict | None = None,
    ) -> Path:
        """Write config.yaml to output_dir and return its path.

        Merge order (later wins, except device which is always cuda):
          1. GPU_DEFAULTS
          2. _experiment_to_params(experiment)
          3. extra_paths
          4. experiment.config_yaml  (research team — always wins, except device)

        Device rule: always "cuda". If experiment.config_yaml contains
        "device: cpu", log WARNING and override back to "cuda".
        """
        # Start with GPU defaults
        config: dict = dict(GPU_DEFAULTS)

        # Layer experiment params
        config.update(self._experiment_to_params(experiment))

        # Layer standard paths
        config["output_dir"] = str(output_dir)
        config["data_path"] = data_path

        # Layer extra paths (teacher_ckpt, encoder_ckpt, etc.)
        if extra_paths:
            config.update(extra_paths)

        # Layer research team overrides from config_yaml
        if experiment.config_yaml:
            try:
                overrides = yaml.safe_load(experiment.config_yaml) or {}
            except yaml.YAMLError as e:
                logger.warning(
                    "experiment %s config_yaml is invalid YAML (%s); skipping overrides",
                    experiment.id, e,
                )
                overrides = {}

            # Device lock: research team cannot override to CPU
            if overrides.get("device", "cuda").lower() == "cpu":
                logger.warning(
                    "experiment %s config_yaml contains 'device: cpu'. "
                    "Overriding to 'device: cuda'. GPU training assumptions "
                    "(batch_size=%s, pin_memory=True) are invalid on CPU. "
                    "Reconfigure explicitly if CPU is intended.",
                    experiment.id, config.get("batch_size"),
                )
                overrides["device"] = "cuda"

            config.update(overrides)

        # Enforce device=cuda regardless (final guarantee)
        config["device"] = "cuda"

        # Write config.yaml
        config_path = output_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        logger.info(
            "Config written to %s for experiment %s", config_path, experiment.id
        )
        return config_path

    def _experiment_to_params(self, experiment: FLExperiment) -> dict:
        """Map ORM experiment fields to YAML config keys.

        Known mappings:
          - algorithm SCAFFOLD  → algo: scaffold
          - algorithm FEDPROX   → mu: <from config if present>
          - phase FINETUNE      → mode: federated_finetune
          - algorithm CENTRALIZED + phase PRETRAIN → mode: centralized
        Unknown/extra fields stay in experiment.config_yaml.
        """
        params: dict = {}

        phase = experiment.phase.value if experiment.phase else ""
        algo = experiment.algorithm.value if experiment.algorithm else ""

        # Mode mapping
        if phase == ExperimentPhase.FINETUNE.value:
            params["mode"] = "federated_finetune"
        elif algo == FLAlgorithm.CENTRALIZED.value:
            params["mode"] = "centralized"

        # Algorithm-specific
        if algo == FLAlgorithm.SCAFFOLD.value:
            params["algo"] = "scaffold"
        elif algo == FLAlgorithm.FEDPROX.value:
            params["algo"] = "fedprox"
        elif algo == FLAlgorithm.FEDAVG.value:
            params["algo"] = "fedavg"

        params["experiment_id"] = str(experiment.id)

        return params
