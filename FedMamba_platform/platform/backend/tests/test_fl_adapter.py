"""
Phase 3 — FL Adapter Layer tests (10 tests).

No external services required. Uses tmp_path fixture for filesystem tests.
Mocks torch.cuda where needed — no GPU required to run these tests.
"""
import csv
import logging
import os
import sys
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.fl_config import FLConfigGenerator, GPU_DEFAULTS, SCRIPT_MAP
from app.core.fl_monitor import FLOutputMonitor
from app.core.fl_runner import FLScriptRunner
from app.core.output_discovery import OutputDirectoryScanner
from app.models.base import Base
from app.models.experiment import (
    ExperimentPhase,
    ExperimentStatus,
    FLAlgorithm,
    FLExperiment,
)

import app.models  # noqa: F401 — register all models


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def research_root(tmp_path: Path) -> Path:
    """A temp dir pretending to be RESEARCH_ROOT with real script stubs."""
    for script in ("train_centralized.py", "train_fedavg.py", "train_fed_finetune.py"):
        (tmp_path / script).write_text("# stub\n")
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "linear_probe.py").write_text("# stub\n")
    return tmp_path


@pytest.fixture
def config_gen(research_root: Path) -> FLConfigGenerator:
    return FLConfigGenerator(str(research_root))


@pytest.fixture
def monitor(research_root: Path) -> FLOutputMonitor:
    return FLOutputMonitor(str(research_root))


@pytest.fixture
def db_session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)


def _make_experiment(db, phase="PRETRAIN", algorithm="CENTRALIZED", config_yaml=None):
    from app.models.user import User, UserRole
    user = User(
        email=f"u{uuid.uuid4().hex[:6]}@test.com",
        hashed_password="x",
        full_name="Test",
        role=UserRole.ADMIN,
    )
    db.add(user)
    db.flush()
    exp = FLExperiment(
        name="test",
        phase=ExperimentPhase[phase],
        algorithm=FLAlgorithm[algorithm],
        config_yaml=config_yaml,
        status=ExperimentStatus.PENDING,
        created_by_id=user.id,
    )
    db.add(exp)
    db.commit()
    db.refresh(exp)
    return exp


# ---------------------------------------------------------------------------
# Test 1: device is always cuda in generated config
# ---------------------------------------------------------------------------

def test_device_always_cuda(config_gen, db_session, tmp_path):
    output_dir = tmp_path / "out1"
    output_dir.mkdir()
    exp = _make_experiment(db_session)

    config_path = config_gen.generate_config(
        experiment=exp,
        output_dir=output_dir,
        data_path="/data",
    )
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    assert cfg["device"] == "cuda", f"Expected device=cuda, got {cfg['device']}"


# ---------------------------------------------------------------------------
# Test 2: device: cpu in config_yaml is overridden to cuda with WARNING
# ---------------------------------------------------------------------------

def test_cpu_override_warns_and_uses_cuda(config_gen, db_session, tmp_path, caplog):
    output_dir = tmp_path / "out2"
    output_dir.mkdir()
    exp = _make_experiment(db_session, config_yaml="device: cpu\nlr: 0.001")

    with caplog.at_level(logging.WARNING, logger="app.core.fl_config"):
        config_path = config_gen.generate_config(
            experiment=exp,
            output_dir=output_dir,
            data_path="/data",
        )

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    assert cfg["device"] == "cuda"
    assert any("device" in rec.message and "cpu" in rec.message for rec in caplog.records), \
        "Expected a WARNING mentioning 'device' and 'cpu'"


# ---------------------------------------------------------------------------
# Test 3: Research team override wins for all fields EXCEPT device
# ---------------------------------------------------------------------------

def test_research_override_wins_except_device(config_gen, db_session, tmp_path):
    output_dir = tmp_path / "out3"
    output_dir.mkdir()
    # Research team sets batch_size=64 (GPU_DEFAULTS has 128)
    # and tries to set device=cpu (should be locked to cuda)
    exp = _make_experiment(
        db_session, config_yaml="lr: 0.001\nbatch_size: 64\ndevice: cpu"
    )

    config_path = config_gen.generate_config(
        experiment=exp,
        output_dir=output_dir,
        data_path="/data",
    )
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    assert cfg["batch_size"] == 64, "Research team batch_size override should win"
    assert cfg["device"] == "cuda", "device lock must always win over research override"


# ---------------------------------------------------------------------------
# Test 4: get_script_path raises FileNotFoundError with helpful message
# ---------------------------------------------------------------------------

def test_get_script_path_missing_script(tmp_path):
    """RESEARCH_ROOT exists but is empty (no scripts)."""
    gen = FLConfigGenerator(str(tmp_path))
    with pytest.raises(FileNotFoundError) as exc_info:
        gen.get_script_path("PRETRAIN", "CENTRALIZED")

    msg = str(exc_info.value)
    assert "train_centralized.py" in msg
    assert str(tmp_path) in msg
    assert "RESEARCH_ROOT" in msg


# ---------------------------------------------------------------------------
# Test 5: OutputDirectoryScanner finds and excludes correctly
# ---------------------------------------------------------------------------

def test_scanner_finds_and_excludes(tmp_path):
    # Should be FOUND
    exp1 = tmp_path / "outputs" / "exp1"
    exp1.mkdir(parents=True)
    (exp1 / "ckpt_latest.pth").write_bytes(b"")
    (exp1 / "training_metrics.csv").write_text("epoch,loss\n1,0.5\n")

    run1 = tmp_path / "eval_results" / "run1"
    run1.mkdir(parents=True)
    (run1 / "fed_finetune_metrics.csv").write_text("round,val_acc\n1,0.8\n")

    # Should be EXCLUDED
    platform_dir = tmp_path / "platform" / "something"
    platform_dir.mkdir(parents=True)
    (platform_dir / "ckpt_latest.pth").write_bytes(b"")

    cache_dir = tmp_path / "__pycache__"
    cache_dir.mkdir()
    (cache_dir / "cache.pyc").write_bytes(b"")

    scanner = OutputDirectoryScanner(str(tmp_path))
    results = scanner.scan()
    found_paths = {r["path"] for r in results}

    assert str(exp1) in found_paths, "exp1 should be found"
    assert str(run1) in found_paths, "run1 should be found"
    assert not any("platform" in p for p in found_paths), "platform/ must be excluded"
    assert not any("__pycache__" in p for p in found_paths), "__pycache__ must be excluded"
    assert len(results) == 2, f"Expected exactly 2 results, got {len(results)}: {found_paths}"


# ---------------------------------------------------------------------------
# Test 6: CSV column mapping for all 3 scripts
# ---------------------------------------------------------------------------

def _write_csv(path: Path, header: str, row: str):
    path.write_text(f"{header}\n{row}\n")


def test_csv_column_mapping_centralized(tmp_path, monitor):
    csv_path = tmp_path / "training_metrics.csv"
    header = ("epoch,loss,student_std,teacher_std,salt_norm_mode,"
               "salt_teacher_std_mean,salt_teacher_std_min,salt_teacher_std_max,"
               "salt_teacher_target_finite,salt_student_centered_finite,"
               "lr,epoch_time_s,gpu_mem_allocated_mb,gpu_mem_reserved_mb,gpu_mem_peak_mb")
    row = "5,0.0423,0.2341,0.3100,layer,0.25,0.20,0.30,1,1,5.00e-04,12.3,1024,2048,2048"
    _write_csv(csv_path, header, row)

    result = monitor.parse_latest_round(str(csv_path))
    assert result is not None
    assert result["round"] == 5.0
    assert result["loss"] == pytest.approx(0.0423)
    assert result["enc_std"] == pytest.approx(0.2341)
    assert result["lr"] == pytest.approx(5e-4)
    assert result["time_s"] == pytest.approx(12.3)
    assert result["gpu_mb"] == pytest.approx(1024.0)
    assert result["val_acc"] is None


def test_csv_column_mapping_fedavg(tmp_path, monitor):
    csv_path = tmp_path / "federated_metrics.csv"
    header = ("round,avg_loss,avg_enc_std,avg_teacher_std,salt_norm_mode,"
               "salt_teacher_std_mean,salt_teacher_std_min,salt_teacher_std_max,"
               "salt_teacher_target_finite,salt_student_centered_finite,"
               "student_update_norms,projector_update_norms,lr,round_time_s,gpu_mb,"
               "client_1_loss")
    row = "10,0.0512,0.1923,0.2800,layer,0.22,0.18,0.28,1,1,0.5,0.4,5.00e-04,45.2,2048,0.055"
    _write_csv(csv_path, header, row)

    result = monitor.parse_latest_round(str(csv_path))
    assert result is not None
    assert result["round"] == 10.0
    assert result["loss"] == pytest.approx(0.0512)
    assert result["enc_std"] == pytest.approx(0.1923)
    assert result["gpu_mb"] == pytest.approx(2048.0)
    assert result["val_acc"] is None


def test_csv_column_mapping_finetune(tmp_path, monitor):
    csv_path = tmp_path / "fed_finetune_metrics.csv"
    header = ("round,val_acc,val_loss_weighted,val_loss_unweighted,balanced_acc,auc,"
               "prediction_hist,per_class_recall,per_class_f1,per_class_support,"
               "feature_norm_mean,feature_std_mean,head_weight_norms,head_biases,"
               "encoder_update_norms,classifier_update_norms,"
               "enc_lr,cls_lr,round_time_s,gpu_mb,client_1_loss")
    row = ("10,78.23,0.3100,0.3200,76.00,0.8341,"
           "[],[]  ,[],[],1.2,0.3,[],[],[],[],1.0e-04,1.0e-03,23.1,1536,0.32")
    _write_csv(csv_path, header, row)

    result = monitor.parse_latest_round(str(csv_path))
    assert result is not None
    assert result["round"] == 10.0
    assert result["val_acc"] == pytest.approx(78.23)
    assert result["val_auc"] == pytest.approx(0.8341)
    assert result["lr"] == pytest.approx(1e-3)
    assert result["enc_std"] is None


# ---------------------------------------------------------------------------
# Test 7: Checkpoint classification
# ---------------------------------------------------------------------------

def test_checkpoint_classification(tmp_path, monitor):
    fixtures = [
        ("ckpt_latest.pth", "LATEST", None),
        ("ckpt_best.pth", "BEST", None),
        ("ckpt_best_finetune.pth", "BEST", None),
        ("ckpt_epoch_0010.pth", "PERIODIC", 10),
        ("ckpt_round_0050.pth", "PERIODIC", 50),
    ]
    for name, _, _ in fixtures:
        (tmp_path / name).write_bytes(b"\x00" * 100)

    results = monitor.find_checkpoints(str(tmp_path))
    result_map = {r["filename"]: r for r in results}

    for name, expected_type, expected_round in fixtures:
        assert name in result_map, f"{name} not found"
        assert result_map[name]["type"] == expected_type, \
            f"{name}: expected {expected_type}, got {result_map[name]['type']}"
        assert result_map[name]["round_number"] == expected_round, \
            f"{name}: expected round={expected_round}, got {result_map[name]['round_number']}"


# ---------------------------------------------------------------------------
# Test 8: FLScriptRunner.build_command structure
# ---------------------------------------------------------------------------

def test_build_command_structure(research_root):
    import fakeredis
    redis_client = fakeredis.FakeRedis(decode_responses=True)
    runner = FLScriptRunner(str(research_root), redis_client)

    script = research_root / "train_centralized.py"
    config = research_root / "config.yaml"
    config.write_text("device: cuda\n")

    cmd = runner.build_command(script, config)

    assert isinstance(cmd, list), "Command must be a list, not a string"
    assert cmd[0] == sys.executable, "First element must be sys.executable"
    assert "--config" in cmd, "--config flag must be present"
    assert str(config) in cmd, "config path must be in command"
    # Verify no shell=True is possible (list form prevents it)
    assert not isinstance(cmd, str), "Command must not be a string (shell injection risk)"


# ---------------------------------------------------------------------------
# Test 9: recover_orphaned_experiments marks dead processes as FAILED
# ---------------------------------------------------------------------------

def test_recover_orphaned_marks_dead_pid_as_failed(db_session):
    from app.tasks.fl_tasks import recover_orphaned_experiments
    from app.models.user import User, UserRole

    user = User(
        email="admin@test.com",
        hashed_password="x",
        full_name="Admin",
        role=UserRole.ADMIN,
    )
    db_session.add(user)
    db_session.flush()

    # PID 99999 is virtually guaranteed to not exist
    exp = FLExperiment(
        name="orphan",
        phase=ExperimentPhase.PRETRAIN,
        algorithm=FLAlgorithm.CENTRALIZED,
        status=ExperimentStatus.RUNNING,
        pid=99999,
        created_by_id=user.id,
    )
    db_session.add(exp)
    db_session.commit()
    exp_id = exp.id  # save before session might be touched

    # Wrap db_session so close() is a no-op — prevents exp from being detached
    class _NoClose:
        def __init__(self, sess):
            self._s = sess
        def __getattr__(self, n):
            return getattr(self._s, n)
        def close(self):
            pass  # prevent detach during test

    # Patch the environment validation so we don't need real research scripts
    with patch("app.tasks.fl_tasks._validate_environment"), \
         patch("app.tasks.fl_tasks.OutputDirectoryScanner.scan", return_value=[]), \
         patch("app.tasks.fl_tasks.SessionLocal", return_value=_NoClose(db_session)):
        recover_orphaned_experiments()

    # Re-query the experiment using the still-open db_session
    refreshed = db_session.query(FLExperiment).filter(
        FLExperiment.id == exp_id
    ).first()
    assert refreshed is not None
    assert refreshed.status == ExperimentStatus.FAILED
    assert refreshed.error_message is not None
    assert "99999" in refreshed.error_message or "terminated" in refreshed.error_message.lower()


# ---------------------------------------------------------------------------
# Test 10: CUDA pre-flight fails fast when CUDA unavailable
# ---------------------------------------------------------------------------

def test_cuda_preflight_fails_fast(db_session, research_root):
    from app.tasks.fl_tasks import _run_experiment_impl
    import app.tasks.fl_tasks as fl_tasks_module
    import fakeredis

    redis_client = fakeredis.FakeRedis(decode_responses=True)
    exp = _make_experiment(db_session)

    # Patch the module-level torch object to report CUDA unavailable
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False

    # Patch settings attributes used by _run_experiment_impl
    with patch.object(fl_tasks_module, "torch", mock_torch), \
         patch.object(fl_tasks_module.settings, "RESEARCH_ROOT", str(research_root)), \
         patch.object(fl_tasks_module.settings, "FL_OUTPUT_DIR", str(research_root)):
        _run_experiment_impl(str(exp.id), db_session, redis_client)

    # Re-query so we see the committed state
    refreshed = db_session.query(FLExperiment).filter(
        FLExperiment.id == exp.id
    ).first()
    assert refreshed.status == ExperimentStatus.FAILED
    assert refreshed.error_message is not None
    assert "CUDA" in refreshed.error_message
    # Confirm no subprocess was launched (pid should be None)
    assert refreshed.pid is None
