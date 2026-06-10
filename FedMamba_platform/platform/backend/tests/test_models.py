import uuid

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session, sessionmaker

from app.models import (
    Base,
    User,
    UserRole,
    Hospital,
    FLExperiment,
    ExperimentStatus,
    ExperimentPhase,
    FLAlgorithm,
    FLRound,
    RoundStatus,
    HospitalParticipation,
    ModelCheckpoint,
    CheckpointType,
    PatientCase,
    DiagnosticResult,
    AuditLog,
)


@pytest.fixture
def db():
    """Create a fresh in-memory SQLite database for each test."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    TestSession = sessionmaker(bind=engine)
    session = TestSession()
    yield session
    session.close()
    engine.dispose()


EXPECTED_TABLES = [
    "users",
    "hospitals",
    "fl_experiments",
    "fl_rounds",
    "hospital_participations",
    "model_checkpoints",
    "patient_cases",
    "diagnostic_results",
    "audit_logs",
]


def test_all_nine_tables_created(db: Session):
    """Verify all 9 tables are created in a fresh database."""
    inspector = inspect(db.bind)
    tables = inspector.get_table_names()
    for expected in EXPECTED_TABLES:
        assert expected in tables, f"Table '{expected}' not found. Got: {tables}"


def test_user_and_hospital_crud(db: Session):
    """Insert a User + Hospital and verify retrieval."""
    hospital = Hospital(name="Hospital Alpha", contact_email="admin@alpha.org")
    db.add(hospital)
    db.flush()

    user = User(
        email="doctor@alpha.org",
        hashed_password="fakehash123",
        full_name="Dr. Ahmed",
        role=UserRole.DOCTOR,
        hospital_id=hospital.id,
    )
    db.add(user)
    db.commit()

    # Retrieve
    fetched_user = db.query(User).filter_by(email="doctor@alpha.org").first()
    assert fetched_user is not None
    assert fetched_user.full_name == "Dr. Ahmed"
    assert fetched_user.role == UserRole.DOCTOR
    assert fetched_user.hospital_id == hospital.id
    assert isinstance(fetched_user.id, uuid.UUID)

    fetched_hospital = db.query(Hospital).filter_by(name="Hospital Alpha").first()
    assert fetched_hospital is not None
    assert fetched_hospital.is_active is True


def test_experiment_with_rounds_relationship(db: Session):
    """Insert an FLExperiment with FLRound and verify the relationship."""
    # Create a user first (experiment needs created_by)
    user = User(
        email="admin@fedmamba.local",
        hashed_password="fakehash",
        full_name="Admin User",
        role=UserRole.ADMIN,
    )
    db.add(user)
    db.flush()

    experiment = FLExperiment(
        name="Retina Pretrain v1",
        phase=ExperimentPhase.PRETRAIN,
        algorithm=FLAlgorithm.FEDAVG,
        output_dir="/output/exp1",
        created_by_id=user.id,
    )
    db.add(experiment)
    db.flush()

    assert experiment.status == ExperimentStatus.PENDING

    round1 = FLRound(
        experiment_id=experiment.id,
        round_number=1,
        status=RoundStatus.COMPLETED,
        avg_loss=0.45,
        val_acc=0.72,
    )
    round2 = FLRound(
        experiment_id=experiment.id,
        round_number=2,
        status=RoundStatus.RUNNING,
        avg_loss=0.38,
    )
    db.add_all([round1, round2])
    db.commit()

    # Verify relationship
    db.refresh(experiment)
    assert len(experiment.rounds) == 2
    assert experiment.rounds[0].round_number == 1
    assert experiment.rounds[1].round_number == 2
    assert experiment.rounds[0].avg_loss == 0.45


def test_uuid_primary_keys_are_auto_generated(db: Session):
    """Verify UUID primary keys are server-generated, never client-supplied."""
    hospital = Hospital(name="Auto UUID Hospital")
    db.add(hospital)
    db.commit()

    assert hospital.id is not None
    assert isinstance(hospital.id, uuid.UUID)

    user = User(
        email="uuid@test.com",
        hashed_password="hash",
        full_name="UUID Test",
        role=UserRole.ADMIN,
    )
    db.add(user)
    db.commit()

    assert user.id is not None
    assert isinstance(user.id, uuid.UUID)
    assert user.id != hospital.id  # unique UUIDs


def test_audit_log_append(db: Session):
    """Verify AuditLog can be created (append-only by design)."""
    log = AuditLog(
        action="LOGIN",
        resource_type="user",
        resource_id="some-user-id",
        ip_address="192.168.1.1",
        details_json='{"method": "password"}',
    )
    db.add(log)
    db.commit()

    fetched = db.query(AuditLog).first()
    assert fetched is not None
    assert fetched.action == "LOGIN"
    assert fetched.user_id is None  # no user linked
    assert isinstance(fetched.id, uuid.UUID)


def test_hospital_participation_and_checkpoint(db: Session):
    """Verify HospitalParticipation and ModelCheckpoint can be created."""
    hospital = Hospital(name="Hospital Beta")
    user = User(
        email="mgr@beta.org",
        hashed_password="hash",
        full_name="Manager",
        role=UserRole.HOSPITAL_MANAGER,
    )
    db.add_all([hospital, user])
    db.flush()

    user.hospital_id = hospital.id

    experiment = FLExperiment(
        name="Finetune v1",
        phase=ExperimentPhase.FINETUNE,
        algorithm=FLAlgorithm.FEDPROX,
        output_dir="/output/ft1",
        created_by_id=user.id,
    )
    db.add(experiment)
    db.flush()

    participation = HospitalParticipation(
        experiment_id=experiment.id,
        hospital_id=hospital.id,
        client_index=0,
        dataset_size=12400,
    )
    checkpoint = ModelCheckpoint(
        experiment_id=experiment.id,
        round_number=10,
        checkpoint_type=CheckpointType.BEST,
        file_path="checkpoints/best_round10.pth",
        val_acc=0.842,
    )
    db.add_all([participation, checkpoint])
    db.commit()

    assert participation.is_active is True
    assert checkpoint.is_deployed is False
    assert checkpoint.val_acc == 0.842


def test_patient_case_and_diagnostic_result(db: Session):
    """Verify PatientCase and DiagnosticResult can be created."""
    hospital = Hospital(name="Hospital Gamma")
    user = User(
        email="doc@gamma.org",
        hashed_password="hash",
        full_name="Dr. Sara",
        role=UserRole.DOCTOR,
    )
    db.add_all([hospital, user])
    db.flush()

    user.hospital_id = hospital.id

    experiment = FLExperiment(
        name="Exp for checkpoint",
        phase=ExperimentPhase.FINETUNE,
        algorithm=FLAlgorithm.FEDAVG,
        output_dir="/output/expg",
        created_by_id=user.id,
    )
    db.add(experiment)
    db.flush()

    checkpoint = ModelCheckpoint(
        experiment_id=experiment.id,
        checkpoint_type=CheckpointType.LATEST,
        file_path="checkpoints/latest.pth",
    )
    db.add(checkpoint)
    db.flush()

    case = PatientCase(
        case_uid="RET-2026-0042",
        hospital_id=hospital.id,
        uploaded_by_id=user.id,
        image_path="images/patient_0042.npy",
    )
    db.add(case)
    db.flush()

    result = DiagnosticResult(
        case_id=case.id,
        checkpoint_id=checkpoint.id,
        prediction_class=1,
        confidence=0.873,
        probabilities_json='[0.127, 0.873]',
        processing_time_ms=89,
    )
    db.add(result)
    db.commit()

    assert case.case_uid == "RET-2026-0042"
    assert result.confidence == 0.873
    assert result.processing_time_ms == 89

    # Verify relationship
    db.refresh(case)
    assert len(case.diagnostic_results) == 1
    assert case.diagnostic_results[0].prediction_class == 1
