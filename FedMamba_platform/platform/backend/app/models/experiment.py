import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, String, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, generate_uuid


class ExperimentStatus(str, enum.Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PAUSED = "PAUSED"
    IMPORTED = "IMPORTED"  # manually-run experiments discovered by scanner


class ExperimentPhase(str, enum.Enum):
    PRETRAIN = "PRETRAIN"
    FINETUNE = "FINETUNE"


class FLAlgorithm(str, enum.Enum):
    FEDAVG = "FEDAVG"
    FEDPROX = "FEDPROX"
    SCAFFOLD = "SCAFFOLD"
    CENTRALIZED = "CENTRALIZED"


class FLExperiment(TimestampMixin, Base):
    __tablename__ = "fl_experiments"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, default=generate_uuid
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[ExperimentStatus] = mapped_column(
        Enum(ExperimentStatus), default=ExperimentStatus.PENDING, nullable=False
    )
    phase: Mapped[ExperimentPhase] = mapped_column(Enum(ExperimentPhase), nullable=False)
    algorithm: Mapped[FLAlgorithm] = mapped_column(Enum(FLAlgorithm), nullable=False)
    # Opaque YAML config passed verbatim to research scripts.
    # The platform must not parse or validate its contents.
    config_yaml: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Set at launch time by the Celery task, not at experiment creation.
    output_dir: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    # Subprocess PID stored in both DB and Redis for fast access.
    pid: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Last 50 stdout lines stored on failure for diagnosis.
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_by_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("users.id"), nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    created_by = relationship("User", back_populates="experiments")
    rounds = relationship("FLRound", back_populates="experiment", order_by="FLRound.round_number")
    participations = relationship("HospitalParticipation", back_populates="experiment")
    checkpoints = relationship("ModelCheckpoint", back_populates="experiment")
