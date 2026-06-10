import enum
import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Enum, Float, ForeignKey, Integer, String, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, generate_uuid


class CheckpointType(str, enum.Enum):
    LATEST = "LATEST"
    BEST = "BEST"
    PERIODIC = "PERIODIC"


class ModelCheckpoint(Base):
    __tablename__ = "model_checkpoints"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, default=generate_uuid
    )
    experiment_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("fl_experiments.id"), nullable=False
    )
    round_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    checkpoint_type: Mapped[CheckpointType] = mapped_column(
        Enum(CheckpointType), nullable=False
    )
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    val_acc: Mapped[float | None] = mapped_column(Float, nullable=True)
    val_auc: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_deployed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    experiment = relationship("FLExperiment", back_populates="checkpoints")
    diagnostic_results = relationship("DiagnosticResult", back_populates="checkpoint")
