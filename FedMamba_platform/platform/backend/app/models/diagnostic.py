import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, generate_uuid


class DiagnosticResult(Base):
    __tablename__ = "diagnostic_results"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, default=generate_uuid
    )
    case_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("patient_cases.id"), nullable=False
    )
    checkpoint_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("model_checkpoints.id"), nullable=False
    )
    prediction_class: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    probabilities_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    gradcam_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    processing_time_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    case = relationship("PatientCase", back_populates="diagnostic_results")
    checkpoint = relationship("ModelCheckpoint", back_populates="diagnostic_results")
