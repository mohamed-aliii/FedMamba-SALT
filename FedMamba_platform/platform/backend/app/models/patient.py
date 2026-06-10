import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, Text, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, generate_uuid


class PatientCase(Base):
    __tablename__ = "patient_cases"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, default=generate_uuid
    )
    case_uid: Mapped[str] = mapped_column(
        String(128), unique=True, nullable=False,
        comment="Anonymized case ID provided by hospital — no PHI"
    )
    hospital_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("hospitals.id"), nullable=False
    )
    uploaded_by_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("users.id"), nullable=False
    )
    # Relative path inside FL_OUTPUT_DIR only.
    # Absolute paths or paths outside this directory must be rejected at the service layer.
    image_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    metadata_json: Mapped[str | None] = mapped_column(
        Text, nullable=True,
        comment="Non-PHI metadata only (e.g. image dimensions, modality)"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    hospital = relationship("Hospital", back_populates="patient_cases")
    uploaded_by = relationship("User", back_populates="uploaded_cases")
    diagnostic_results = relationship("DiagnosticResult", back_populates="case")
