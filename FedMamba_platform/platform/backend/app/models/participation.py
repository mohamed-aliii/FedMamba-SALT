import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, generate_uuid


class HospitalParticipation(Base):
    __tablename__ = "hospital_participations"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, default=generate_uuid
    )
    experiment_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("fl_experiments.id"), nullable=False
    )
    hospital_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("hospitals.id"), nullable=False
    )
    client_index: Mapped[int] = mapped_column(Integer, nullable=False)
    dataset_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    joined_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    experiment = relationship("FLExperiment", back_populates="participations")
    hospital = relationship("Hospital", back_populates="participations")
