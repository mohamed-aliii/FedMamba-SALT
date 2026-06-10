import enum
import uuid

from sqlalchemy import Boolean, Enum, ForeignKey, String, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, generate_uuid


class UserRole(str, enum.Enum):
    ADMIN = "ADMIN"
    HOSPITAL_MANAGER = "HOSPITAL_MANAGER"
    DOCTOR = "DOCTOR"


class User(TimestampMixin, Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, default=generate_uuid
    )
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(128), nullable=False)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[UserRole] = mapped_column(Enum(UserRole), nullable=False)
    hospital_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("hospitals.id"), nullable=True
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    must_change_password: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    hospital = relationship("Hospital", back_populates="users")
    experiments = relationship("FLExperiment", back_populates="created_by")
    uploaded_cases = relationship("PatientCase", back_populates="uploaded_by")
    audit_logs = relationship("AuditLog", back_populates="user")
