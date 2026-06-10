import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, field_validator

from app.models.user import UserRole


class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: str
    role: UserRole
    hospital_id: uuid.UUID | None = None

    @field_validator("password")
    @classmethod
    def password_length(cls, v: str) -> str:
        if len(v) < 12:
            raise ValueError("Password must be at least 12 characters")
        if len(v) > 128:
            raise ValueError("Password must not exceed 128 characters")
        return v


class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: str
    full_name: str
    role: UserRole
    hospital_id: uuid.UUID | None
    is_active: bool
    must_change_password: bool
    created_at: datetime


class UserUpdate(BaseModel):
    """Fields a user (or admin) can update. Role is intentionally excluded."""
    full_name: str | None = None
    is_active: bool | None = None
