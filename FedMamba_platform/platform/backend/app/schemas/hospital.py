import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class HospitalCreate(BaseModel):
    name: str
    description: str | None = None
    contact_email: str | None = None


class HospitalRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    description: str | None
    is_active: bool
    contact_email: str | None
    created_at: datetime
    updated_at: datetime | None


class HospitalUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    is_active: bool | None = None
    contact_email: str | None = None
