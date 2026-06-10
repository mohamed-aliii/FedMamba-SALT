import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class PatientCaseCreate(BaseModel):
    case_uid: str
    hospital_id: uuid.UUID
    image_path: str
    metadata_json: str | None = None


class PatientCaseRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    case_uid: str
    hospital_id: uuid.UUID
    uploaded_by_id: uuid.UUID
    image_path: str
    metadata_json: str | None
    created_at: datetime


class DiagnosticResultCreate(BaseModel):
    case_id: uuid.UUID
    checkpoint_id: uuid.UUID
    prediction_class: int
    confidence: float
    probabilities_json: str | None = None
    gradcam_path: str | None = None
    processing_time_ms: int


class DiagnosticResultRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    case_id: uuid.UUID
    checkpoint_id: uuid.UUID
    prediction_class: int
    confidence: float
    probabilities_json: str | None
    gradcam_path: str | None
    processing_time_ms: int
    created_at: datetime
