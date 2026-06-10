import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.models.experiment import ExperimentPhase, ExperimentStatus, FLAlgorithm
from app.models.round import RoundStatus
from app.models.checkpoint import CheckpointType


# ---------------------------------------------------------------------------
# Experiment schemas
# ---------------------------------------------------------------------------

class FLExperimentCreate(BaseModel):
    name: str
    description: str | None = None
    phase: ExperimentPhase
    algorithm: FLAlgorithm
    config_yaml: str | None = None
    # output_dir removed — set at launch time by the Celery task


class FLExperimentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    description: str | None
    status: ExperimentStatus
    phase: ExperimentPhase
    algorithm: FLAlgorithm
    config_yaml: str | None
    output_dir: str | None
    pid: int | None
    error_message: str | None
    created_by_id: uuid.UUID
    created_at: datetime
    updated_at: datetime | None
    started_at: datetime | None
    completed_at: datetime | None


class FLExperimentUpdate(BaseModel):
    """Only allowed when status=PENDING."""
    name: str | None = None
    description: str | None = None
    config_yaml: str | None = None


# ---------------------------------------------------------------------------
# Round schemas
# ---------------------------------------------------------------------------

class FLRoundRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    experiment_id: uuid.UUID
    round_number: int
    status: RoundStatus
    avg_loss: float | None
    val_acc: float | None
    val_auc: float | None
    metrics_json: str | None
    created_at: datetime
    completed_at: datetime | None


# ---------------------------------------------------------------------------
# Checkpoint schemas
# ---------------------------------------------------------------------------

class CheckpointRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    experiment_id: uuid.UUID
    round_number: int | None
    checkpoint_type: CheckpointType
    file_path: str
    val_acc: float | None
    val_auc: float | None
    is_deployed: bool
    created_at: datetime


# ---------------------------------------------------------------------------
# Participation schemas (kept from Phase 1)
# ---------------------------------------------------------------------------

class HospitalParticipationCreate(BaseModel):
    experiment_id: uuid.UUID
    hospital_id: uuid.UUID
    client_index: int
    dataset_size: int | None = None


class HospitalParticipationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    experiment_id: uuid.UUID
    hospital_id: uuid.UUID
    client_index: int
    dataset_size: int | None
    is_active: bool
    joined_at: datetime
