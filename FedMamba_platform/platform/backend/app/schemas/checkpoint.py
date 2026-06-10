import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.models.checkpoint import CheckpointType


class ModelCheckpointCreate(BaseModel):
    experiment_id: uuid.UUID
    round_number: int | None = None
    checkpoint_type: CheckpointType
    file_path: str
    val_acc: float | None = None
    val_auc: float | None = None


class ModelCheckpointRead(BaseModel):
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
