import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class AuditLogRead(BaseModel):
    """Read-only schema for audit logs.

    There is intentionally no AuditLogCreate schema — audit logs are created
    programmatically by the application, never by API clients directly.
    """

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    user_id: uuid.UUID | None
    action: str
    resource_type: str
    resource_id: str | None
    ip_address: str | None
    details_json: str | None
    created_at: datetime
