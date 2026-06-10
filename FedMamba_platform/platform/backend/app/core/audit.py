"""
Audit logging helper. Every state-changing endpoint calls log_action().

IMPORTANT: AuditLog records are INSERT-ONLY. This module provides no delete
function and no delete endpoint should ever be created for audit logs.
"""
import json
import uuid

from fastapi import Request
from sqlalchemy.orm import Session

from app.models.audit import AuditLog


def log_action(
    db: Session,
    *,
    user_id: uuid.UUID | None,
    action: str,
    resource_type: str,
    resource_id: str | None = None,
    details: dict | None = None,
    request: Request | None = None,
) -> AuditLog:
    """Write an immutable audit log entry.

    Args:
        db: Database session.
        user_id: UUID of the acting user (None for anonymous/system actions).
        action: Verb describing the action (e.g. "LOGIN", "CREATE_EXPERIMENT").
        resource_type: Entity type affected (e.g. "User", "FLExperiment").
        resource_id: String representation of the affected resource ID.
        details: Optional dict with additional context (will be JSON-serialised).
        request: FastAPI Request object for extracting the client IP.
    """
    ip_address: str | None = None
    if request is not None:
        ip_address = request.client.host if request.client else None

    entry = AuditLog(
        user_id=user_id,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        ip_address=ip_address,
        details_json=json.dumps(details) if details else None,
    )
    db.add(entry)
    db.commit()
    return entry
