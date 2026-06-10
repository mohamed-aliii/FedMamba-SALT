"""
Experiments API endpoints.

Role access:
  POST   /experiments/           ADMIN — create
  GET    /experiments/           ADMIN=all, HOSPITAL_MANAGER=own hospital only
  GET    /experiments/{id}       ADMIN or own hospital
  PATCH  /experiments/{id}       ADMIN only, status must be PENDING
  POST   /experiments/{id}/start ADMIN — enqueue Celery task
  POST   /experiments/{id}/stop  ADMIN — enqueue stop task
  GET    /experiments/{id}/rounds       paginated
  GET    /experiments/{id}/checkpoints  list
  GET    /experiments/{id}/logs         SSE stream from Redis
"""
import json
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

import redis as redis_lib

from app.core.audit import log_action
from app.core.deps import get_current_user, get_redis, require_role
from app.database import get_db
from app.models.checkpoint import ModelCheckpoint
from app.models.experiment import ExperimentStatus, FLExperiment
from app.models.round import FLRound
from app.models.user import User, UserRole
from app.schemas.experiment import (
    CheckpointRead,
    FLExperimentCreate,
    FLExperimentRead,
    FLExperimentUpdate,
    FLRoundRead,
)

router = APIRouter(prefix="/experiments", tags=["experiments"])

_admin_only = require_role(UserRole.ADMIN)


def _get_experiment_or_404(db: Session, experiment_id: uuid.UUID) -> FLExperiment:
    exp = db.get(FLExperiment, experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return exp


def _check_experiment_access(experiment: FLExperiment, current_user: User) -> None:
    """ADMIN sees all; HOSPITAL_MANAGER only sees their hospital's experiments."""
    if current_user.role == UserRole.ADMIN:
        return
    if current_user.role == UserRole.HOSPITAL_MANAGER:
        # Check if hospital is participating
        participating = any(
            str(p.hospital_id) == str(current_user.hospital_id)
            for p in experiment.participations
        )
        if not participating:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return
    raise HTTPException(status_code=403, detail="Insufficient permissions")


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

@router.post("/", response_model=FLExperimentRead, status_code=201)
def create_experiment(
    body: FLExperimentCreate,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(_admin_only),
):
    experiment = FLExperiment(
        name=body.name,
        description=body.description,
        phase=body.phase,
        algorithm=body.algorithm,
        config_yaml=body.config_yaml,
        created_by_id=current_user.id,
        status=ExperimentStatus.PENDING,
    )
    db.add(experiment)
    db.commit()
    db.refresh(experiment)
    log_action(
        db,
        user_id=current_user.id,
        action="CREATE_EXPERIMENT",
        resource_type="FLExperiment",
        resource_id=str(experiment.id),
        details={"name": experiment.name, "phase": experiment.phase.value,
                 "algorithm": experiment.algorithm.value},
        request=request,
    )
    return experiment


@router.get("/", response_model=list[FLExperimentRead])
def list_experiments(
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if current_user.role == UserRole.ADMIN:
        return db.query(FLExperiment).offset(skip).limit(limit).all()
    if current_user.role == UserRole.HOSPITAL_MANAGER:
        # Return experiments where the user's hospital is participating
        return (
            db.query(FLExperiment)
            .join(FLExperiment.participations)
            .filter_by(hospital_id=current_user.hospital_id)
            .offset(skip).limit(limit).all()
        )
    raise HTTPException(status_code=403, detail="Insufficient permissions")


@router.get("/{experiment_id}", response_model=FLExperimentRead)
def get_experiment(
    experiment_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    experiment = _get_experiment_or_404(db, experiment_id)
    _check_experiment_access(experiment, current_user)
    return experiment


@router.patch("/{experiment_id}", response_model=FLExperimentRead)
def update_experiment(
    experiment_id: uuid.UUID,
    body: FLExperimentUpdate,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(_admin_only),
):
    experiment = _get_experiment_or_404(db, experiment_id)
    if experiment.status != ExperimentStatus.PENDING:
        raise HTTPException(
            status_code=409,
            detail=f"Can only update PENDING experiments (current: {experiment.status.value})",
        )
    if body.name is not None:
        experiment.name = body.name
    if body.description is not None:
        experiment.description = body.description
    if body.config_yaml is not None:
        experiment.config_yaml = body.config_yaml
    db.commit()
    db.refresh(experiment)
    log_action(
        db, user_id=current_user.id, action="UPDATE_EXPERIMENT",
        resource_type="FLExperiment", resource_id=str(experiment.id), request=request,
    )
    return experiment


# ---------------------------------------------------------------------------
# Control
# ---------------------------------------------------------------------------

@router.post("/{experiment_id}/start", status_code=202)
def start_experiment(
    experiment_id: uuid.UUID,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(_admin_only),
):
    experiment = _get_experiment_or_404(db, experiment_id)
    if experiment.status != ExperimentStatus.PENDING:
        raise HTTPException(
            status_code=409,
            detail=f"Experiment is not PENDING (status={experiment.status.value})",
        )
    from app.tasks.fl_tasks import run_fl_experiment
    run_fl_experiment.delay(str(experiment.id))
    log_action(
        db, user_id=current_user.id, action="START_EXPERIMENT",
        resource_type="FLExperiment", resource_id=str(experiment.id), request=request,
    )
    return {"detail": "Experiment queued", "experiment_id": str(experiment.id)}


@router.post("/{experiment_id}/stop", status_code=202)
def stop_experiment(
    experiment_id: uuid.UUID,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(_admin_only),
):
    experiment = _get_experiment_or_404(db, experiment_id)
    if experiment.status != ExperimentStatus.RUNNING:
        raise HTTPException(
            status_code=409,
            detail=f"Experiment is not RUNNING (status={experiment.status.value})",
        )
    from app.tasks.fl_tasks import stop_fl_experiment
    stop_fl_experiment.delay(str(experiment.id))
    log_action(
        db, user_id=current_user.id, action="STOP_EXPERIMENT",
        resource_type="FLExperiment", resource_id=str(experiment.id), request=request,
    )
    return {"detail": "Stop signal sent", "experiment_id": str(experiment.id)}


# ---------------------------------------------------------------------------
# Sub-resources
# ---------------------------------------------------------------------------

@router.get("/{experiment_id}/rounds", response_model=list[FLRoundRead])
def list_rounds(
    experiment_id: uuid.UUID,
    skip: int = 0,
    limit: int = 200,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    experiment = _get_experiment_or_404(db, experiment_id)
    _check_experiment_access(experiment, current_user)
    return (
        db.query(FLRound)
        .filter(FLRound.experiment_id == experiment_id)
        .order_by(FLRound.round_number)
        .offset(skip).limit(limit).all()
    )


@router.get("/{experiment_id}/checkpoints", response_model=list[CheckpointRead])
def list_checkpoints(
    experiment_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    experiment = _get_experiment_or_404(db, experiment_id)
    _check_experiment_access(experiment, current_user)
    return (
        db.query(ModelCheckpoint)
        .filter(ModelCheckpoint.experiment_id == experiment_id)
        .all()
    )


@router.get("/{experiment_id}/logs")
def stream_logs(
    experiment_id: uuid.UUID,
    db: Session = Depends(get_db),
    redis: redis_lib.Redis = Depends(get_redis),
    current_user: User = Depends(get_current_user),
):
    """SSE stream: returns buffered logs from Redis list, then closes.

    For live streaming during a run, use GET /sse/experiments/{id}.
    This endpoint returns the historical log buffer.
    """
    experiment = _get_experiment_or_404(db, experiment_id)
    _check_experiment_access(experiment, current_user)

    logs = redis.lrange(f"experiment:{experiment_id}:logs", 0, -1)

    def _generate():
        for line in logs:
            yield f"data: {json.dumps({'log': line})}\n\n"
        yield "data: {\"type\": \"end\"}\n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream")
