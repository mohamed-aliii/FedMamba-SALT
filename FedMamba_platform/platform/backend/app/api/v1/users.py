"""
User management endpoints (Admin operations + self-service).

Role rules:
- POST   /users/        ADMIN only — creates users with must_change_password=True
- GET    /users/        ADMIN only — paginated list
- GET    /users/{id}    ADMIN or self
- PATCH  /users/{id}    ADMIN or self; cannot change own role
- DELETE /users/{id}    ADMIN only — soft delete (is_active=False)
"""
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.core.audit import log_action
from app.core.deps import get_current_user, require_role
from app.core.security import hash_password
from app.database import get_db
from app.models.user import User, UserRole
from app.schemas.user import UserCreate, UserRead, UserUpdate

router = APIRouter(prefix="/users", tags=["users"])

_admin_only = require_role(UserRole.ADMIN)


@router.post("/", response_model=UserRead, status_code=201)
def create_user(
    body: UserCreate,
    request: Request,
    db: Session = Depends(get_db),
    _: User = Depends(_admin_only),
    current_user: User = Depends(get_current_user),
):
    if db.query(User).filter(User.email == body.email).first():
        raise HTTPException(status_code=409, detail="Email already registered")

    user = User(
        email=body.email,
        hashed_password=hash_password(body.password),
        full_name=body.full_name,
        role=body.role,
        hospital_id=body.hospital_id,
        must_change_password=True,  # Force reset on first login
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    log_action(
        db,
        user_id=current_user.id,
        action="CREATE_USER",
        resource_type="User",
        resource_id=str(user.id),
        details={"email": user.email, "role": user.role.value},
        request=request,
    )
    return user


@router.get("/", response_model=list[UserRead])
def list_users(
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db),
    _: User = Depends(_admin_only),
):
    return db.query(User).offset(skip).limit(limit).all()


@router.get("/{user_id}", response_model=UserRead)
def get_user(
    user_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if current_user.role != UserRole.ADMIN and current_user.id != user_id:
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.patch("/{user_id}", response_model=UserRead)
def update_user(
    user_id: uuid.UUID,
    body: UserUpdate,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if current_user.role != UserRole.ADMIN and current_user.id != user_id:
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if body.full_name is not None:
        user.full_name = body.full_name
    # is_active can only be changed by ADMIN
    if body.is_active is not None:
        if current_user.role != UserRole.ADMIN:
            raise HTTPException(status_code=403, detail="Only admins can change active status")
        user.is_active = body.is_active

    db.commit()
    db.refresh(user)

    log_action(
        db,
        user_id=current_user.id,
        action="UPDATE_USER",
        resource_type="User",
        resource_id=str(user.id),
        request=request,
    )
    return user


@router.delete("/{user_id}", status_code=204)
def delete_user(
    user_id: uuid.UUID,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(_admin_only),
):
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user.id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot deactivate your own account")

    user.is_active = False
    db.commit()

    log_action(
        db,
        user_id=current_user.id,
        action="DEACTIVATE_USER",
        resource_type="User",
        resource_id=str(user.id),
        request=request,
    )
