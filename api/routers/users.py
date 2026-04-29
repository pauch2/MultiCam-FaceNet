"""
users.py — Public registration + self-service password change.

POST /users/register          — public, creates a "user" role account
POST /users/me/password       — any authenticated user changes own password
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, field_validator
from sqlalchemy.orm import Session

from api.auth import get_password_hash, verify_password, get_current_user
from database.models import User
from database.session import get_db

router = APIRouter(prefix="/users", tags=["Users"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    username: str
    password: str
    password2: str

    @field_validator("username")
    @classmethod
    def username_ok(cls, v):
        v = v.strip()
        if len(v) < 3:
            raise ValueError("Username must be at least 3 characters")
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Username can only contain letters, digits, _ and -")
        return v

    @field_validator("password2")
    @classmethod
    def passwords_match(cls, v, info):
        pw = info.data.get("password", "")
        if v != pw:
            raise ValueError("Passwords do not match")
        return v


class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str
    new_password2: str

    @field_validator("new_password")
    @classmethod
    def pw_length(cls, v):
        if len(v) < 6:
            raise ValueError("New password must be at least 6 characters")
        return v

    @field_validator("new_password2")
    @classmethod
    def pw_match(cls, v, info):
        if v != info.data.get("new_password", ""):
            raise ValueError("New passwords do not match")
        return v


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/register", status_code=201)
def register(body: RegisterRequest, db: Session = Depends(get_db)):
    """Public self-registration. Always creates a 'user' role account."""
    username = body.username
    if len(body.password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters")
    if db.query(User).filter(User.username == username).first():
        raise HTTPException(409, f"Username '{username}' is already taken")

    user = User(
        username      = username,
        password_hash = get_password_hash(body.password),
        role          = "user",
        is_active     = True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"username": user.username, "role": user.role, "id": user.id}


@router.post("/me/password")
def change_password(
    body: ChangePasswordRequest,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Any authenticated user can change their own password."""
    user = db.query(User).filter(User.username == current_user.username).first()
    if not user:
        raise HTTPException(404, "User not found in DB")
    if not verify_password(body.old_password, user.password_hash):
        raise HTTPException(400, "Current password is incorrect")

    user.password_hash = get_password_hash(body.new_password)
    db.commit()
    return {"status": "Password changed successfully"}