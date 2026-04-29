from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.auth import require_role, get_current_user, get_password_hash
from database.models import AuditLog, User
from database.session import get_db

router = APIRouter(prefix="/admin", tags=["Admin"])


# ── Helpers ────────────────────────────────────────────────────────────────────

def _audit(db: Session, actor, action: str, target_type: str = "",
           target_id: int | None = None, details: str | None = None):
    """Write one audit log entry. Never raises — audit failure must not break the main action."""
    try:
        user = db.query(User).filter(User.username == actor.username).first()
        db.add(AuditLog(
            actor_id    = user.id if user else None,
            action      = action,
            target_type = target_type,
            target_id   = target_id,
            details     = details,
            timestamp   = datetime.utcnow(),
        ))
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"[audit] Warning: could not write audit log — {e}")


# ── Schemas ────────────────────────────────────────────────────────────────────

class CreateUserRequest(BaseModel):
    username: str
    password: str
    role: str = "user"   # "user" | "moderator" | "admin"

class UpdateRoleRequest(BaseModel):
    role: str

class UpdateNameRequest(BaseModel):
    display_name: str


# ── User management (admin only) ───────────────────────────────────────────────

@router.get("/users")
def list_users(
    current_user = Depends(require_role(["admin", "moderator"])),
    db: Session = Depends(get_db),
):
    users = db.query(User).order_by(User.id).all()
    return {"data": [
        {"id": u.id, "username": u.username,
         "display_name": u.display_name or u.username,
         "role": u.role,
         "is_active": u.is_active, "created_at": u.created_at.isoformat() if u.created_at else None}
        for u in users
    ]}


@router.post("/users", status_code=201)
def create_user(
    body: CreateUserRequest,
    current_user = Depends(require_role(["admin"])),
    db: Session = Depends(get_db),
):
    if body.role not in ("user", "moderator", "admin"):
        raise HTTPException(400, "role must be user | moderator | admin")
    if db.query(User).filter(User.username == body.username).first():
        raise HTTPException(409, f"Username '{body.username}' already exists")

    user = User(
        username      = body.username.strip(),
        password_hash = get_password_hash(body.password),
        role          = body.role,
        created_at    = datetime.utcnow(),
        is_active     = True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    _audit(db, current_user, "create_user", "user", target_id=user.id,
           details=f"Created user '{user.username}' with role '{user.role}'")
    return {"id": user.id, "username": user.username, "display_name": user.display_name or user.username, "role": user.role}


@router.patch("/users/{user_id}/role")
def update_user_role(
    user_id: int,
    body: UpdateRoleRequest,
    current_user = Depends(require_role(["admin"])),
    db: Session = Depends(get_db),
):
    if body.role not in ("user", "moderator", "admin"):
        raise HTTPException(400, "role must be user | moderator | admin")
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(404, "User not found")

    old_role   = user.role
    user.role  = body.role
    db.commit()

    _audit(db, current_user, "update_role", "user", target_id=user_id,
           details=f"'{user.username}': {old_role} → {body.role}")
    return {"id": user.id, "username": user.username, "display_name": user.display_name or user.username, "role": user.role}


@router.patch("/users/{user_id}/name")
def update_user_name(
    user_id: int,
    body: UpdateNameRequest,
    current_user = Depends(require_role(["admin", "moderator"])),
    db: Session = Depends(get_db),
):
    """Moderator/Admin: update a user's display name (not their login username)."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(404, "User not found")
    old_name = user.display_name
    user.display_name = body.display_name.strip()
    db.commit()
    _audit(db, current_user, "update_display_name", "user", target_id=user_id,
           details=f"'{user.username}': display_name '{old_name}' → '{user.display_name}'")
    return {"id": user.id, "username": user.username, "display_name": user.display_name}


@router.delete("/users/{user_id}")
def delete_user(
    user_id: int,
    current_user = Depends(require_role(["admin"])),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(404, "User not found")

    username = user.username
    db.delete(user)
    db.commit()

    _audit(db, current_user, "delete_user", "user", target_id=user_id,
           details=f"Deleted user '{username}'")
    return {"deleted": True, "id": user_id, "username": username}


# ── Audit log (admin only) ─────────────────────────────────────────────────────

@router.get("/audit")
def get_audit_logs(
    limit: int = Query(200, le=1000),
    current_user = Depends(require_role(["admin"])),
    db: Session = Depends(get_db),
):
    rows = (
        db.query(AuditLog)
        .order_by(AuditLog.timestamp.desc())
        .limit(limit)
        .all()
    )

    # Resolve actor_id -> username in one query (avoid N+1)
    actor_ids = {r.actor_id for r in rows if r.actor_id is not None}
    user_map: dict[int, str] = {}
    if actor_ids:
        users = db.query(User).filter(User.id.in_(actor_ids)).all()
        user_map = {u.id: u.username for u in users}

    def fmt_time(ts):
        if not ts: return "-"
        return ts.strftime("%Y-%m-%d %H:%M:%S")

    return {"status": "success", "data": [
        {
            "id":          r.id,
            "actor_id":    r.actor_id,
            "actor":       user_map.get(r.actor_id, f"#{r.actor_id}" if r.actor_id else "-"),
            "action":      r.action,
            "target_type": r.target_type,
            "target_id":   r.target_id,
            "details":     r.details or "-",
            "timestamp":   fmt_time(r.timestamp),
        }
        for r in rows
    ]}