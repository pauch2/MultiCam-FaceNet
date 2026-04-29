from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.auth import get_current_user, require_role
from api.shared_db import face_db
from database.models import AuditLog, User
from database.session import get_db

router = APIRouter(prefix="/database", tags=["Database"])


# ── Helpers ────────────────────────────────────────────────────────────────────

def _audit(db: Session, actor, action: str, target_type: str,
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
        ))
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"[audit] Warning: could not write audit log — {e}")


# ── Schemas ────────────────────────────────────────────────────────────────────

class RenameRequest(BaseModel):
    new_name: str

class LinkRequest(BaseModel):
    face_name: str   # name in vector_store
    user_id:   int   # id in users table


# ── Face DB endpoints ──────────────────────────────────────────────────────────

@router.get("/summary")
def face_summary(
    current_user = Depends(require_role(["admin", "moderator"])),
    db: Session = Depends(get_db),
):
    """One row per distinct name: name, embedding count, linked user display_name & id."""
    from database.models import User as UserModel
    rows = face_db.get_summary()
    # Build username → user lookup
    user_map = {u.username: u for u in db.query(UserModel).all()}
    enriched = []
    for row in rows:
        u = user_map.get(row["name"])
        enriched.append({
            **row,
            "user_id":      u.id           if u else None,
            "display_name": (u.display_name or u.username) if u else None,
            "linked":       u is not None,
        })
    return {"data": enriched}


@router.get("/")
def list_faces(
    current_user = Depends(get_current_user),
):
    """
    Admin/Moderator: all records.
    Regular user: only their own name.
    """
    records = [{"id": r[0], "name": r[1]} for r in face_db.get_all_embeddings()]
    if current_user.role in ("admin", "moderator"):
        return {"records": records}
    return {"records": [r for r in records if r["name"] == current_user.username]}


@router.patch("/{face_name}/rename")
def rename_face(
    face_name: str,
    body: RenameRequest,
    current_user = Depends(require_role(["admin", "moderator"])),
    db: Session = Depends(get_db),
):
    """Rename all embeddings for face_name → body.new_name."""
    new_name = body.new_name.strip()
    if not new_name:
        raise HTTPException(400, "new_name cannot be empty")

    count = face_db.rename_person(face_name, new_name)
    if count == 0:
        raise HTTPException(404, f"No embeddings found for '{face_name}'")

    _audit(db, current_user, "rename_face", "face",
           details=f"{face_name!r} → {new_name!r} ({count} embeddings)")
    return {"renamed": count, "old_name": face_name, "new_name": new_name}


@router.delete("/name/{face_name}")
def delete_face_by_name(
    face_name: str,
    current_user = Depends(require_role(["admin"])),
    db: Session = Depends(get_db),
):
    """Admin: delete ALL embeddings for a given name."""
    count = face_db.delete_by_name(face_name)
    if count == 0:
        raise HTTPException(404, f"No embeddings found for '{face_name}'")

    _audit(db, current_user, "delete_face", "face",
           details=f"Deleted {count} embeddings for {face_name!r}")
    return {"deleted": count, "name": face_name}


@router.delete("/id/{face_id}")
def delete_face_by_id(
    face_id: int,
    current_user = Depends(require_role(["admin"])),
    db: Session = Depends(get_db),
):
    """Admin: delete a single embedding row by its DB id."""
    ok = face_db.delete_by_id(face_id)
    if not ok:
        raise HTTPException(404, f"No embedding with id={face_id}")

    _audit(db, current_user, "delete_embedding", "face", target_id=face_id,
           details=f"Deleted single embedding id={face_id}")
    return {"deleted": True, "id": face_id}