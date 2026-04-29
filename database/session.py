from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from database.models import Base

DATABASE_URL = "sqlite:///./face_id.db"   # swap for PostgreSQL in production

engine       = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Create all tables defined in models.py that do not yet exist."""
    # Import ALL model classes so SQLAlchemy registers them on Base.metadata
    # before create_all is called. Without this, new tables (Camera, etc.)
    # added after the initial schema won't be created on existing DBs.
    from database.models import (  # noqa: F401
        User, Camera, CameraSession, Detection, AuditLog, UnknownFace
    )
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency — yields a scoped session and guarantees cleanup."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def migrate_db():
    """
    Add any columns / tables that init_db() cannot create on existing DBs
    (SQLAlchemy create_all never alters existing tables).
    Safe to call every startup.
    """
    import sqlite3
    db_path = str(engine.url).replace("sqlite:///", "")
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()

    def _cols(table: str) -> set[str]:
        rows = cur.execute(f"PRAGMA table_info({table})").fetchall()
        return {r[1] for r in rows}

    def _tables() -> set[str]:
        return {r[0] for r in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}

    tables = _tables()

    # ── cameras ─────────────────────────────────────────────────────────────
    if "cameras" in tables:
        cc = _cols("cameras")
        if "run_recognition" not in cc:
            cur.execute("ALTER TABLE cameras ADD COLUMN run_recognition INTEGER DEFAULT 1")
            print("[migrate_db] cameras.run_recognition added")

    # ── users ────────────────────────────────────────────────────────────────
    if "users" in tables:
        uc2 = _cols("users")
        if "display_name" not in uc2:
            cur.execute("ALTER TABLE users ADD COLUMN display_name TEXT")
            print("[migrate_db] users.display_name added")

    # ── detections ──────────────────────────────────────────────────────────
    if "detections" in tables:
        dc = _cols("detections")
        if "name" not in dc:
            cur.execute("ALTER TABLE detections ADD COLUMN name TEXT")
            print("[migrate_db] detections.name added")
        if "image_path" not in dc:
            cur.execute("ALTER TABLE detections ADD COLUMN image_path TEXT")
            print("[migrate_db] detections.image_path added")
        if "camera_db_id" not in dc:
            cur.execute("ALTER TABLE detections ADD COLUMN camera_db_id INTEGER")
            print("[migrate_db] detections.camera_db_id added")

    # ── unknown_faces ────────────────────────────────────────────────────────
    if "unknown_faces" in tables:
        uc = _cols("unknown_faces")
        if "camera_db_id" not in uc:
            cur.execute("ALTER TABLE unknown_faces ADD COLUMN camera_db_id INTEGER")
            print("[migrate_db] unknown_faces.camera_db_id added")

    conn.commit()
    conn.close()


def seed_users():
    """
    Ensure every bootstrap entry in auth.USERS_DB exists in the SQL users table.
    Safe to call on every startup — skips already-existing usernames.
    """
    from datetime import datetime
    from database.models import User
    from api.auth import USERS_DB

    db = SessionLocal()
    try:
        for username, info in USERS_DB.items():
            if not db.query(User).filter(User.username == username).first():
                db.add(User(
                    username      = username,
                    password_hash = info["password_hash"],
                    role          = info["role"],
                    created_at    = datetime.utcnow(),
                    is_active     = True,
                ))
                print(f"[seed_users] Seeded '{username}'")
        db.commit()
    finally:
        db.close()


def seed_cameras():
    """
    Ensure at least one Camera row exists (source "0" = first local webcam).
    The server CameraManager will look this up or create on demand.
    """
    from database.models import Camera

    db = SessionLocal()
    try:
        if not db.query(Camera).first():
            db.add(Camera(name="Default Camera", source="0", is_active=True))
            db.commit()
            print("[seed_cameras] Default camera seeded")
    finally:
        db.close()