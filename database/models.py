from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime

Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id            = Column(Integer, primary_key=True, index=True)
    username      = Column(String, unique=True, index=True)   # login credential — never shown in UI
    display_name  = Column(String, nullable=True)              # shown in UI / detection logs
    password_hash = Column(String)
    role          = Column(String, default="user")
    created_at    = Column(DateTime, default=datetime.utcnow)
    is_active     = Column(Boolean, default=True)


class Camera(Base):
    """One row per physical (or virtual) camera."""
    __tablename__ = "cameras"
    id              = Column(Integer, primary_key=True, index=True)
    name            = Column(String, unique=True, index=True)
    source          = Column(String, nullable=False)   # "0", "rtsp://…", "http://…"
    is_active       = Column(Boolean, default=True)
    run_recognition = Column(Boolean, default=True)    # per-camera recognition toggle
    created_at      = Column(DateTime, default=datetime.utcnow)

    sessions = relationship("CameraSession", back_populates="camera",
                            cascade="all, delete-orphan")


class CameraSession(Base):
    """Tracks when a camera stream is online / offline."""
    __tablename__ = "camera_sessions"
    id         = Column(Integer, primary_key=True, index=True)
    camera_id  = Column(Integer, ForeignKey("cameras.id", ondelete="CASCADE"),
                        nullable=False, index=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at   = Column(DateTime, nullable=True)

    camera = relationship("Camera", back_populates="sessions")


class Detection(Base):
    __tablename__ = "detections"
    id           = Column(Integer, primary_key=True, index=True)
    user_id      = Column(Integer, ForeignKey("users.id"), nullable=True)
    camera_db_id = Column(Integer, ForeignKey("cameras.id"), nullable=True, index=True)
    name         = Column(String,  nullable=True)
    timestamp    = Column(DateTime, default=datetime.utcnow)
    image_path   = Column(String,  nullable=True)
    camera_id    = Column(String,  nullable=True)
    confidence   = Column(Float,   nullable=True)


class AuditLog(Base):
    __tablename__ = "audit_logs"
    id          = Column(Integer, primary_key=True, index=True)
    actor_id    = Column(Integer, ForeignKey("users.id"))
    action      = Column(String)
    target_type = Column(String)
    target_id   = Column(Integer)
    timestamp   = Column(DateTime, default=datetime.utcnow)
    details     = Column(String)


class UnknownFace(Base):
    __tablename__ = "unknown_faces"
    id           = Column(Integer, primary_key=True, index=True)
    camera_db_id = Column(Integer, ForeignKey("cameras.id"), nullable=True, index=True)
    timestamp    = Column(DateTime, default=datetime.utcnow)
    image_path   = Column(String, nullable=True)
    camera_id    = Column(String, nullable=True)