"""
cameras.py — Camera registry, stream control, per-camera recognition toggle.

GET    /cameras/                 list all cameras + streaming status
POST   /cameras/                 register new camera
PATCH  /cameras/{id}             update name / source / flags
DELETE /cameras/{id}             remove camera
GET    /cameras/{id}/sessions    online/offline history

POST   /cameras/{id}/start       start streaming this camera
POST   /cameras/{id}/stop        stop streaming this camera
POST   /cameras/{id}/recognition toggle recognition on/off (persists to DB)

GET    /cameras/stream/{id}      MJPEG stream for one camera  (no auth — img src)
GET    /cameras/frame/{id}       single JPEG snapshot
"""

import asyncio
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request, Query
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.auth import require_role
from database.models import Camera, CameraSession
from database.session import get_db, engine

router = APIRouter(prefix="/cameras", tags=["Cameras"])


# ── Ensure tables exist ───────────────────────────────────────────────────────

def _ensure_tables():
    try:
        from database.models import Base
        Base.metadata.create_all(bind=engine)
    except Exception as exc:
        import traceback
        print(f"[cameras] WARNING: table ensure failed: {exc}")
        traceback.print_exc()

_ensure_tables()


# ── Pydantic ──────────────────────────────────────────────────────────────────

class CameraCreate(BaseModel):
    name:            str
    source:          str
    run_recognition: bool = True


class CameraUpdate(BaseModel):
    name:            str  | None = None
    source:          str  | None = None
    is_active:       bool | None = None
    run_recognition: bool | None = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt(ts: datetime | None) -> str:
    return ts.strftime("%Y-%m-%d %H:%M:%S") if ts else "-"


def _pool(request: Request):
    pool = getattr(request.app.state, "pool", None)
    if pool is None:
        raise HTTPException(503, "Camera pool not initialised")
    return pool


def _cam_dict(cam: Camera, streaming: bool = False,
               booting: bool = False, mgr=None) -> dict:
    thr = mgr.similarity_threshold if mgr is not None else None
    return {
        "id":                   cam.id,
        "name":                 cam.name,
        "source":               cam.source,
        "is_active":            cam.is_active,
        "run_recognition":      cam.run_recognition,
        "streaming":            streaming,
        "booting":              booting,
        "created_at":           _fmt(cam.created_at),
        "similarity_threshold": thr,
    }


def _session_dict(s: CameraSession) -> dict:
    diff = (s.ended_at or datetime.utcnow()) - s.started_at
    dur  = str(diff).split(".")[0]
    if not s.ended_at:
        dur += " (ongoing)"
    return {
        "id":         s.id,
        "camera_id":  s.camera_id,
        "started_at": _fmt(s.started_at),
        "ended_at":   _fmt(s.ended_at),
        "duration":   dur,
        "online":     s.ended_at is None,
    }


# ── Camera CRUD ───────────────────────────────────────────────────────────────

@router.get("/")
def list_cameras(
    request: Request,
    db: Session = Depends(get_db),
    current_user = Depends(require_role(["admin", "moderator"])),
):
    pool = _pool(request)
    try:
        cams = db.query(Camera).order_by(Camera.id).all()
    except Exception:
        _ensure_tables()
        cams = db.query(Camera).order_by(Camera.id).all()

    streaming_ids = set(pool.active_ids())
    return {
        "cameras":     [_cam_dict(c, c.id in streaming_ids,
                                   booting=pool.is_booting(c.id),
                                   mgr=pool.get(c.id)) for c in cams],
        "streaming":   sorted(streaming_ids),
    }


@router.post("/")
def create_camera(
    body: CameraCreate,
    request: Request,
    db: Session = Depends(get_db),
    current_user = Depends(require_role(["admin"])),
):
    name   = body.name.strip()
    source = body.source.strip()
    if not name:   raise HTTPException(400, "Name cannot be empty")
    if not source: raise HTTPException(400, "Source cannot be empty")
    if db.query(Camera).filter(Camera.name == name).first():
        raise HTTPException(400, f"Camera \"{name}\" already exists")

    cam = Camera(name=name, source=source, run_recognition=body.run_recognition)
    db.add(cam)
    db.commit()
    db.refresh(cam)
    pool = _pool(request)
    return {"status": "created", "camera": _cam_dict(cam, pool.is_streaming(cam.id))}


@router.patch("/{cam_id}")
def update_camera(
    cam_id: int,
    body: CameraUpdate,
    request: Request,
    db: Session = Depends(get_db),
    current_user = Depends(require_role(["admin"])),
):
    cam = db.query(Camera).filter(Camera.id == cam_id).first()
    if not cam: raise HTTPException(404, "Camera not found")

    if body.name            is not None: cam.name            = body.name.strip()
    if body.source          is not None: cam.source          = body.source.strip()
    if body.is_active       is not None: cam.is_active       = body.is_active
    if body.run_recognition is not None: cam.run_recognition = body.run_recognition

    db.commit()
    db.refresh(cam)

    # Sync recognition state to live manager if running
    pool = _pool(request)
    if body.run_recognition is not None:
        pool.set_recognition(cam_id, cam.run_recognition)

    return {"status": "updated", "camera": _cam_dict(cam, pool.is_streaming(cam_id))}


@router.delete("/{cam_id}")
def delete_camera(
    cam_id: int,
    request: Request,
    db: Session = Depends(get_db),
    current_user = Depends(require_role(["admin"])),
):
    cam = db.query(Camera).filter(Camera.id == cam_id).first()
    if not cam: raise HTTPException(404, "Camera not found")

    pool = _pool(request)
    if pool.is_streaming(cam_id):
        pool.stop(cam_id)

    db.delete(cam)
    db.commit()
    return {"status": "deleted", "camera_id": cam_id}


@router.get("/{cam_id}/sessions")
def get_sessions(
    cam_id: int,
    request: Request,
    limit: int = Query(100, le=1000),
    db: Session = Depends(get_db),
    current_user = Depends(require_role(["admin", "moderator"])),
):
    cam = db.query(Camera).filter(Camera.id == cam_id).first()
    if not cam: raise HTTPException(404, "Camera not found")

    pool     = _pool(request)
    sessions = (
        db.query(CameraSession)
        .filter(CameraSession.camera_id == cam_id)
        .order_by(CameraSession.started_at.desc())
        .limit(limit)
        .all()
    )
    return {
        "camera":   _cam_dict(cam, pool.is_streaming(cam_id)),
        "sessions": [_session_dict(s) for s in sessions],
    }


# ── Stream control ────────────────────────────────────────────────────────────

@router.post("/{cam_id}/start")
def start_camera(
    cam_id: int,
    request: Request,
    db: Session = Depends(get_db),
    current_user = Depends(require_role(["admin"])),
):
    """Start streaming this camera (creates a CameraManager in the pool)."""
    cam = db.query(Camera).filter(Camera.id == cam_id).first()
    if not cam: raise HTTPException(404, "Camera not found")

    pool = _pool(request)
    try:
        pool.start(cam_id, cam.source, run_recognition=cam.run_recognition)
    except Exception as exc:
        raise HTTPException(500, f"Failed to start stream: {exc}")

    # Boot runs in a background thread; streaming=True means boot was launched.
    return {"status": "starting", "camera": _cam_dict(cam, True)}


@router.post("/{cam_id}/stop")
def stop_camera(
    cam_id: int,
    request: Request,
    db: Session = Depends(get_db),
    current_user = Depends(require_role(["admin"])),
):
    """Stop streaming this camera."""
    cam = db.query(Camera).filter(Camera.id == cam_id).first()
    if not cam: raise HTTPException(404, "Camera not found")

    pool = _pool(request)
    pool.stop(cam_id)
    return {"status": "stopped", "camera": _cam_dict(cam, False)}


@router.post("/{cam_id}/recognition")
def toggle_recognition(
    cam_id: int,
    enabled: bool,
    request: Request,
    db: Session = Depends(get_db),
    current_user = Depends(require_role(["admin", "moderator"])),
):
    """Toggle recognition on/off for a camera (persists to DB + updates live stream)."""
    cam = db.query(Camera).filter(Camera.id == cam_id).first()
    if not cam: raise HTTPException(404, "Camera not found")

    cam.run_recognition = enabled
    db.commit()

    pool = _pool(request)
    pool.set_recognition(cam_id, enabled)

    return {
        "status":          "updated",
        "cam_id":          cam_id,
        "run_recognition": enabled,
    }


# ── Quality control ──────────────────────────────────────────────────────────

class QualityBody(BaseModel):
    jpeg_quality:         int   = 70    # 1-95: JPEG encode quality for browser delivery
    width:                int   = 0     # 0 = native: resize stream output (post-detection)
    height:               int   = 0     # 0 = native
    process_width:        int   = 0     # 0 = native: resize frame BEFORE detection runs
    similarity_threshold: float = -1.0  # -1 = don't change; 0-1 = new threshold

@router.post("/{cam_id}/quality")
def set_quality(
    cam_id: int,
    body: QualityBody,
    request: Request,
    current_user = Depends(require_role(["admin", "moderator"])),
):
    """Set JPEG encode quality, output resolution, processing resolution, and recognition threshold."""
    pool = _pool(request)
    pool.set_quality(cam_id, jpeg_quality=body.jpeg_quality,
                     width=body.width, height=body.height,
                     process_width=body.process_width)
    mgr = pool.get(cam_id)
    if mgr is not None and 0.0 <= body.similarity_threshold <= 1.0:
        mgr.similarity_threshold = float(body.similarity_threshold)  # set directly
        print(f"[DEBUG /quality] cam {cam_id}: mgr id={id(mgr)}  "
              f"threshold now={mgr.similarity_threshold:.3f}", flush=True)
    elif mgr is None:
        print(f"[DEBUG /quality] cam {cam_id}: mgr is None — pool has no live manager", flush=True)
    else:
        print(f"[DEBUG /quality] cam {cam_id}: threshold {body.similarity_threshold!r} "
              f"outside [0,1] — skipped", flush=True)
    thr = mgr.similarity_threshold if mgr else body.similarity_threshold
    return {"status": "updated", "cam_id": cam_id,
            "jpeg_quality": body.jpeg_quality,
            "width": body.width, "height": body.height,
            "process_width": body.process_width,
            "similarity_threshold": thr}


@router.get("/{cam_id}/metrics")
def get_metrics(cam_id: int, request: Request,
                current_user = Depends(require_role(["admin", "moderator"]))):
    """Live latency metrics: detection fps/ms, recognition fps/ms, frame fps/ms, avg batch size."""
    pool = _pool(request)
    return {"cam_id": cam_id, **pool.get_metrics(cam_id)}


@router.get("/{cam_id}/threshold")
def get_threshold(cam_id: int, request: Request,
                  current_user = Depends(require_role(["admin", "moderator"]))):
    """Return the current live similarity threshold for this camera."""
    pool = _pool(request)
    mgr  = pool.get(cam_id)
    if mgr is None:
        raise HTTPException(404, "Camera not streaming or not found")
    return {"cam_id": cam_id, "similarity_threshold": mgr.similarity_threshold}


@router.post("/{cam_id}/threshold")
def set_threshold_direct(
    cam_id: int,
    threshold: float = Query(..., ge=0.0, le=1.0),
    request: Request = None,
    current_user = Depends(require_role(["admin", "moderator"])),
):
    """Directly set the recognition threshold for a live camera."""
    pool = _pool(request)
    mgr  = pool.get(cam_id)
    if mgr is None:
        raise HTTPException(404, "Camera not streaming or not found")
    mgr.similarity_threshold = float(threshold)
    return {"cam_id": cam_id, "similarity_threshold": mgr.similarity_threshold}


@router.get("/{cam_id}/ping")
def ping_camera(cam_id: int, request: Request,
                current_user = Depends(require_role(["admin", "moderator"]))):
    """Return whether a frame is available right now."""
    pool  = _pool(request)
    frame = pool.get_frame_bytes(cam_id)
    return {"cam_id": cam_id, "alive": frame is not None,
            "streaming": pool.is_streaming(cam_id),
            "booting": pool.is_booting(cam_id)}


# ── Per-camera anchor / registration endpoints ───────────────────────────────
# These let the Register page use a specific server camera for face capture.

@router.get("/{cam_id}/anchors/status")
def cam_anchors_status(cam_id: int, request: Request,
                       current_user = Depends(require_role(["admin", "moderator"]))):
    pool = _pool(request)
    mgr  = pool.get(cam_id)
    if not mgr:
        raise HTTPException(503, "Camera not streaming")
    return mgr.anchors_status()


@router.post("/{cam_id}/anchors/start")
def cam_anchors_start(cam_id: int, request: Request,
                      interval_sec: float = Query(0.15, gt=0.0),
                      current_user = Depends(require_role(["admin", "moderator"])),
                      db: Session = Depends(get_db)):
    """Start anchor recording. Auto-starts camera if it is idle."""
    pool = _pool(request)
    cam  = db.query(Camera).filter(Camera.id == cam_id).first()
    if not cam: raise HTTPException(404, "Camera not found")

    # Auto-start camera if not already running
    if not pool.is_streaming(cam_id) and not pool.is_booting(cam_id):
        try:
            pool.start(cam_id, cam.source, run_recognition=False)
        except Exception as exc:
            raise HTTPException(500, f"Failed to start camera: {exc}")

    mgr = pool.get(cam_id)
    if mgr:
        mgr.run_detection  = True   # faces need to be detected to be captured
        mgr.run_recognition = False  # skip recognition while registering
        mgr.anchors_start_recording(interval_sec=interval_sec)
    return {"status": "recording_started", "cam_id": cam_id, "interval_sec": interval_sec}


@router.post("/{cam_id}/anchors/stop")
def cam_anchors_stop(cam_id: int, request: Request,
                     current_user = Depends(require_role(["admin", "moderator"]))):
    pool = _pool(request)
    mgr  = pool.get(cam_id)
    if not mgr:
        raise HTTPException(503, "Camera not streaming")
    mgr.anchors_stop_recording()
    return {"status": "recording_stopped", "cam_id": cam_id}


@router.post("/{cam_id}/anchors/reset")
def cam_anchors_reset(cam_id: int, request: Request,
                      current_user = Depends(require_role(["admin", "moderator"]))):
    pool = _pool(request)
    mgr  = pool.get(cam_id)
    if not mgr:
        raise HTTPException(503, "Camera not streaming")
    mgr.anchors_reset()
    return {"status": "samples_reset", "cam_id": cam_id}


@router.post("/{cam_id}/anchors/save")
def cam_anchors_save(cam_id: int, request: Request,
                     name: str = Query(..., min_length=1),
                     current_user = Depends(require_role(["admin"]))):
    pool = _pool(request)
    mgr  = pool.get(cam_id)
    if not mgr:
        raise HTTPException(503, "Camera not streaming")
    try:
        result = mgr.anchors_save(name.strip())
    except (ValueError, RuntimeError) as e:
        raise HTTPException(400, str(e))
    return result


# ── Registration mode toggle ──────────────────────────────────────────────────
# "Registration mode" = camera runs, detection ON, recognition OFF, ready for
# anchor capture.  It stays visible as a server-camera option in the Register page.

@router.post("/{cam_id}/reg_mode")
def set_reg_mode(
    cam_id: int,
    enabled: bool,
    request: Request,
    db: Session = Depends(get_db),
    current_user = Depends(require_role(["admin", "moderator"])),
):
    """Toggle registration mode for a camera.
    ON  → start camera (if idle), turn recognition OFF, keep detection ON.
    OFF → turn recognition back to its DB default, stop anchor recording.
    """
    cam = db.query(Camera).filter(Camera.id == cam_id).first()
    if not cam: raise HTTPException(404, "Camera not found")

    pool = _pool(request)

    if enabled:
        # Start camera if not already running
        if not pool.is_streaming(cam_id):
            try:
                pool.start(cam_id, cam.source, run_recognition=False)
            except Exception as exc:
                raise HTTPException(500, f"Failed to start camera: {exc}")
        else:
            pool.set_recognition(cam_id, False)
        # Ensure detection is on (needed to capture face crops)
        mgr = pool.get(cam_id)
        if mgr:
            mgr.run_detection = True
    else:
        # Restore recognition to DB default, stop any anchor recording
        pool.set_recognition(cam_id, cam.run_recognition)
        mgr = pool.get(cam_id)
        if mgr:
            mgr.anchors_stop_recording()
            mgr.run_detection = cam.run_recognition  # detection = recognition setting

    return {"status": "updated", "cam_id": cam_id, "registration_mode": enabled}


# ── MJPEG stream per camera ───────────────────────────────────────────────────

async def _mjpeg_gen(request: Request, pool, cam_id: int):
    while True:
        if await request.is_disconnected():
            break
        frame = pool.get_frame_bytes(cam_id)
        if frame:
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\nCache-Control: no-store\r\n\r\n"
                + frame + b"\r\n"
            )
        await asyncio.sleep(0.033)   # ~30 fps poll


@router.get("/stream/{cam_id}", include_in_schema=False)
async def camera_stream(cam_id: int, request: Request):
    """Public MJPEG stream for a single camera (no auth — used in <img> tags)."""
    pool = _pool(request)
    if not pool.is_streaming(cam_id):
        raise HTTPException(404, f"Camera {cam_id} is not streaming")
    return StreamingResponse(
        _mjpeg_gen(request, pool, cam_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )


@router.get("/frame/{cam_id}", include_in_schema=False)
def camera_frame(cam_id: int, request: Request,
                 current_user = Depends(require_role(["admin","moderator"]))):
    """Single JPEG snapshot for a camera."""
    pool  = _pool(request)
    frame = pool.get_frame_bytes(cam_id)
    if not frame:
        raise HTTPException(404, "No frame available")
    return Response(content=frame, media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"})