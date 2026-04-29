from __future__ import annotations

import os
import asyncio
from typing import Optional

from fastapi import APIRouter, Depends, Request, HTTPException, Query
from fastapi.responses import StreamingResponse, Response, HTMLResponse
from pydantic import BaseModel

from api.auth import get_current_user, require_role, User

router = APIRouter(prefix="/stream", tags=["Video Stream"])

# DEV-only: enable public stream endpoint for <img> tags (no Authorization header)
PUBLIC_STREAM_ENABLED = os.getenv("PUBLIC_STREAM_ENABLED", "1") == "1"


class StreamSettings(BaseModel):
    camera_source: str | int
    run_detection: bool
    run_recognition: bool
    anti_spoofing: bool


def get_camera(request: Request):
    # Legacy: single camera in app.state.camera
    cam = getattr(request.app.state, "camera", None)
    if cam is not None:
        return cam
    # New: pick first streaming camera from the pool
    pool = getattr(request.app.state, "pool", None)
    if pool is not None:
        ids = pool.active_ids()
        if ids:
            return pool.get(ids[0])
    raise HTTPException(status_code=503,
        detail="No camera is streaming. Start a camera in section 7.")


def _translate_camera_error(e: Exception) -> HTTPException:
    """
    Normalize CameraManager errors into client-friendly responses.
    """
    # Typical "user / state" errors -> 400
    if isinstance(e, (ValueError, RuntimeError)):
        return HTTPException(status_code=400, detail=str(e))

    # Anything else -> 500
    return HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


async def mjpeg_generator(request: Request, camera):
    """
    Yields MJPEG frames until client disconnects.
    IMPORTANT: This is an infinite stream; Swagger UI will show "Loading..." forever.
    """
    while True:
        if await request.is_disconnected():
            break

        frame_bytes = camera.get_frame_bytes()
        if frame_bytes:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Cache-Control: no-store\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )

        await asyncio.sleep(0.03)


@router.get("/live")
async def live_stream(
    request: Request,
    current_user: User = Depends(get_current_user),
    camera=Depends(get_camera),
):
    headers = {
        "Cache-Control": "no-store",
        "Pragma": "no-cache",
        "Expires": "0",
    }
    return StreamingResponse(
        mjpeg_generator(request, camera),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers=headers,
    )


@router.get("/frame")
def snapshot(
    current_user: User = Depends(get_current_user),
    camera=Depends(get_camera),
):
    frame_bytes = camera.get_frame_bytes()
    if not frame_bytes:
        raise HTTPException(status_code=503, detail="No frame available yet")
    return Response(content=frame_bytes, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@router.post("/settings")
def update_settings(
    settings: StreamSettings,
    current_user: User = Depends(get_current_user),
    camera=Depends(get_camera),
):
    # Avoid unnecessary reopen
    if str(camera.camera_source) != str(settings.camera_source):
        camera.set_source(settings.camera_source)

    camera.run_detection = settings.run_detection
    camera.run_recognition = settings.run_recognition
    camera.anti_spoofing = settings.anti_spoofing
    return {"status": "Settings updated"}


@router.post("/register")
def register_face(
    name: str = Query(..., min_length=1),
    current_user: User = Depends(require_role(["admin"])),
    camera=Depends(get_camera),
):
    camera.trigger_registration(name.strip())
    return {"status": "Registration triggered. Please look at the camera."}


@router.get("/view", response_class=HTMLResponse)
def view_page():
    # Note: /stream/live needs Authorization header so <img> won't work.
    # Use /stream/live_public for dev viewing (if enabled).
    public_note = "ENABLED" if PUBLIC_STREAM_ENABLED else "DISABLED"
    return f"""
    <html>
      <body>
        <h2>MJPEG Stream</h2>
        <p>/stream/live requires Authorization header (Swagger works, &lt;img&gt; doesn't).</p>
        <p>/stream/live_public is {public_note} (env PUBLIC_STREAM_ENABLED=1/0)</p>
        <hr/>
        <h3>Public stream (dev only)</h3>
        <img src="/stream/live_public" />
      </body>
    </html>
    """


@router.get("/live_public")
async def live_stream_public(
    request: Request,
    camera=Depends(get_camera),
):
    if not PUBLIC_STREAM_ENABLED:
        raise HTTPException(status_code=404, detail="Public stream is disabled")

    headers = {
        "Cache-Control": "no-store",
        "Pragma": "no-cache",
        "Expires": "0",
    }
    return StreamingResponse(
        mjpeg_generator(request, camera),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers=headers,
    )


# -------------------------
# Anchor registration APIs
# -------------------------

@router.get("/anchors/status")
def anchors_status(
    current_user: User = Depends(require_role(["admin"])),
    camera=Depends(get_camera),
):
    try:
        return camera.anchors_status()
    except Exception as e:
        raise _translate_camera_error(e)


@router.post("/anchors/freeze")
def anchors_freeze(
    current_user: User = Depends(require_role(["admin"])),
    camera=Depends(get_camera),
):
    try:
        camera.anchors_freeze()
        return {"status": "frozen"}
    except Exception as e:
        raise _translate_camera_error(e)


@router.post("/anchors/unfreeze")
def anchors_unfreeze(
    current_user: User = Depends(require_role(["admin"])),
    camera=Depends(get_camera),
):
    try:
        camera.anchors_unfreeze()
        return {"status": "unfrozen"}
    except Exception as e:
        raise _translate_camera_error(e)


@router.post("/anchors/click")
def anchors_click(
    x: int = Query(..., ge=0),
    y: int = Query(..., ge=0),
    current_user: User = Depends(require_role(["admin"])),
    camera=Depends(get_camera),
):
    try:
        return camera.anchors_click_capture(x, y)
    except Exception as e:
        raise _translate_camera_error(e)


@router.post("/anchors/start")
def anchors_start(
    interval_sec: float = Query(0.2, gt=0.0),  # 0.2 ~ 5/sec
    current_user: User = Depends(require_role(["admin"])),
    camera=Depends(get_camera),
):
    try:
        camera.anchors_start_recording(interval_sec=interval_sec)
        return {"status": "recording_started", "interval_sec": interval_sec}
    except Exception as e:
        raise _translate_camera_error(e)


@router.post("/anchors/stop")
def anchors_stop(
    current_user: User = Depends(require_role(["admin"])),
    camera=Depends(get_camera),
):
    try:
        camera.anchors_stop_recording()
        return {"status": "recording_stopped"}
    except Exception as e:
        raise _translate_camera_error(e)


@router.post("/anchors/reset")
def anchors_reset(
    current_user: User = Depends(require_role(["admin"])),
    camera=Depends(get_camera),
):
    try:
        camera.anchors_reset()
        return {"status": "samples_reset"}
    except Exception as e:
        raise _translate_camera_error(e)


@router.post("/anchors/save")
def anchors_save(
    name: str = Query(..., min_length=1),
    current_user: User = Depends(require_role(["admin"])),
    camera=Depends(get_camera),
):
    try:
        return camera.anchors_save(name.strip())
    except Exception as e:
        raise _translate_camera_error(e)