"""
client_cam.py — WebSocket endpoint for browser webcam frames.

Two modes (toggled per-connection via JSON text messages):

  recognize  (default)
    - Server runs detection + recognition on each frame.
    - Returns JSON: {type:"result", w, h, detections:[...]}

  register
    - Server detects faces, embeds the largest one, accumulates embeddings
      in an in-memory buffer scoped to this connection.
    - Returns JSON: {type:"register_frame", detected:bool, samples:N}
    - "register_save" flushes the buffer to the face DB.
    - "register_reset" clears the buffer without saving.

Binary messages are always JPEG frames.
Text messages are JSON control commands.
"""

import json
import logging

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from api.auth import decode_token
from api.frame_processor import FrameProcessor
from api.shared_db import face_db

log    = logging.getLogger(__name__)
router = APIRouter(prefix="/clientcam", tags=["Client Camera"])

# Shared, stateless per-frame helpers — safe to reuse across connections
_processor = FrameProcessor()
_db        = face_db   # shared singleton — same cache as frame_processor.py


# ── Auth ──────────────────────────────────────────────────────────────────────

def _auth_ws_token(token: str) -> dict | None:
    try:
        payload  = decode_token(token)
        username = payload.get("sub")
        return {"username": username, "role": payload.get("role")} if username else None
    except Exception:
        return None


# ── Per-connection state ──────────────────────────────────────────────────────

class _ConnState:
    def __init__(self):
        self.mode: str             = "recognize"   # "recognize" | "register"
        self.run_recognition: bool = True
        self.anchor_samples: list  = []            # list[np.ndarray]

    def reset_samples(self):
        self.anchor_samples.clear()


# ── WebSocket ─────────────────────────────────────────────────────────────────

@router.websocket("/ws")
async def ws_client_camera(ws: WebSocket):
    """
    Authenticate via ?token=<JWT> query param.

    Text message API
    ─────────────────────────────────────────────────────────────────────────
    {"type": "set_mode",       "mode": "recognize"|"register"}
    {"type": "settings",       "run_recognition": true|false}
    {"type": "register_save",  "name": "<person name>"}
    {"type": "register_reset"}

    Server replies
    ─────────────────────────────────────────────────────────────────────────
    recognize:      {"type": "result",          "w":…, "h":…, "detections":[…]}
    register frame: {"type": "register_frame",  "detected": bool, "samples": N}
    save done:      {"type": "register_saved",  "name": …, "anchors_saved": N}
    ack:            {"type": "ack",             …}
    error:          {"type": "error",           "detail": "…"}
    """
    token = ws.query_params.get("token", "")
    user  = _auth_ws_token(token)
    if not user:
        await ws.close(code=1008)
        return

    await ws.accept()
    log.info("[clientcam] connected: %s (%s)", user["username"], user["role"])
    state = _ConnState()

    try:
        while True:
            msg = await ws.receive()

            # Starlette delivers a disconnect dict instead of raising
            # WebSocketDisconnect when the client closes cleanly.
            # Calling receive() again after this raises RuntimeError.
            if msg.get("type") == "websocket.disconnect":
                break

            if msg.get("text") is not None:
                await _handle_text(ws, msg["text"], state)
            elif msg.get("bytes") is not None:
                await _handle_frame(ws, msg["bytes"], state)

    except WebSocketDisconnect:
        pass
    except RuntimeError:
        pass   # already disconnected — safe to ignore
    finally:
        log.info("[clientcam] disconnected: %s", user["username"])


async def _handle_text(ws: WebSocket, raw: str, state: _ConnState):
    try:
        data = json.loads(raw)
    except Exception:
        await ws.send_json({"type": "error", "detail": "Invalid JSON"})
        return

    t = data.get("type", "")

    if t == "set_mode":
        mode = data.get("mode", "recognize")
        if mode not in ("recognize", "register"):
            await ws.send_json({"type": "error", "detail": f"Unknown mode: {mode}"})
            return
        state.mode = mode
        # Do NOT reset samples on a mode switch. The client switches to
        # "recognize" after collecting enough samples but still needs them
        # for the subsequent register_save. Only reset on register_reset or save.
        await ws.send_json({"type": "ack", "mode": state.mode,
                            "samples": len(state.anchor_samples)})

    elif t == "settings":
        state.run_recognition = bool(data.get("run_recognition", True))
        await ws.send_json({"type": "ack", "settings": {"run_recognition": state.run_recognition}})

    elif t == "register_save":
        name = (data.get("name") or "").strip()
        if not name:
            await ws.send_json({"type": "error", "detail": "Name cannot be empty"})
            return
        if not state.anchor_samples:
            await ws.send_json({"type": "error", "detail": "No samples captured yet"})
            return
        _db.add_person_many(name, state.anchor_samples)
        saved = len(state.anchor_samples)
        state.reset_samples()
        state.mode = "recognize"
        log.info("[clientcam] registered '%s' — %d anchors", name, saved)
        await ws.send_json({"type": "register_saved", "name": name, "anchors_saved": saved})

    elif t == "register_reset":
        state.reset_samples()
        await ws.send_json({"type": "ack", "samples": 0})

    else:
        await ws.send_json({"type": "error", "detail": f"Unknown message type: {t}"})


async def _handle_frame(ws: WebSocket, jpg_bytes: bytes, state: _ConnState):
    nparr = np.frombuffer(jpg_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        await ws.send_json({"type": "error", "detail": "JPEG decode failed"})
        return

    h, w = frame.shape[:2]

    # ── Recognize ─────────────────────────────────────────────────────────────
    if state.mode == "recognize":
        dets = _processor.process(frame, run_recognition=state.run_recognition)
        await ws.send_json({"type": "result", "w": w, "h": h, "detections": dets})
        return

    # ── Register ──────────────────────────────────────────────────────────────
    detections = _processor.detect_only(frame)   # list[(bbox, crop_bgr)]

    if not detections:
        await ws.send_json({"type": "register_frame", "detected": False,
                            "samples": len(state.anchor_samples)})
        return

    def _area(bbox):
        x1, y1, x2, y2 = bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    _bbox, crop = max(detections, key=lambda t: _area(t[0]))
    emb         = _processor.embed(crop)

    if emb is None:
        await ws.send_json({"type": "register_frame", "detected": True,
                            "detail": "Embedding failed",
                            "samples": len(state.anchor_samples)})
        return

    state.anchor_samples.append(emb)
    x1, y1, x2, y2 = [int(v) for v in _bbox]
    await ws.send_json({"type": "register_frame", "detected": True,
                        "samples": len(state.anchor_samples),
                        "bbox": [x1, y1, x2, y2]})