import cv2
import threading
import time
import sys
import os
import platform
from collections import deque

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.detector import FaceDetector
from models.embedding_model import load_embedding_model
from database.vector_store import FaceDatabase
from database.models import Camera, CameraSession
from database.session import SessionLocal
import config

def _bbox_area(bbox) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)

def _point_in_bbox(x: int, y: int, bbox) -> bool:
    x1, y1, x2, y2 = bbox
    return (x1 <= x <= x2) and (y1 <= y <= y2)

class CameraManager:
    def __init__(self, source=0, fps=30):
        s = str(source).strip()
        self.camera_source = int(s) if s.isdigit() else s
        self.target_dt = 1.0 / max(1, int(fps))

        # Stream quality controls
        self.jpeg_quality  = 70    # 1-95, lower = faster/smaller
        self.stream_width  = 0     # 0 = native; else resize before encode
        self.stream_height = 0

        # Processing resolution: resize frame BEFORE detection + display.
        # Smaller = faster ML, circles proportional to display, lower bandwidth.
        # 0 = use native camera resolution.
        self.process_width = 0

        # Per-camera recognition threshold (cosine similarity 0-1).
        # Overrides config.SIMILARITY_THRESHOLD for this camera.
        self.similarity_threshold: float = getattr(config, "SIMILARITY_THRESHOLD", 0.5)

        # ── Latency metrics ────────────────────────────────────────────────
        # Rolling window of the last N timings (seconds).
        _W = 30
        self._det_times: deque = deque(maxlen=_W)   # YOLO detection wall time
        self._rec_times: deque = deque(maxlen=_W)   # embed + vector-search time
        self._frame_times: deque = deque(maxlen=_W) # full frame processing time

        # ML State Toggles
        self.run_detection = False
        self.run_recognition = False
        self.anti_spoofing = False  # Placeholder for future logic

        self.register_next_face = False
        self.register_name = ""

        self.detector = FaceDetector()
        self.db = FaceDatabase()
        # ---- device / preprocessing ----
        device_str = getattr(config, "DEVICE", None)
        self.device = torch.device(device_str) if device_str else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_embedding_model(device=str(self.device))

        # Register with the process-wide batch embedder so all cameras share
        # one GPU forward pass.  The singleton is created here on the first
        # CameraManager and reused by all subsequent ones.
        from api.batch_embedder import get_batch_embedder
        self._embedder = get_batch_embedder(
            model=self.model, device=self.device, wait_ms=8.0
        )

        # If you already have a "transform" in your scripts, mirror it here.
        # Fallback defaults: 224 + ImageNet normalization
        input_size = getattr(config, "INPUT_SIZE", 224)
        mean = getattr(config, "NORM_MEAN", (0.485, 0.456, 0.406))
        std = getattr(config, "NORM_STD", (0.229, 0.224, 0.225))
        self.transform = T.Compose([
            T.Resize((input_size, input_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

        # Optional: mirror-flip like your script
        self.flip_horizontal = getattr(config, "FLIP_HORIZONTAL", True)

        # ---- periodic logging like your script ----
        self.log_interval_sec = getattr(config, "LOG_INTERVAL_SEC", 2.0)
        self._last_log_time = 0.0
        self.logger = None  # you can set this from outside if you have a logger with .log(name, sim, status)

        # Locks / state
        self._cap_lock = threading.Lock()
        self._frame_lock = threading.Lock()
        self._stop_event = threading.Event()

        self.cap = None
        self.current_frame = None

        # Raw frame shared between capture thread → processing thread.
        # The capture thread writes here as fast as possible (no ML blocking it).
        # The processing thread reads the latest value whenever it's free.
        self._raw_frame      = None
        self._raw_frame_lock = threading.Lock()

        self._open_capture(self.camera_source)

        # Internal camera/session DB IDs (set by _register_camera)
        self._camera_db_id  = None
        self._session_db_id = None
        self._register_camera()

        # Thread 1: capture — drains buffer, always keeps _raw_frame fresh.
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

        # Thread 2: process — reads _raw_frame, runs ML, writes current_frame.
        self.thread = threading.Thread(target=self._update_frame, daemon=True)
        self.thread.start()

        # ---- Anchor registration state (like your script) ----
        self._anchor_lock = threading.Lock()

        self._fp = None   # FrameProcessor, lazy-loaded for DB logging

        self.anchor_frozen = False
        self.anchor_frozen_frame = None                  # np.ndarray
        self.anchor_frozen_visible = []                  # list[(bbox, crop)]

        self.anchor_visible = []                         # live visible list for selection/recording

        self.anchor_recording = False
        self.anchor_sample_interval_sec = 0.1            # ~10/sec (set to 0.2 for ~5/sec)
        self._anchor_last_sample_time = 0.0

        self.anchor_samples = []                         # list[np.ndarray] embeddings buffer

        self.anchor_last_mark_bbox = None
        self.anchor_last_mark_frames_left = 0

    def _backend_flag(self):
        """Use DirectShow on Windows ONLY for integer (webcam index) sources.
        RTSP / HTTP / file URLs must use the default (FFMPEG) backend.
        """
        src = self.camera_source
        if platform.system().lower().startswith("win") and isinstance(src, int):
            return cv2.CAP_DSHOW
        return 0  # default / FFMPEG backend

    def _open_capture(self, source, timeout_sec: float = 8.0):
        """Open VideoCapture with a hard wall-clock timeout.

        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC is not honoured by all OpenCV builds /
        backends, so we run the blocking VideoCapture() call in a daemon thread
        and join it for at most *timeout_sec* seconds.  If it times out we
        raise RuntimeError immediately so the pool boot thread also exits fast.
        """
        # Coerce "0" -> int so DirectShow works on Windows
        s = str(source).strip()
        source = int(s) if s.isdigit() else s
        backend = self._backend_flag()

        result = []   # [cap] on success, [None, exc] on failure

        def _do_open():
            try:
                cap = (cv2.VideoCapture(source, backend)
                       if backend else cv2.VideoCapture(source))
                result.append(cap)
            except Exception as exc:
                result.extend([None, exc])

        t = threading.Thread(target=_do_open, daemon=True, name="cap-open")
        t.start()
        t.join(timeout=timeout_sec)

        if t.is_alive():
            # Thread still blocked inside VideoCapture() — abandon it
            raise RuntimeError(
                f"Timed out after {timeout_sec}s opening source: {source}"
            )

        if not result or result[0] is None:
            exc = result[1] if len(result) > 1 else Exception("unknown error")
            raise RuntimeError(f"Failed to open camera source: {source} — {exc}")

        cap = result[0]
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"VideoCapture not opened for source: {source}")

        # Reduce internal buffer to cut latency on local webcams
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        # Warm-up reads only for local integer sources (URL streams can block)
        if isinstance(source, int):
            for _ in range(3):
                cap.read()

        self.cap = cap

    def _register_camera(self) -> int | None:
        """
        Ensure a Camera row exists for self.camera_source and open a new
        CameraSession.  Returns the Camera.id or None on failure.
        """
        try:
            from datetime import datetime
            source_str = str(self.camera_source)
            db = SessionLocal()
            try:
                cam = db.query(Camera).filter(Camera.source == source_str).first()
                if cam is None:
                    cam = Camera(
                        name=f"Camera {source_str}",
                        source=source_str,
                        is_active=True,
                    )
                    db.add(cam)
                    db.flush()

                session = CameraSession(camera_id=cam.id, started_at=datetime.utcnow())
                db.add(session)
                db.commit()

                self._camera_db_id  = cam.id
                self._session_db_id = session.id
                return cam.id
            finally:
                db.close()
        except Exception as exc:
            print(f"[CameraManager] Could not register camera in DB: {exc}")
            return None

    def _close_camera_session(self):
        """Mark the running CameraSession as ended."""
        if not getattr(self, "_session_db_id", None):
            return
        try:
            from datetime import datetime
            db = SessionLocal()
            try:
                s = db.query(CameraSession).filter(
                    CameraSession.id == self._session_db_id
                ).first()
                if s and s.ended_at is None:
                    s.ended_at = datetime.utcnow()
                    db.commit()
            finally:
                db.close()
        except Exception as exc:
            print(f"[CameraManager] Could not close camera session: {exc}")

    def stop(self):
        """Call on application shutdown."""
        self._stop_event.set()
        if self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)

        self._close_camera_session()

        with self._cap_lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None

    def set_source(self, source):
        """Switch camera source safely while the capture thread is running."""
        new_source = int(source) if str(source).isdigit() else source

        self._close_camera_session()

        with self._cap_lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None

            self.camera_source = new_source
            self._open_capture(self.camera_source)

        self._register_camera()

    def trigger_registration(self, name: str):
        self.register_name = name
        self.register_next_face = True

    def _capture_loop(self):
        """
        Dedicated capture thread — runs independently of ML processing.

        Continuously drains the OpenCV buffer so _raw_frame always holds
        the most recent camera frame.  ML (YOLO, embedding) runs in the
        separate _update_frame thread and simply reads _raw_frame whenever
        it finishes the previous inference — it may skip frames, but it
        will never process a stale one.

        Buffer-drain strategy:
          grab() is cheap (no decode).  We call it in a tight loop until
          it blocks (>8 ms), which means the buffer is empty and this IS
          the live edge.  Then we decode only that last frame.
        """
        fail_count = 0
        while not self._stop_event.is_set():
            with self._cap_lock:
                cap = self.cap

            if cap is None:
                time.sleep(0.05)
                continue

            # Drain stale frames, keep only the live edge
            last_ok  = False
            grab_count = 0
            while True:
                t0  = time.perf_counter()
                ok  = cap.grab()
                dt  = time.perf_counter() - t0
                if not ok:
                    break
                grab_count += 1
                # grab blocked → buffer was empty → this is the live frame
                if dt > 0.008 or grab_count >= 30:
                    last_ok = True
                    break

            if last_ok:
                ret, frame = cap.retrieve()
                if ret and frame is not None:
                    fail_count = 0
                    with self._raw_frame_lock:
                        self._raw_frame = frame
                    continue   # immediately grab the next live frame

            # grab failed — camera stalled
            fail_count += 1
            time.sleep(0.05)

            if fail_count >= 10:
                print("[CameraManager] Capture stalled, reopening…")
                try:
                    with self._cap_lock:
                        if self.cap is not None:
                            self.cap.release()
                            self.cap = None
                        self._open_capture(self.camera_source)
                    fail_count = 0
                except Exception:
                    pass

    def _update_frame(self):
        """
        ML processing thread.  Reads the latest raw frame from _raw_frame
        (written by _capture_loop) and runs detection / recognition on it.
        Skips frames if ML is slower than the camera — that is intentional:
        we always process the NEWEST frame available, never a queued-up one.
        """
        while not self._stop_event.is_set():
            t0 = time.time()

            # Get the latest captured frame (non-blocking)
            with self._raw_frame_lock:
                frame = self._raw_frame
                self._raw_frame = None   # clear so we don't reprocess same frame

            if frame is None:
                # Capture thread hasn't produced a frame yet — wait briefly
                time.sleep(0.005)
                continue

            # Optional mirror flip
            if self.flip_horizontal:
                frame = cv2.flip(frame, 1)

            # Resize frame before detection + display.
            # Keeps detection circles proportional to what the browser shows,
            # and makes YOLO significantly faster on high-res phone streams.
            pw = int(self.process_width)
            if pw > 0 and frame.shape[1] != pw:
                scale  = pw / frame.shape[1]
                ph     = int(frame.shape[0] * scale)
                frame  = cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_LINEAR)

            # Apply ML logic based on toggles
            if self.run_detection or self.run_recognition or self.register_next_face or self.anchor_recording or self.anchor_frozen:
                # ── detection timing ──────────────────────────────────────
                _t_det0 = time.perf_counter()
                detections = self._iter_detections(frame)
                self._det_times.append(time.perf_counter() - _t_det0)

                visible = []


                frame_log_entries = []
                registered_this_frame = False

                for bbox, cropped_face in detections:
                    if bbox is None or cropped_face is None or getattr(cropped_face, "size", 0) == 0:
                        continue
                    visible.append((bbox, cropped_face))
                    x1, y1, x2, y2 = bbox

                    name = None
                    similarity = float("nan")
                    status = "detected"

                    # Recognition path (your script)
                    if self.run_recognition:
                        _t_rec0 = time.perf_counter()
                        embedding = self._embed_from_crop(cropped_face)
                        if embedding is not None:
                            # Use the same matching settings as your script
                            name, similarity = self.db.find_closest(
                                embedding,
                                backend="faiss",
                                index_type="hnsw",
                                aggregation="topk",
                                top_k=10,
                                faiss_k=120,
                                threshold=self.similarity_threshold,
                            )
                            # DEBUG: print threshold + raw score every ~5 sec
                            _now_dbg = time.time()
                            if not hasattr(self, "_last_dbg") or _now_dbg - self._last_dbg > 5:
                                self._last_dbg = _now_dbg
                                print(f"[DEBUG cam {self.camera_source}] "
                                      f"threshold={self.similarity_threshold:.3f}  "
                                      f"best_score={similarity:.3f}  "
                                      f"result={'PASS: '+str(name) if name else 'BLOCKED'}", flush=True)
                        self._rec_times.append(time.perf_counter() - _t_rec0)

                        if name is not None:
                            display_text = f"{name} ({similarity:.2f})"
                            color = getattr(config, "COLOR_DETECTION", (0, 255, 0))
                            status = "recognized"
                            log_name = name
                            log_sim = float(similarity)
                        else:
                            display_text = "Unknown"
                            color = getattr(config, "COLOR_UNKNOWN", (0, 0, 255))
                            status = "unrecognized"
                            log_name = "Unknown"
                            log_sim = float("nan")
                    else:
                        # Detection-only path
                        display_text = "Detected"
                        color = getattr(config, "COLOR_DETECTION", (0, 255, 0))
                        log_name = "Detected"
                        log_sim = float("nan")

                    # Draw (use your circle style)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    r = max((x2 - x1), (y2 - y1)) // 2
                    cv2.circle(frame, (cx, cy), r, color, 1, lineType=cv2.LINE_AA)
                    cv2.putText(frame, display_text, (x1, max(0, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1)

                    # Buffer logging for THIS frame (all detections)
                    frame_log_entries.append((log_name, log_sim, status))

                    # Registration: save the next detected face once
                    if self.register_next_face and not registered_this_frame:
                        embedding = self._embed_from_crop(cropped_face)
                        if embedding is not None:
                            # You must adapt this to your FaceDatabase insert API:
                            # examples: add_person(name, emb), add_embedding(name, emb), insert(name, emb), etc.
                            try:
                                if hasattr(self.db, "add_person"):
                                    self.db.add_person(self.register_name, embedding)
                                elif hasattr(self.db, "add_embedding"):
                                    self.db.add_embedding(self.register_name, embedding)
                                elif hasattr(self.db, "insert"):
                                    self.db.insert(self.register_name, embedding)
                                else:
                                    # last resort: raise so you notice it during testing
                                    raise AttributeError(
                                        "No known insert method on FaceDatabase (add_person/add_embedding/insert).")
                                registered_this_frame = True
                                self.register_next_face = False
                                cv2.putText(frame, f"Registered {self.register_name}!", (50, 50),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            except Exception as e:
                                cv2.putText(frame, f"Register failed: {type(e).__name__}", (50, 50),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                # DB logging via FrameProcessor (respects cooldown + saves crops)
                now = time.time()
                if (now - self._last_log_time) > self.log_interval_sec and frame_log_entries:
                    self._log_detections_to_db(frame, visible, frame_log_entries)
                    self._last_log_time = now

                # Keep latest visible list for anchor features
                with self._anchor_lock:
                    if not self.anchor_frozen:
                        self.anchor_visible = visible

                # If recording: capture anchors periodically from LIVE (not frozen)
                with self._anchor_lock:
                    recording = self.anchor_recording and (not self.anchor_frozen)
                    interval = self.anchor_sample_interval_sec
                    last_t = self._anchor_last_sample_time

                if recording:
                    now = time.time()
                    if now - last_t >= interval and len(visible) > 0:
                        best_bbox, best_crop = max(visible, key=lambda bc: _bbox_area(bc[0]))
                        emb = self._embed_from_crop(best_crop)
                        if emb is not None:
                            with self._anchor_lock:
                                self.anchor_samples.append(emb)
                                self.anchor_last_mark_bbox = best_bbox
                                self.anchor_last_mark_frames_left = 10
                                self._anchor_last_sample_time = now
                with self._anchor_lock:
                    if (
                    not self.anchor_frozen) and self.anchor_last_mark_frames_left > 0 and self.anchor_last_mark_bbox is not None:
                        x1, y1, x2, y2 = self.anchor_last_mark_bbox
                        color = getattr(config, "COLOR_CAPTURING", (0, 255, 255))
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        r = max((x2 - x1), (y2 - y1)) // 2
                        cv2.circle(frame, (cx, cy), r, color, 1, lineType=cv2.LINE_AA)
                        cv2.putText(frame, "CAPTURED", (x1, max(0, y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)
                        self.anchor_last_mark_frames_left -= 1

            with self._frame_lock:
                self.current_frame = frame  # keep latest

            self._frame_times.append(time.time() - t0)

            # No artificial sleep — ML processing time is the natural pacer.
            # Sleeping here would cause the next frame read to be even more stale.
            # target_dt is kept for reference but not used for sleeping.
            _ = self.target_dt

    def get_metrics(self) -> dict:
        """Return latency/fps metrics as a dict (thread-safe snapshot)."""
        def _stats(d: deque):
            if not d:
                return {"fps": None, "ms": None}
            arr = list(d)
            avg = sum(arr) / len(arr)
            return {
                "fps": round(1.0 / avg, 1) if avg > 0 else None,
                "ms":  round(avg * 1000, 1),
            }
        return {
            "detection":  _stats(self._det_times),
            "recognition": _stats(self._rec_times),
            "frame":      _stats(self._frame_times),
            "batch_size": round(self._embedder.avg_batch_size, 2),
        }

    def get_frame_bytes(self):
        # Copy the frame quickly under lock; encode outside lock
        with self._frame_lock:
            frame = None if self.current_frame is None else self.current_frame.copy()

        if frame is None:
            return None

        # Optional downscale for lower bandwidth / faster delivery
        w, h = int(self.stream_width), int(self.stream_height)
        if w > 0 and h > 0:
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
        elif w > 0:
            scale = w / frame.shape[1]
            frame = cv2.resize(frame, (w, int(frame.shape[0] * scale)),
                               interpolation=cv2.INTER_LINEAR)

        quality = max(1, min(95, int(self.jpeg_quality)))
        ok, buffer = cv2.imencode(".jpg", frame,
                                  [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buffer.tobytes() if ok else None

    def _iter_detections(self, frame):
        """
        Normalize detector output to an iterable of (bbox, cropped_face).
        Supports:
          - single: (bbox, cropped_face)
          - multi:  [(bbox, cropped_face), ...]
        """
        out = self.detector.detect_and_crop(frame)

        if out is None:
            return []
        # multi
        if isinstance(out, (list, tuple)) and len(out) > 0 and isinstance(out[0], (list, tuple)) and len(out[0]) == 2:
            # could be [(bbox, crop), ...] OR (bbox, crop) (ambiguous), so detect by first element shape
            # if first element is bbox-like, then it's single; if first element is a pair, it's multi
            if out and isinstance(out[0][0], (list, tuple, np.ndarray)):
                return out
        # single
        if isinstance(out, (list, tuple)) and len(out) == 2:
            return [out]
        return []

    def _embed_from_crop(self, cropped_face):
        """Submit one crop to the shared batch embedder (blocking, ≤ ~8 ms wait)."""
        if cropped_face is None or getattr(cropped_face, "size", 0) == 0:
            return None

        crop_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
        pil_img  = Image.fromarray(crop_rgb)
        tensor   = self.transform(pil_img)   # (C, H, W) — embedder adds batch dim

        return self._embedder.embed_sync(tensor)

    def _log_detections_to_db(self, frame, visible, frame_log_entries):
        """Write recognized / unknown detections directly to the SQL DB.

        Uses its own cooldown dict (_log_cooldowns) so we don't need to
        instantiate a heavy FrameProcessor (which would load another model).
        """
        import logging as _log
        _logger = _log.getLogger(__name__)

        # Lazy-init cooldown tracker {(name, cam_src): last_logged_ts}
        if not hasattr(self, "_log_cooldowns"):
            self._log_cooldowns: dict = {}
        COOLDOWN = 60.0  # seconds between logging the same person on this camera

        cam_id_str = str(self.camera_source)
        now_ts = time.time()

        try:
            from database.session import SessionLocal
            from database.models import Camera, Detection, UnknownFace, User
        except Exception as exc:
            _logger.debug("DB import failed in cam logging: %s", exc)
            return

        try:
            from api.frame_processor import _save_crop
        except ImportError:
            try:
                from frame_processor import _save_crop
            except ImportError:
                _save_crop = lambda crop, label, cam_name: None  # no photo saving

        for (log_name, log_sim, status), (bbox, crop) in zip(frame_log_entries, visible):
            if status not in ("recognized", "unrecognized", "unknown"):
                continue
            if crop is None or getattr(crop, "size", 0) == 0:
                continue

            key = (log_name, cam_id_str)
            if now_ts - self._log_cooldowns.get(key, 0) < COOLDOWN:
                continue

            db = None
            try:
                db = SessionLocal()
                cam_row   = db.query(Camera).filter(
                    Camera.id == self._camera_db_id).first() if self._camera_db_id else None
                cam_name  = cam_row.name if cam_row else cam_id_str
                cam_db_id = cam_row.id   if cam_row else None

                if status == "recognized":
                    img_path = _save_crop(crop, log_name, cam_name)
                    # log_name is the vector-DB key, which is always the username
                    user_row = db.query(User).filter(User.username == log_name).first()
                    # Store display_name in Detection.name for human-readable logs
                    shown_name = (user_row.display_name or user_row.username) if user_row else log_name
                    db.add(Detection(
                        user_id      = user_row.id if user_row else None,
                        camera_db_id = cam_db_id,
                        name         = shown_name,  # cosmetic only — queries use user_id
                        camera_id    = cam_id_str,
                        confidence   = float(log_sim) if log_sim == log_sim else None,
                        image_path   = img_path,
                    ))
                else:  # unknown
                    img_path = _save_crop(crop, "unknown", cam_name)
                    db.add(UnknownFace(
                        camera_db_id = cam_db_id,
                        camera_id    = cam_id_str,
                        image_path   = img_path,
                    ))

                db.commit()
                self._log_cooldowns[key] = now_ts
            except Exception as exc:
                if db:
                    db.rollback()
                _logger.debug("cam db log error: %s", exc)
            finally:
                if db:
                    db.close()

    # --- Anchor workflow control (API callable) ---

    def anchors_status(self) -> dict:
        with self._anchor_lock:
            return {
                "frozen": self.anchor_frozen,
                "recording": self.anchor_recording,
                "samples": len(self.anchor_samples),
                "interval_sec": self.anchor_sample_interval_sec,
            }

    def anchors_freeze(self):
        """Freeze the *next* available live detections' snapshot."""
        # We'll implement freeze by grabbing the most recent live detections+frame
        with self._anchor_lock:
            if self.current_frame is None:
                raise RuntimeError("No frame available to freeze yet.")

        # Create a snapshot using latest frame + live visible list (stored by update loop)
        with self._frame_lock:
            snap = None if self.current_frame is None else self.current_frame.copy()

        with self._anchor_lock:
            if snap is None:
                raise RuntimeError("No frame available to freeze yet.")
            if len(self.anchor_visible) == 0:
                raise RuntimeError("No detections available to freeze.")

            self.anchor_frozen = True
            self.anchor_frozen_frame = snap
            self.anchor_frozen_visible = list(self.anchor_visible)

    def anchors_unfreeze(self):
        with self._anchor_lock:
            self.anchor_frozen = False
            self.anchor_frozen_frame = None
            self.anchor_frozen_visible = []

    def anchors_reset(self):
        with self._anchor_lock:
            self.anchor_samples = []
            self.anchor_last_mark_bbox = None
            self.anchor_last_mark_frames_left = 0

    def anchors_start_recording(self, interval_sec: float = 0.2):
        with self._anchor_lock:
            self.anchor_sample_interval_sec = max(0.05, float(interval_sec))
            self.anchor_recording = True
            self._anchor_last_sample_time = 0.0

    def anchors_stop_recording(self):
        with self._anchor_lock:
            self.anchor_recording = False

    def anchors_save(self, name: str):
        name = (name or "").strip()
        if not name:
            raise ValueError("Name cannot be empty.")

        with self._anchor_lock:
            if len(self.anchor_samples) == 0:
                raise RuntimeError("No samples to save.")
            samples = list(self.anchor_samples)
            self.anchor_samples = []

        # Bulk insert anchors (like your script)
        if hasattr(self.db, "add_person_many"):
            self.db.add_person_many(name, samples)
        else:
            # fallback: insert one by one if you don't have add_person_many
            if hasattr(self.db, "add_person"):
                for emb in samples:
                    self.db.add_person(name, emb)
            else:
                raise AttributeError("FaceDatabase missing add_person_many (or add_person).")

        return {"saved_name": name, "anchors_saved": len(samples)}

    def anchors_click_capture(self, x: int, y: int):
        """
        Like your script:
        - requires frozen mode
        - choose smallest bbox containing point
        - capture ONE anchor into buffer
        - auto-unfreeze
        """
        with self._anchor_lock:
            if not self.anchor_frozen:
                raise RuntimeError("Not frozen. Call /anchors/freeze first.")
            visible = list(self.anchor_frozen_visible)

        candidates = []
        for i, (bbox, crop) in enumerate(visible):
            if bbox is None or crop is None:
                continue
            if _point_in_bbox(x, y, bbox):
                candidates.append((_bbox_area(bbox), i))

        if not candidates:
            raise RuntimeError("Click not inside any face bbox.")

        candidates.sort(key=lambda t: t[0])
        idx = candidates[0][1]
        bbox, crop = visible[idx]

        emb = self._embed_from_crop(crop)  # uses your existing helper
        if emb is None:
            raise RuntimeError("Failed to embed clicked crop.")

        with self._anchor_lock:
            self.anchor_samples.append(emb)
            self.anchor_last_mark_bbox = bbox
            self.anchor_last_mark_frames_left = 25

            # auto-unfreeze like script
            self.anchor_frozen = False
            self.anchor_frozen_frame = None
            self.anchor_frozen_visible = []

        return {"captured": 1, "total_samples": self.anchors_status()["samples"], "selected_index": idx}