"""
frame_processor.py

Stateless per-frame helpers used by both the server camera pipeline and the
client-camera WebSocket.  The class is instantiated once at startup and shared
across connections.

Public API
──────────
process(frame, run_recognition, camera_id)
    Full pipeline: detect → recognise → log to SQLAlchemy DB.
    Returns list[dict] of detections for JSON serialisation.

detect_only(frame)
    Run the YOLO detector and return list[(bbox, crop_bgr)].
    Used by client_cam.py in register mode.

embed(crop_bgr)
    Embed a single BGR crop. Returns np.ndarray or None.
    Used by client_cam.py in register mode.
"""

import logging
import os
import time
from datetime import datetime

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from sqlalchemy.orm import Session

import config
from database.models import Camera, Detection, UnknownFace, User
from database.session import SessionLocal
from api.shared_db import face_db
from models.detector import FaceDetector
from models.embedding_model import load_embedding_model

log = logging.getLogger(__name__)

# Root directory for saved detection images — always absolute, anchored to project root
try:
    import config as _cfg
    _IMG_ROOT = os.path.join(_cfg.BASE_DIR, "storage", "detections")
except Exception:
    _IMG_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "storage", "detections")


def _save_crop(crop_bgr: np.ndarray, label: str, camera_name: str) -> str | None:
    """
    Save a face crop to disk.
    Path:  <project>/storage/detections/<camera_name>/<YYYY-MM-DD>/<label>_<ms>.jpg
    Returns the ABSOLUTE path string, or None on failure.
    """
    try:
        today    = datetime.utcnow().strftime("%Y-%m-%d")
        safe_cam = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in camera_name)
        safe_lbl = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in label)
        folder   = os.path.join(_IMG_ROOT, safe_cam, today)
        os.makedirs(folder, exist_ok=True)
        ms       = int(time.time() * 1000)
        path     = os.path.join(folder, f"{safe_lbl}_{ms}.jpg")
        cv2.imwrite(path, crop_bgr)
        return path   # absolute path
    except Exception as exc:
        log.warning("Could not save crop: %s", exc)
        return None


class FrameProcessor:
    def __init__(self):
        self.detector = FaceDetector()
        self.db       = face_db   # shared singleton — same cache as client_cam.py

        # Cooldown keyed by (name, camera_source) — 60 seconds
        # Prevents spamming the log table for the same person on the same camera.
        self._last_logged: dict[tuple[str, str], float] = {}
        self._log_cooldown_sec: float = 60.0

        # Embedding model
        device_str  = getattr(config, "DEVICE", None)
        self.device = torch.device(device_str) if device_str else \
                      torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = load_embedding_model(device=str(self.device))

        input_size = getattr(config, "INPUT_SIZE",  224)
        mean       = getattr(config, "NORM_MEAN",   (0.485, 0.456, 0.406))
        std        = getattr(config, "NORM_STD",    (0.229, 0.224, 0.225))

        self.transform = T.Compose([
            T.Resize((input_size, input_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    # ── Public API ────────────────────────────────────────────────────────────

    def detect_only(self, frame_bgr: np.ndarray) -> list[tuple]:
        """Run YOLO detection; return list[(bbox, crop_bgr)]."""
        return self._iter_detections(frame_bgr)

    @torch.no_grad()
    def embed(self, crop_bgr: np.ndarray) -> np.ndarray | None:
        """Embed a single BGR crop. Public wrapper for client_cam.py."""
        return self._embed_from_crop(crop_bgr)

    def process(self, frame_bgr: np.ndarray,
                run_recognition: bool = True,
                camera_id: str = "client") -> list[dict]:
        """
        Full pipeline: detect → (optionally recognise) → log.
        camera_id: the camera source string ("0", "rtsp://…", etc.) or a label.
        Returns a JSON-serialisable list of detection dicts.
        """
        results: list[dict] = []
        db: Session = SessionLocal()

        try:
            # Resolve camera DB row (for FK + name label used in file paths)
            cam_row  = db.query(Camera).filter(Camera.source == camera_id).first()
            cam_db_id   = cam_row.id   if cam_row else None
            cam_name    = cam_row.name if cam_row else camera_id

            for bbox, crop in self._iter_detections(frame_bgr):
                if bbox is None or crop is None:
                    continue

                item: dict = {
                    "bbox":       [int(x) for x in bbox],
                    "name":       None,
                    "similarity": None,
                    "status":     "detected",
                }

                if run_recognition:
                    emb = self._embed_from_crop(crop)
                    if emb is not None:
                        name, sim = self.db.find_closest(
                            emb,
                            backend="faiss",
                            index_type="hnsw",
                            aggregation="topk",
                            top_k=10,
                            faiss_k=120,
                        )
                        if name is not None:
                            item.update({"name": str(name), "similarity": float(sim),
                                         "status": "recognized"})
                            self._log_detection(db, item, crop, camera_id,
                                                cam_db_id, cam_name)
                        else:
                            item.update({"name": "Unknown", "status": "unknown"})
                            self._log_unknown(db, crop, camera_id, cam_db_id, cam_name)

                results.append(item)

            db.commit()
        except Exception as exc:
            log.error("DB logging error: %s", exc)
            db.rollback()
        finally:
            db.close()

        return results

    # ── Private helpers ───────────────────────────────────────────────────────

    def _iter_detections(self, frame_bgr: np.ndarray) -> list[tuple]:
        """Normalise detector output to list[(bbox, crop_bgr)]."""
        out = self.detector.detect_and_crop(frame_bgr)
        if not out:
            return []
        if isinstance(out, list):
            return out
        if isinstance(out, tuple) and len(out) == 2:
            return [out]
        return []

    @torch.no_grad()
    def _embed_from_crop(self, crop_bgr: np.ndarray) -> np.ndarray | None:
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        try:
            rgb    = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            tensor = self.transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
            emb    = self.model.get_embedding(tensor)
            if isinstance(emb, torch.Tensor):
                emb = emb.cpu().numpy()
            if emb.ndim == 2 and emb.shape[0] == 1:
                emb = emb[0]
            return emb
        except Exception as exc:
            log.warning("Embedding failed: %s", exc)
            return None

    def _on_cooldown(self, key: tuple[str, str]) -> bool:
        """Return True if this (name, camera) was logged within the cooldown window."""
        last = self._last_logged.get(key)
        return last is not None and (time.time() - last) < self._log_cooldown_sec

    def _log_detection(self, db: Session, item: dict, crop: np.ndarray,
                       camera_source: str, cam_db_id: int | None, cam_name: str):
        name = item["name"]
        key  = (name, camera_source)
        if self._on_cooldown(key):
            return

        img_path = _save_crop(crop, name, cam_name)

        user = db.query(User).filter(User.username == name).first()
        db.add(Detection(
            user_id      = user.id if user else None,
            camera_db_id = cam_db_id,
            name         = name,
            camera_id    = camera_source,
            confidence   = item["similarity"],
            image_path   = img_path,
            timestamp    = datetime.utcnow(),
        ))
        self._last_logged[key] = time.time()

    def _log_unknown(self, db: Session, crop: np.ndarray,
                     camera_source: str, cam_db_id: int | None, cam_name: str):
        key = ("Unknown", camera_source)
        if self._on_cooldown(key):
            return

        img_path = _save_crop(crop, "unknown", cam_name)
        db.add(UnknownFace(
            camera_db_id = cam_db_id,
            camera_id    = camera_source,
            image_path   = img_path,
            timestamp    = datetime.utcnow(),
        ))
        self._last_logged[key] = time.time()