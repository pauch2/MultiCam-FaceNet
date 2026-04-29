"""
camera_pool.py

Manages multiple simultaneous CameraManager instances, one per camera DB row.
Replaces the single app.state.camera pattern.

Usage (from routers):
    pool: CameraPool = request.app.state.pool
    pool.start(cam_db_id, source, run_recognition)
    pool.stop(cam_db_id)
    pool.get_frame_bytes(cam_db_id) -> bytes | None
    pool.active_ids() -> list[int]
"""

import threading
from typing import Optional
try:
    from api.camera_stream import CameraManager   # package layout
except ImportError:
    from camera_stream import CameraManager        # flat layout


class CameraPool:
    def __init__(self):
        self._lock    = threading.Lock()
        self._cameras: dict[int, CameraManager] = {}   # cam_db_id -> CameraManager

    # ── Start / stop ──────────────────────────────────────────────────────────

    @staticmethod
    def _coerce_source(source: str):
        """'0' -> int(0) so OpenCV VideoCapture works on Windows."""
        s = str(source).strip()
        return int(s) if s.isdigit() else s

    def start(self, cam_db_id: int, source: str,
              run_recognition: bool = True) -> "CameraManager":
        """Start streaming from a camera. Replaces any existing stream for that id.

        CameraManager.__init__ calls _open_capture which may block for several
        seconds on RTSP sources.  We stop the old manager first (quick), then
        build the new one in a daemon thread so the API call returns immediately.
        The caller can poll /cameras/ to see when streaming=True.
        """
        # Stop old manager synchronously (it's already running, stop is fast)
        with self._lock:
            old = self._cameras.pop(cam_db_id, None)
        if old is not None:
            try:
                old.stop()
            except Exception:
                pass

        # Insert sentinel (None) immediately so is_streaming() / active_ids()
        # return True while the background thread is still opening the capture.
        with self._lock:
            self._cameras[cam_db_id] = None  # type: ignore[assignment]

        def _boot():
            try:
                mgr = CameraManager(source=self._coerce_source(source))
                mgr.run_detection   = True
                mgr.run_recognition = run_recognition
                with self._lock:
                    self._cameras[cam_db_id] = mgr
            except Exception as exc:
                print(f"[CameraPool] failed to start cam {cam_db_id}: {exc}")
                with self._lock:
                    self._cameras.pop(cam_db_id, None)  # remove sentinel on failure

        t = threading.Thread(target=_boot, daemon=True, name=f"cam-boot-{cam_db_id}")
        t.start()
        return None  # manager not yet available; pool.get() returns None briefly

    def stop(self, cam_db_id: int):
        with self._lock:
            mgr = self._cameras.pop(cam_db_id, None)
        if mgr is not None:
            mgr.stop()

    def stop_all(self):
        with self._lock:
            ids = list(self._cameras.keys())
        for cid in ids:
            self.stop(cid)

    # ── Query ─────────────────────────────────────────────────────────────────

    def get(self, cam_db_id: int) -> Optional[CameraManager]:
        with self._lock:
            mgr = self._cameras.get(cam_db_id)
        return mgr  # may be None (booting) or absent (not started)

    def active_ids(self) -> list[int]:
        with self._lock:
            return list(self._cameras.keys())

    def is_streaming(self, cam_db_id: int) -> bool:
        with self._lock:
            return cam_db_id in self._cameras

    def get_frame_bytes(self, cam_db_id: int) -> bytes | None:
        mgr = self.get(cam_db_id)
        return mgr.get_frame_bytes() if mgr is not None else None

    def set_recognition(self, cam_db_id: int, enabled: bool):
        mgr = self.get(cam_db_id)
        if mgr is not None:
            mgr.run_recognition = enabled

    def set_quality(self, cam_db_id: int, jpeg_quality: int = 70,
                    width: int = 0, height: int = 0,
                    process_width: int = 0):
        mgr = self.get(cam_db_id)
        if mgr is not None:
            mgr.jpeg_quality   = max(1, min(95, jpeg_quality))
            mgr.stream_width   = max(0, width)
            mgr.stream_height  = max(0, height)
            mgr.process_width  = max(0, process_width)

    def set_threshold(self, cam_db_id: int, threshold: float):
        mgr = self.get(cam_db_id)
        if mgr is not None:
            mgr.similarity_threshold = max(0.0, min(1.0, float(threshold)))

    def get_metrics(self, cam_db_id: int) -> dict:
        mgr = self.get(cam_db_id)
        if mgr is None:
            return {}
        return mgr.get_metrics()

    def is_booting(self, cam_db_id: int) -> bool:
        """True if start() was called but the manager isn't ready yet."""
        with self._lock:
            return cam_db_id in self._cameras and self._cameras[cam_db_id] is None

    def status(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "cam_db_id":            cid,
                    "run_recognition":      mgr.run_recognition       if mgr else False,
                    "run_detection":        mgr.run_detection         if mgr else False,
                    "booting":              mgr is None,
                    "similarity_threshold": mgr.similarity_threshold  if mgr else None,
                    "metrics":              mgr.get_metrics()          if mgr else {},
                }
                for cid, mgr in self._cameras.items()
            ]