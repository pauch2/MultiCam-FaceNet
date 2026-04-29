import platform
import cv2
from ultralytics import YOLO
import config


def _open_capture(source: int) -> cv2.VideoCapture:
    """
    Open a VideoCapture with the correct backend for the current OS.

    Windows: use DirectShow (CAP_DSHOW).
      - MSMF (the default) produces "OnReadSample error -1072875772" on many
        webcams and simply never returns frames.  CAP_DSHOW is the fix.
      - Do NOT call cap.set(CAP_PROP_BUFFERSIZE) — that corrupts the frame
        buffer on some cameras and causes vertical-stripe artifacts.

    Linux / macOS: use the default backend (V4L2 / AVFoundation).

    A 5-frame warm-up discards the first frames while auto-exposure settles.
    """
    if platform.system() == "Windows":
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        return cap

    # Warm up — discard frames while the sensor stabilises.
    # Do NOT set CAP_PROP_BUFFERSIZE; leave the buffer at its default size.
    for _ in range(5):
        cap.read()

    return cap


class FaceDetector:
    def __init__(self, model_path: str = config.YOLO_MODEL_PATH):
        try:
            self.model = YOLO(model_path, task="detect")
        except Exception:
            print(f"Model '{model_path}' not found, falling back to yolo11n.pt")
            self.model = YOLO("yolo11n.pt", task="detect")

    def detect_and_crop(self, frame) -> list[tuple]:
        """
        Returns list of ((x1, y1, x2, y2), cropped_bgr) tuples.
        """
        results = self.model(frame, classes=[1, 2], verbose=False)

        h, w = frame.shape[:2]
        detections = []

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return detections

        for (x1, y1, x2, y2) in r.boxes.xyxy.cpu().numpy().astype(int):
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(((x1, y1, x2, y2), frame[y1:y2, x1:x2]))

        return detections