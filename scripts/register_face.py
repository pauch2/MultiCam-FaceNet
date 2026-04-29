import cv2
import torch
import torchvision.transforms as T
from PIL import Image
import sys
import os
import numpy as np
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.detector import FaceDetector, _open_capture
from models.embedding_model import FaceEmbeddingModel
from database.vector_store import FaceDatabase
import config

def bbox_area(bbox) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def point_in_bbox(x: int, y: int, bbox) -> bool:
    x1, y1, x2, y2 = bbox
    return (x1 <= x <= x2) and (y1 <= y <= y2)


def on_mouse(event, x, y, flags, param):
    """
    Manual freeze mode:
      - Press 'f' -> frozen snapshot
      - Click -> captures ONE anchor sample from frozen crop, then auto-unfreeze
      - Overlap handling: choose smallest bbox containing the click
    """
    state = param
    if not state["frozen"]:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        candidates = []
        for i, (bbox, crop) in enumerate(state["frozen_visible"]):
            if bbox is None or crop is None:
                continue
            if point_in_bbox(x, y, bbox):
                candidates.append((bbox_area(bbox), i))  # smallest area wins

        if not candidates:
            return

        candidates.sort(key=lambda t: t[0])
        idx = candidates[0][1]
        bbox, crop = state["frozen_visible"][idx]

        if crop is None or crop.size == 0:
            print("Clicked crop invalid. Try again.")
            return

        emb = state["crop_to_embedding"](crop)
        state["samples"].append(emb)

        state["last_mark_bbox"] = bbox
        state["last_mark_frames_left"] = 25

        print(f"[MANUAL] Captured anchor #{len(state['samples'])} (clicked #{idx}).")

        # auto-unfreeze
        state["frozen"] = False
        state["frozen_frame"] = None
        state["frozen_visible"] = []


def main():
    detector = FaceDetector()

    model = FaceEmbeddingModel(embedding_dim=config.EMBEDDING_DIM)
    if os.path.exists(config.MODEL_WEIGHTS_PATH):
        model.load_state_dict(torch.load(config.MODEL_WEIGHTS_PATH, map_location="cpu"))
    else:
        print("Warning: model weights not found; using current weights.")
    model.eval()

    db = FaceDatabase()

    transform = T.Compose([
        T.Resize(config.IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def crop_to_embedding(cropped_face) -> np.ndarray:
        crop_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)
        tensor_img = transform(pil_img)
        return model.get_embedding(tensor_img)  # numpy

    cap = _open_capture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0).")

    print("\n=== Anchor Registration (manual + video) ===")
    print("Manual anchors:")
    print("  f: freeze video+boxes, click to capture ONE anchor (auto-unfreeze)")
    print("Video anchors (~5/sec):")
    print("  s: start recording anchors from LIVE video (captures largest face)")
    print("  y: stop recording and SAVE anchors under a name")
    print("General:")
    print("  r: reset current samples buffer")
    print("  n: cancel freeze")
    print("  q: quit\n")

    state = {
        # manual freeze selection
        "frozen": False,
        "frozen_frame": None,
        "visible": [],
        "frozen_visible": [],

        # video recording
        "recording": False,
        "sample_interval_sec": 0.1,   # 5 per second
        "last_sample_time": 0.0,

        # samples buffer (anchors)
        "samples": [],

        # embedding
        "crop_to_embedding": crop_to_embedding,

        # UI feedback
        "last_mark_bbox": None,
        "last_mark_frames_left": 0,
    }

    window_name = "Register Anchors"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse, state)

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if not ret:
            break

        # If not frozen, update detections from live frame
        if not state["frozen"]:
            detections = detector.detect_and_crop(frame)
            visible = []
            for bbox, crop in detections:
                if bbox is None or crop is None or crop.size == 0:
                    continue
                visible.append((bbox, crop))
            state["visible"] = visible

            display_frame = frame.copy()
            draw_list = state["visible"]
        else:
            display_frame = state["frozen_frame"].copy()
            draw_list = state["frozen_visible"]

        # Draw boxes (green)
        for i, (bbox, _) in enumerate(draw_list):
            x1, y1, x2, y2 = bbox
            color = config.COLOR_DETECTION
            # cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(display_frame, ((x2 + x1) // 2, (y2 + y1) // 2), (max((x2 - x1), (y2 - y1)) // 2), color, 1,
                       lineType=cv2.LINE_AA)
            cv2.putText(display_frame, f"Detected face #{i}", (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)

        # Highlight last captured bbox briefly (yellow)
        if (not state["frozen"]) and state["last_mark_frames_left"] > 0 and state["last_mark_bbox"] is not None:
            x1, y1, x2, y2 = state["last_mark_bbox"]
            color = config.COLOR_CAPTURING
            # cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
            cv2.circle(display_frame, ((x2 + x1) // 2, (y2 + y1) // 2), (max((x2 - x1), (y2 - y1)) // 2), color, 1,
                       lineType=cv2.LINE_AA)
            cv2.putText(display_frame, "CAPTURED", (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)
            state["last_mark_frames_left"] -= 1

        # If recording: capture anchors every interval from LIVE (not frozen)
        if state["recording"] and (not state["frozen"]):
            now = time.time()
            if now - state["last_sample_time"] >= state["sample_interval_sec"]:
                state["last_sample_time"] = now

                if len(state["visible"]) > 0:
                    # capture the largest face (practical w/o tracking)
                    best_bbox, best_crop = max(state["visible"], key=lambda bc: bbox_area(bc[0]))

                    emb = crop_to_embedding(best_crop)
                    state["samples"].append(emb)

                    state["last_mark_bbox"] = best_bbox
                    state["last_mark_frames_left"] = 10

        # HUD
        hud1 = f"Samples: {len(state['samples'])} | Recording: {state['recording']} (s start, y stop+save)"
        hud2 = "Manual: f freeze+click capture | r reset | n cancel freeze | q quit"
        if state["frozen"]:
            hud3 = "FROZEN: click a face to capture ONE anchor (auto-unfreeze)"
        else:
            hud3 = "LIVE"

        cv2.putText(display_frame, hud1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display_frame, hud2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(display_frame, hud3, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF

        # Quit
        if key == ord("q"):
            break

        # Manual freeze snapshot
        if key == ord("f"):
            if len(state["visible"]) == 0:
                print("No valid detections to select.")
                continue
            state["frozen"] = True
            state["frozen_frame"] = frame.copy()
            state["frozen_visible"] = list(state["visible"])
            print("FROZEN (manual). Click a face to capture one anchor.")
            continue

        # Cancel freeze
        if key == ord("n"):
            if state["frozen"]:
                state["frozen"] = False
                state["frozen_frame"] = None
                state["frozen_visible"] = []
                print("Freeze cancelled.")
            continue

        # Reset samples buffer
        if key == ord("r"):
            state["samples"] = []
            print("Samples reset.")
            continue

        # Start recording (video mode)
        if key == ord("s"):
            if state["recording"]:
                print("Already recording.")
                continue
            state["recording"] = True
            state["last_sample_time"] = 0.0
            print("Recording started (capturing largest face). Press 'y' to stop+save.")
            continue

        # Stop recording and SAVE (or just save current samples)
        if key == ord("y"):
            if state["recording"]:
                state["recording"] = False
                print("Recording stopped.")

            if len(state["samples"]) == 0:
                print("No samples to save.")
                continue

            name = input(f"Enter name to register ({len(state['samples'])} anchors): ").strip()
            if not name:
                print("Name cannot be empty.")
                continue

            # Bulk insert anchors (fast)
            db.add_person_many(name, state["samples"])
            print(f"Saved {name} with {len(state['samples'])} anchors.")

            # Clear buffer
            state["samples"] = []
            continue

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()