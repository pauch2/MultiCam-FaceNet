import cv2
import torch
import torchvision.transforms as T
from PIL import Image
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.detector import FaceDetector, _open_capture
from models.embedding_model import FaceEmbeddingModel
from database.vector_store import FaceDatabase
from logs.logger import AccessLogger
import config


def main():
    detector = FaceDetector()
    logger   = AccessLogger()
    db       = FaceDatabase()

    model = FaceEmbeddingModel(embedding_dim=config.EMBEDDING_DIM)
    if os.path.exists(config.MODEL_WEIGHTS_PATH):
        ckpt = torch.load(config.MODEL_WEIGHTS_PATH, map_location="cpu")
        # Support both plain state_dict and full checkpoint saved by train.py
        model.load_state_dict(ckpt.get("model_state", ckpt))
    model.eval()

    transform = T.Compose([
        T.Resize(config.IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    cap = _open_capture(0)
    if not cap.isOpened():
        print("ERROR: could not open camera.")
        return

    print("Starting Recognition... Press 'q' to quit.")
    last_log_time = time.time()
    read_failures = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            read_failures += 1
            if read_failures > 10:
                print("ERROR: camera stopped returning frames.")
                break
            time.sleep(0.05)
            continue
        read_failures = 0   # reset on successful read

        frame = cv2.flip(frame, 1)

        detections = detector.detect_and_crop(frame)
        frame_log_entries = []

        for bbox, cropped_face in detections:
            if cropped_face.size == 0:
                continue
            x1, y1, x2, y2 = bbox

            crop_rgb   = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
            tensor_img = transform(Image.fromarray(crop_rgb))
            embedding  = model.get_embedding(tensor_img)

            name, similarity = db.find_closest(
                embedding, backend="faiss", index_type="hnsw",
                aggregation="topk", top_k=10, faiss_k=120,
            )

            if name is not None:
                display_text = f"{name} ({similarity:.2f})"
                color  = config.COLOR_DETECTION
                status = "recognized"
            else:
                display_text = "Unknown"
                color  = config.COLOR_UNKNOWN
                status = "unrecognized"
                name, similarity = "Unknown", float("nan")

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            r  = max(x2 - x1, y2 - y1) // 2
            cv2.circle(frame, (cx, cy), r, color, 1, lineType=cv2.LINE_AA)
            cv2.putText(frame, display_text, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1)

            frame_log_entries.append((name, similarity, status))

        now = time.time()
        if now - last_log_time > 2.0 and frame_log_entries:
            for log_name, log_sim, log_status in frame_log_entries:
                logger.log(log_name, log_sim, log_status)
            last_log_time = now

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()