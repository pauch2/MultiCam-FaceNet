# MultiCam-FaceNet
End-to-end real-time face recognition platform for multi-camera systems. Combines YOLOv26 detection with a lightweight MobileNet embedding model trained using Triplet Loss or ArcFace. Features live monitoring, instant identification, structured logging with image snapshots, and a role-based admin dashboard for seamless control and scaling.




---

## Project Structure

```
face_id_demo/
│
├── config.py                   # All tuneable constants
├── requirements.txt
│
├── api/                        # FastAPI application
│   ├── main.py                 # App entry point, startup/shutdown
│   ├── auth.py                 # JWT authentication, bcrypt, role helpers
│   ├── shared_db.py            # Single shared FaceDatabase instance
│   ├── camera_stream.py        # CameraManager — per-camera capture + ML threads
│   ├── camera_pool.py          # CameraPool — manages N CameraManagers
│   ├── batch_embedder.py       # SharedBatchEmbedder — batches GPU inference across cameras
│   ├── frame_processor.py      # FrameProcessor — used by WebSocket client-cam path
│   ├── routers/
│   │   ├── admin.py            # /admin/* — user management, audit log
│   │   ├── cameras.py          # /cameras/* — CRUD, stream, threshold, metrics
│   │   ├── client_cam.py       # /clientcam/ws — browser WebSocket camera
│   │   ├── faces.py            # /database/* — face vector DB CRUD
│   │   ├── logs.py             # /logs/* — detections, unknowns, xlsx export
│   │   ├── stream.py           # /stream/* — legacy single-camera endpoints
│   │   └── users.py            # /users/* — registration, password change
│   └── static/
│       ├── ui.html
│       ├── ui.js
│       └── ui.css
│
├── database/
│   ├── models.py               # SQLAlchemy ORM models
│   ├── session.py              # DB init, migrations, seeding
│   ├── vector_store.py         # FaceDatabase — FAISS HNSW + SQLite embedding store
│   └── db_editor.py            # CLI tool to rename/inspect the vector DB
│
├── models/
│   ├── embedding_model.py      # FaceEmbeddingModel (Swin-T + optional projection head)
│   └── detector.py             # FaceDetector (YOLOv8 wrapper)
│
├── training/
│   ├── train.py                # Training loop — supports Triplet Loss and ArcFace
│   ├── test.py                 # Evaluation — threshold sweep, confusion matrix
│   ├── dataset.py              # TripletFaceDataset — online random triplet sampling
│   ├── arcface_dataset.py      # LabeledFaceDataset — (image, label) pairs for ArcFace
│   ├── triplet_loss.py         # SemiHardTripletLoss
│   └── arcface_loss.py         # ArcFaceHead — additive angular margin loss
│
├── scripts/
│   ├── recognize_face.py       # Standalone webcam recognition (no server needed)
│   └── register_face.py        # Standalone webcam registration (no server needed)
│
└── logs/
    └── logger.py               # AccessLogger — CSV log writer for standalone scripts
```

---

## Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU:** Install the CUDA build of PyTorch first if you want GPU acceleration:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```
> Then run `pip install -r requirements.txt` as normal — pip will skip torch since it is already installed.

### 3. Place model weights

```
face_id_demo/
├── models/
│   └── swin_t_VGGFace2_train_pre_train.pth   ← trained embedding model
└── best26_v19s_0_7.pt                          ← YOLOv26 face detector
```

Paths are configurable in `config.py`.

### 4. Run the server

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8080
```

Open **http://localhost:8080/ui** — log in with `admin1` / `password`.

---

## Configuration (`config.py`)

| Setting | Default | Description |
|---|---|---|
| `MODEL_NAME` | `swin_t_VGGFace2_train_pre_train` | Stem for weights and DB filenames |
| `YOLO_MODEL_PATH` | `best26_v19s_0_7.pt` | YOLO face detector weights |
| `EMBEDDING_DIM` | `768` | `768` = raw Swin-T output; any other value adds a projection MLP (requires retraining) |
| `SIMILARITY_THRESHOLD` | `0.5` | Default cosine similarity cutoff, overridable per camera at runtime |
| `MARGIN` | `0.3` | Triplet loss margin |
| `DEVICE` | `None` | `None` = auto-detect CUDA. Set `"cpu"` or `"cuda:0"` to override |
| `IMAGE_SIZE` | `(224, 224)` | Face crop size fed to the embedding model |
| `FLIP_HORIZONTAL` | `True` | Mirror-flip frames (webcams) |
| `LOG_INTERVAL_SEC` | `2.0` | Min seconds between DB log entries for the same person per camera |

---

## Training

Dataset structure (same for both loss functions):

```
dataset/
├── Alice/
│   ├── 001.jpg
│   └── 002.jpg
├── Bob/
│   └── 001.jpg
...
```

### Triplet Loss (original)

```bash
python training/train.py /path/to/dataset
python training/train.py /path/to/dataset --epochs 50 --batch_size 128
```

### ArcFace

```bash
# Auto-tunes margin and scale for your dataset size
python training/train.py /path/to/dataset --loss arcface

# Manual control
python training/train.py /path/to/dataset --loss arcface \
    --arc_margin 0.4 --arc_scale 50 --arc_warmup 10

# Resume from checkpoint
python training/train.py /path/to/dataset --loss arcface --resume
```

**ArcFace tips:**
- `--arc_easy_margin` is on by default — prevents 0 % accuracy at init
- `--arc_warmup N` ramps margin from 0 → target over N epochs (default 5)
- Margin and scale are auto-set from dataset size if not specified
- ArcFace tends to outperform Triplet on large datasets (> 1 000 identities); results vary on small ones

### Evaluation

```bash
python training/test.py --data_dir /path/to/dataset

# Reuse an existing DB (skip the embedding build step)
python training/test.py --data_dir /path/to/dataset --db_mode reuse

# Point at a specific DB file
python training/test.py --data_dir /path/to/dataset \
    --db_path database/my_experiment.db --db_mode reuse

# Full threshold sweep
python training/test.py --data_dir /path/to/dataset \
    --thr_min 0.3 --thr_max 0.8 --thr_steps 21
```

Outputs saved to `runs/eval_<model>_split_80_20/`:
- `confusion_matrix.png`
- `threshold_sweep.csv` / `threshold_sweep.png`
- `fp_examples_thr_*.csv` — highest-similarity false positives per threshold
- `report.txt`




## Web UI

<img width="1919" height="1079" alt="Screenshot 2026-04-29 145338" src="https://github.com/user-attachments/assets/93d42a6d-d503-4c66-a5da-1cdb8a2fbe6d" />

| Page | Roles | Description |
|---|---|---|
| **Live Feeds** | staff | MJPEG streams from all active cameras in a grid |
| **My Log** | all | Detections of the logged-in user |
| **Register** | staff | Capture face anchors via webcam or server camera, save under a username |
| **Cameras** | staff | Start/stop cameras, adjust recognition threshold, stream quality, processing resolution |
| **📊 Metrics** | staff | Live detection fps, recognition fps, frame fps, GPU batch size (auto-refreshes) |
| **Users** | staff | Create accounts, change display names, change roles |
| **Face DB** | staff | View registered face embeddings, see which user each is linked to, delete records |
| **Detection Log** | staff | All detections — filter by camera, name, date. Export to Excel |
| **Audit Log** | admin | Every admin/moderator action |

---

## API

Full interactive docs at **http://localhost:8080/docs**.

```
POST /token                               Login → JWT token

GET  /cameras/                            List cameras with live status
POST /cameras/                            Register new camera
POST /cameras/{id}/start                  Start streaming
POST /cameras/{id}/stop                   Stop streaming
POST /cameras/{id}/recognition            Toggle recognition on/off
POST /cameras/{id}/quality                Set JPEG quality, resolution, threshold
GET  /cameras/{id}/threshold              Get live threshold
POST /cameras/{id}/threshold              Set threshold (query param)
GET  /cameras/{id}/metrics                Live fps/ms per pipeline stage
GET  /cameras/stream/{id}                 MJPEG stream  (no auth — <img src> safe)
GET  /cameras/frame/{id}                  Single JPEG snapshot

GET  /logs/detections                     All detections (staff), with filters
GET  /logs/my                             Current user's detections
GET  /logs/export/xlsx                    Excel export (3 sheets)

GET  /database/summary                    Face embeddings per username + linked user
DELETE /database/name/{name}              Delete all embeddings for a username

GET  /admin/users                         List users
POST /admin/users                         Create user
PATCH /admin/users/{id}/role              Change role
PATCH /admin/users/{id}/name             Change display name
DELETE /admin/users/{id}                  Delete user
GET  /admin/audit                         Audit log

POST /users/register                      Public self-registration
POST /users/me/password                   Change own password
```

---

## How Recognition Works

1. **Capture** — two threads per camera: one drains the OpenCV buffer via `grab()`, the other runs ML on the latest frame (stale frames are dropped, not queued)
2. **Detection** — YOLOv8 finds face bounding boxes in the (optionally downscaled) frame
3. **Embedding** — each crop is submitted to `SharedBatchEmbedder`, which collects all crops from all cameras for 8 ms and runs a single batched GPU forward pass through the Swin-T backbone → L2-normalised vector
4. **Search** — FAISS HNSW nearest-neighbour search over the registered embeddings
5. **Aggregation** — for each candidate user, the mean of their top-K matching embeddings is taken
6. **Threshold** — if best score ≥ `similarity_threshold`, the face is labelled; otherwise "Unknown"

The recognition threshold can be set globally in `config.py` or per camera at runtime via the UI slider or `POST /cameras/{id}/threshold?threshold=0.7`.

---

## Roles & Permissions

| | User | Moderator | Admin |
|---|:---:|:---:|:---:|
| View own detections | ✅ | ✅ | ✅ |
| Change own password | ✅ | ✅ | ✅ |
| View all detections / manage cameras / register faces | ❌ | ✅ | ✅ |
| Change display names / create users | ❌ | ✅ | ✅ |
| Change roles / delete users / view audit log | ❌ | ❌ | ✅ |

---

## Security Notes

- Change `SECRET_KEY` in `api/auth.py` (or set `JWT_SECRET_KEY` env var) before any non-local deployment
- The bootstrap account `admin1` / `password` is seeded on first run — change the password or delete it immediately in production
- Passwords are hashed with **bcrypt**; tokens use **JWT HS256**
