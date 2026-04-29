import os

# Project Roots
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "mobilenet_v3__small_VGGFace(scheduler+arcface+old_aug)_train_pre_train(1000_epoch)"
DB_PATH = os.path.join(BASE_DIR, 'database', f'(MODEL_TEST){MODEL_NAME}.db')
# DB_PATH = os.path.join(BASE_DIR, 'database', f'faces_mobilenet_v3_small_VGGFace2_train_pre_train.db')
LOG_PATH = os.path.join(BASE_DIR, 'logs', 'access_log.csv')
MODEL_WEIGHTS_PATH = os.path.join(BASE_DIR, 'models', f'{MODEL_NAME}.pth')

# Make directories if they don't exist
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

# YOLO config
# Note: Standard YOLOv8 detects people. For faces, a community model like 'yolov8n-face.pt' is recommended.
YOLO_MODEL_PATH = r"C:\Users\pv_ts\Desktop\Temp\best26_v19s_0_7_int8_openvino_model" #
# YOLO_MODEL_PATH = 'best26_v19s_0_7.pt'

# Embedding Model Config
EMBEDDING_DIM = 128
IMAGE_SIZE = (224, 224)
MARGIN = 0.3

# Recognition Config
SIMILARITY_THRESHOLD = 0.5 # Cosine similarity threshold


# Colours (GRB)
COLOR_CAPTURING = (49, 232, 248)
COLOR_DETECTION = (235, 115, 12)
COLOR_UNKNOWN = (100, 97, 255)