import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1

class FaceEmbeddingModel(nn.Module):
    def __init__(self, embedding_dim=128):
        super(FaceEmbeddingModel, self).__init__()
        # ResNet18 backbone, pretrained=False as requested
        # self.backbone = models.resnet18(weights="IMAGENET1K_V1")
        self.backbone = models.mobilenet_v3_small()
        # self.backbone = models.mnasnet1_3(weights="DEFAULT")
        self.backbone.classifier = nn.Identity()
        # self.backbone = InceptionResnetV1(pretrained='vggface2')
        # self.backbone.classify = False
        # Replace the final fully connected layer

        # self.backbone.fc = nn.Identity()
        # self.backbone = models.swin_t(weights="DEFAULT")
        # self.backbone.head = nn.Identity()

    def forward(self, x):
        # Extract features
        x = self.backbone(x)
        # L2 Normalization
        x = F.normalize(x, p=2, dim=1)
        return x

    def get_embedding(self, face_tensor):
        """Helper to get embedding for a single preprocessed face tensor"""
        self.eval()
        with torch.no_grad():
            if len(face_tensor.shape) == 3:
                face_tensor = face_tensor.unsqueeze(0)  # Add batch dimension
            embedding = self.forward(face_tensor)
        return embedding.squeeze(0).cpu().numpy()

def load_embedding_model(
        embedding_dim: int = None,
        weights_path: str = None,
        device: str = None,
) -> "FaceEmbeddingModel":
    """
    Convenience factory: build, load weights, move to device, set eval mode.
    Call this everywhere instead of manually constructing FaceEmbeddingModel
    so weight-loading can never be accidentally omitted.
    """
    import torch
    import os
    import config as _config

    if embedding_dim is None:
        embedding_dim = _config.EMBEDDING_DIM
    if weights_path is None:
        weights_path = _config.MODEL_WEIGHTS_PATH

    model = FaceEmbeddingModel(embedding_dim=embedding_dim)

    if os.path.exists(weights_path):
        ckpt = torch.load(weights_path, map_location="cpu")
        # Support both plain state_dict and full checkpoint dicts (train.py saves both)
        model.load_state_dict(ckpt.get("model_state", ckpt))
        print(f"[EmbeddingModel] Loaded weights: {weights_path}")
    else:
        print(f"[EmbeddingModel] WARNING: weights not found at {weights_path} "
              f"— using pretrained backbone only. Recognition will be unreliable.")

    if device is None:
        device_str = getattr(_config, "DEVICE", None)
        device = device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu")

    return model.to(device).eval()