"""
arcface_dataset.py
------------------
Labeled face dataset for ArcFace training.
Returns (image_tensor, class_index) pairs — unlike TripletFaceDataset
which returns (anchor, positive, negative) triplets.

Same folder structure as TripletFaceDataset:
    root_dir/
        person1/
            img1.jpg
            img2.jpg
        person2/
            ...

The class index is the alphabetically sorted position of the identity folder.
This is deterministic across runs as long as the folder names don't change.
"""

import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Standard ImageNet normalisation — same as the rest of the pipeline
_DEFAULT_TRANSFORM = T.Compose([
    T.RandomHorizontalFlip(),                            # light augmentation
    T.ColorJitter(brightness=0.2, contrast=0.2,
                  saturation=0.1, hue=0.05),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


class LabeledFaceDataset(Dataset):
    """
    Flat labeled dataset for classification-style losses (ArcFace, CosFace, …).

    Parameters
    ----------
    root_dir : str | Path
        Root folder containing one sub-folder per identity.
    transform : callable, optional
        Image transform.  Defaults to resize + ImageNet normalisation + mild augmentation.
    min_imgs : int
        Skip identities with fewer than this many images.
    """

    def __init__(self, root_dir, transform=None, min_imgs: int = 1):
        self.root_dir  = Path(root_dir)
        self.transform = transform or _DEFAULT_TRANSFORM

        # Sorted class list → deterministic class indices
        all_classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

        self.classes    = []
        self.class_to_idx: dict[str, int] = {}
        self.samples: list[tuple[Path, int]] = []   # (img_path, class_idx)

        idx = 0
        skipped = 0
        for cls in all_classes:
            cls_dir = self.root_dir / cls
            imgs = [
                cls_dir / fn
                for fn in os.listdir(cls_dir)
                if Path(fn).suffix.lower() in IMG_EXTS
            ]
            if len(imgs) < min_imgs:
                skipped += 1
                continue

            self.classes.append(cls)
            self.class_to_idx[cls] = idx
            for img_path in imgs:
                self.samples.append((img_path, idx))
            idx += 1

        if skipped:
            print(f"[LabeledFaceDataset] Skipped {skipped} identities "
                  f"with < {min_imgs} image(s).")

        print(f"[LabeledFaceDataset] {len(self.classes)} classes, "
              f"{len(self.samples)} images.")

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, class_idx = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, class_idx