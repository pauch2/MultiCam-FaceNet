import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class TripletFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir should be structured as:
        root_dir/
            person1/
                img1.jpg
                img2.jpg
            person2/
                ...
        """
        self.root_dir = root_dir
        self.transform = transform or T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.image_paths = {cls: [] for cls in self.classes}

        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_dir):
                self.image_paths[cls].append(os.path.join(cls_dir, img_name))

    def __len__(self):
        # Arbitrary epoch size, usually based on number of identities or total images
        return sum([len(paths) for paths in self.image_paths.values()])

    def __getitem__(self, idx):
        # Anchor and Positive
        anchor_class = random.choice(self.classes)
        while len(self.image_paths[anchor_class]) < 2:
            anchor_class = random.choice(self.classes)

        anchor_path, positive_path = random.sample(self.image_paths[anchor_class], 2)

        # Negative
        negative_class = random.choice([c for c in self.classes if c != anchor_class])
        negative_path = random.choice(self.image_paths[negative_class])

        anchor_img = Image.open(anchor_path).convert('RGB')
        positive_img = Image.open(positive_path).convert('RGB')
        negative_img = Image.open(negative_path).convert('RGB')

        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return anchor_img, positive_img, negative_img