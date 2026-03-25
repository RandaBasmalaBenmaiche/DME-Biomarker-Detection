import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# ===== CONFIG =====
TARGET_MEAN = 0.4
TARGET_STD = 0.2
TARGET_SIZE = 512


class MedicalDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

        self.images = sorted(os.listdir(images_dir))
        self.masks = sorted(os.listdir(masks_dir))

    def __len__(self):
        return len(self.images)

    def normalize_to_target(self, image):
        # Convert to float [0,1]
        image = image.astype(np.float32) / 255.0

        mean = image.mean()
        std = image.std() + 1e-8

        # Only normalize if needed
        if abs(mean - TARGET_MEAN) > 1e-3 or abs(std - TARGET_STD) > 1e-3:
            image = (image - mean) / std
            image = image * TARGET_STD + TARGET_MEAN

        return np.clip(image, 0, 1)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])

        # Load image & mask
        image = np.array(Image.open(img_path).convert("L"))
        mask = np.array(Image.open(mask_path).convert("L"))

        # Binary mask (VERY IMPORTANT)
        mask = (mask != 0).astype("float32")

        # Normalize brightness/contrast
        image = self.normalize_to_target(image)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]              # [1, H, W]
            mask = augmented["mask"].unsqueeze(0)   # 🔥 FIX: [1, H, W]
        else:
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return image, mask


# ===== TRANSFORMS =====
def get_train_transform():
    return A.Compose([
        A.LongestMaxSize(max_size=TARGET_SIZE),  # keep ratio

        A.PadIfNeeded(
            min_height=TARGET_SIZE,
            min_width=TARGET_SIZE,
            border_mode=cv2.BORDER_CONSTANT
        ),

        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),

        ToTensorV2()
    ])