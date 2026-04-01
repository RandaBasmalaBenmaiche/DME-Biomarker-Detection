import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


class HyperReflectiveDataset(Dataset):
    """
    Dataset for hyper-reflective dot detection.
    Loads grayscale OCT images and corresponding .npy heatmap files.
    """
    def __init__(self, images_dir, heatmaps_dir, transform=None, target_size=512):
        self.images_dir = images_dir
        self.heatmaps_dir = heatmaps_dir
        self.transform = transform
        self.target_size = target_size

        # Get all image files
        self.image_files = sorted([f for f in os.listdir(images_dir)
                                   if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])

        # Verify corresponding heatmaps exist
        self.valid_files = []
        for img_file in self.image_files:
            base_name = os.path.splitext(img_file)[0]
            heatmap_file = base_name + '.npy'
            heatmap_path = os.path.join(heatmaps_dir, heatmap_file)
            if os.path.exists(heatmap_path):
                self.valid_files.append(img_file)
            else:
                print(f"Warning: No heatmap found for {img_file}")

        print(f"Found {len(self.valid_files)} image-heatmap pairs")

    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, idx):
        img_file = self.valid_files[idx]
        base_name = os.path.splitext(img_file)[0]
        heatmap_file = base_name + '.npy'

        # Load image
        img_path = os.path.join(self.images_dir, img_file)
        image = np.array(Image.open(img_path).convert("L"))

        # Load heatmap
        heatmap_path = os.path.join(self.heatmaps_dir, heatmap_file)
        heatmap = np.load(heatmap_path).astype(np.float32)

        # Resize
        image = cv2.resize(image, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
        heatmap = cv2.resize(heatmap, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)

        # Normalize
        image = image.astype(np.float32) / 255.0
        heatmap = np.clip(heatmap, 0, 1)

        if self.transform:
            # Albumentations expects (H, W)
            augmented = self.transform(image=image, mask=heatmap)

            image = augmented["image"]      # (1, H, W)
            heatmap = augmented["mask"]     # (H, W)

            # Ensure heatmap has channel dimension
            if heatmap.ndim == 2:
                heatmap = heatmap.unsqueeze(0)

        else:
            image = torch.from_numpy(image).unsqueeze(0).float()
            heatmap = torch.from_numpy(heatmap).unsqueeze(0).float()

        return image, heatmap


# ================= AUGMENTATIONS ================= #

def get_train_transform(target_size=512):
    """Training transforms with OCT-safe augmentation"""
    return A.Compose([
        A.HorizontalFlip(p=0.5),

        A.Affine(
            scale=(0.95, 1.05),
            translate_percent=(0.05, 0.05),
            rotate=(-3, 3),
            p=0.5
        ),

        A.RandomBrightnessContrast(p=0.3),

        A.GaussianBlur(blur_limit=3, p=0.2),

        A.GaussNoise(var_limit=(5.0, 20.0), p=0.15),

        ToTensorV2()
    ])


def get_val_transform(target_size=512):
    """Validation/test transforms (no augmentation)"""
    return A.Compose([
        ToTensorV2()
    ])


# ================= DATALOADERS ================= #

def create_dataloaders(train_images_dir, train_heatmaps_dir,
                       val_images_dir, val_heatmaps_dir,
                       test_images_dir, test_heatmaps_dir,
                       batch_size=16, target_size=512):
    """
    Create train/val/test dataloaders from pre-split directories.
    """

    train_dataset = HyperReflectiveDataset(
        images_dir=train_images_dir,
        heatmaps_dir=train_heatmaps_dir,
        transform=get_train_transform(target_size),
        target_size=target_size
    )

    val_dataset = HyperReflectiveDataset(
        images_dir=val_images_dir,
        heatmaps_dir=val_heatmaps_dir,
        transform=get_val_transform(target_size),
        target_size=target_size
    )

    test_dataset = HyperReflectiveDataset(
        images_dir=test_images_dir,
        heatmaps_dir=test_heatmaps_dir,
        transform=get_val_transform(target_size),
        target_size=target_size
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=2, pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=2, pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=2, pin_memory=True
    )

    print(f"Data loaded: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")

    return train_loader, val_loader, test_loader