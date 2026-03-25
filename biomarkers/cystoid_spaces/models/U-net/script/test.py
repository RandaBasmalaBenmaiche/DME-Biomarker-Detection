import torch
import numpy as np
import cv2
import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm import tqdm

# ===== CONFIG =====
IMAGES_DIR = "/content/test_324/test/images"
MASKS_DIR = "/content/test_324/test/masks"
MODEL_PATH = "/content/best_model (2).pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_MEAN, TARGET_STD, SIZE = 0.4, 0.2, 512

# ===== UTILS =====
def normalize_to_target(image):
    image = image.astype(np.float32) / 255.0
    mean, std = image.mean(), image.std() + 1e-8
    if abs(mean - TARGET_MEAN) > 1e-3 or abs(std - TARGET_STD) > 1e-3:
        image = (image - mean) / std
        image = image * TARGET_STD + TARGET_MEAN
    return image # Removed np.clip per our discussion

def dice_score(pred, target, eps=1e-7):
    inter = (pred * target).sum()
    return (2. * inter + eps) / (pred.sum() + target.sum() + eps)

def iou_score(pred, target, eps=1e-7):
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return (inter + eps) / (union + eps)

# ===== SETUP =====
transform = A.Compose([
    A.LongestMaxSize(max_size=SIZE),
    A.PadIfNeeded(min_height=SIZE, min_width=SIZE, border_mode=cv2.BORDER_CONSTANT),
    ToTensorV2()
])

model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=1, classes=1).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ===== BATCH PROCESSING =====
image_files = sorted(os.listdir(IMAGES_DIR))
all_dice, all_iou = [], []
results_to_plot = []

# Select 5 random indices to visualize later
sample_indices = random.sample(range(len(image_files)), min(5, len(image_files)))

print(f"Evaluating {len(image_files)} images...")

with torch.no_grad():
    for i, filename in enumerate(tqdm(image_files)):
        # Load
        img_path = os.path.join(IMAGES_DIR, filename)
        mask_path = os.path.join(MASKS_DIR, filename.replace(".jpg", ".png")) # Adjust extension if needed
        
        raw_img = np.array(Image.open(img_path).convert("L"))
        raw_mask = (np.array(Image.open(mask_path).convert("L")) != 0).astype("float32")

        # Process
        img_norm = normalize_to_target(raw_img)
        aug = transform(image=img_norm, mask=raw_mask)
        img_tensor = aug["image"].unsqueeze(0).to(DEVICE)
        mask_tensor = aug["mask"].to(DEVICE)

        # Predict
        out = model(img_tensor)
        pred = (torch.sigmoid(out) > 0.5).float().squeeze(0) # [1, H, W]

        # Score
        d = dice_score(pred, mask_tensor).item()
        io = iou_score(pred, mask_tensor).item()
        all_dice.append(d)
        all_iou.append(io)

        # Save samples for visual
        if i in sample_indices:
            results_to_plot.append((img_tensor.squeeze().cpu(), mask_tensor.squeeze().cpu(), pred.squeeze().cpu()))

# ===== FINAL STATS =====
print(f"\nMean Dice: {np.mean(all_dice):.4f}")
print(f"Mean IoU : {np.mean(all_iou):.4f}")

# ===== VISUALIZE 5 SAMPLES =====
plt.figure(figsize=(15, 10))
for idx, (img, gt, pd) in enumerate(results_to_plot):
    plt.subplot(5, 3, idx*3 + 1)
    plt.imshow(img, cmap="gray"); plt.title("Input Image"); plt.axis("off")
    plt.subplot(5, 3, idx*3 + 2)
    plt.imshow(gt, cmap="gray"); plt.title("Ground Truth"); plt.axis("off")
    plt.subplot(5, 3, idx*3 + 3)
    plt.imshow(pd, cmap="gray"); plt.title("Prediction"); plt.axis("off")
plt.tight_layout()
plt.show()