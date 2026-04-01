import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import os
import torch
import numpy as np
from PIL import Image
import cv2
import yaml
from dataset.medical_dataset import HyperReflectiveDataset
from model.unetplusplus_model import get_unetplusplus_model


def evaluate_model(config):
    """
    Run inference on test set and save predicted heatmaps.
    """
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = get_unetplusplus_model(
        in_channels=int(config['in_channels']),
        out_classes=int(config['out_classes'])
    ).to(device)

    checkpoint_path = os.path.join(config['checkpoint_dir'], 'best_model.pth')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded model from: {checkpoint_path}")

    # Create output directory
    output_dir = config.get('output_dir', 'output/predictions')
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    dataset = HyperReflectiveDataset(
        images_dir=config['images_dir'],
        heatmaps_dir=config['heatmaps_dir'],
        transform=None,
        target_size=int(config.get('target_size', 512))
    )

    print(f"\nRunning inference on {len(dataset)} samples...\n")

    # Inference
    with torch.no_grad():
        for idx in range(len(dataset)):
            img_file = dataset.valid_files[idx]
            base_name = os.path.splitext(img_file)[0]

            # Load and preprocess image
            img_path = os.path.join(config['images_dir'], img_file)
            image = np.array(Image.open(img_path).convert("L"))
            image_resized = cv2.resize(image, (int(config.get('target_size', 512)), int(config.get('target_size', 512))), interpolation=cv2.INTER_LINEAR)
            image_norm = image_resized.astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_norm).unsqueeze(0).unsqueeze(0).to(device)

            # Predict
            output = model(image_tensor)
            pred_heatmap = torch.sigmoid(output).squeeze().cpu().numpy()

            # Save prediction
            pred_path = os.path.join(output_dir, f"{base_name}_pred.npy")
            np.save(pred_path, pred_heatmap)

            # Save as PNG for visualization
            pred_png_path = os.path.join(output_dir, f"{base_name}_pred.png")
            pred_vis = (pred_heatmap * 255).astype(np.uint8)
            pred_vis_colored = cv2.applyColorMap(pred_vis, cv2.COLORMAP_JET)
            cv2.imwrite(pred_png_path, pred_vis_colored)

            # Load ground truth for comparison
            gt_path = os.path.join(config['heatmaps_dir'], f"{base_name}.npy")
            gt_heatmap = np.load(gt_path)
            gt_resized = cv2.resize(gt_heatmap, (int(config.get('target_size', 512)), int(config.get('target_size', 512))), interpolation=cv2.INTER_LINEAR)

            # Calculate simple MSE
            mse = np.mean((pred_heatmap - gt_resized) ** 2)

            print(f"[{idx+1}/{len(dataset)}] {img_file} | MSE: {mse:.4f}")

    print(f"\nPredictions saved to: {output_dir}")


if __name__ == "__main__":
    # Load config
    config_path = project_root / "configs" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    evaluate_model(config)
