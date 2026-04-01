import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset.medical_dataset import HyperReflectiveDataset, get_train_transform, get_val_transform, create_dataloaders
from model.unetplusplus_model import get_unetplusplus_model
import os
import yaml


def combined_loss(preds, targets, bce_weight=0.5, mse_weight=0.5):
    """
    Combined BCE + MSE loss for heatmap prediction.
    preds: raw logits (before sigmoid)
    targets: heatmap values in [0, 1]
    """
    bce = nn.BCEWithLogitsLoss()(preds, targets)
    mse = nn.MSELoss()(torch.sigmoid(preds), targets)
    return bce_weight * bce + mse_weight * mse


def train_model(config):
    """
    Train U-Net++ for hyper-reflective dot heatmap prediction.
    """
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_images_dir=config['train_images'],
        train_heatmaps_dir=config['train_heatmaps'],
        val_images_dir=config['val_images'],
        val_heatmaps_dir=config['val_heatmaps'],
        test_images_dir=config['test_images'],
        test_heatmaps_dir=config['test_heatmaps'],
        batch_size=int(config['batch_size']),
        target_size=int(config.get('target_size', 512))
    )

    # Model
    model = get_unetplusplus_model(
        in_channels=int(config['in_channels']),
        out_classes=int(config['out_classes'])
    ).to(device)

    # Loss, Optimizer, Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config['lr']))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    best_val_loss = float('inf')
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    best_model_path = os.path.join(config['checkpoint_dir'], 'best_model.pth')

    print(f"\nStarting training for {config['epochs']} epochs...\n")
    print("-" * 70)

    for epoch in range(int(config['epochs'])):
        # Training
        model.train()
        train_loss = 0.0

        for images, heatmaps in train_loader:
            images = images.to(device)
            heatmaps = heatmaps.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = combined_loss(outputs, heatmaps)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, heatmaps in val_loader:
                images = images.to(device)
                heatmaps = heatmaps.to(device)

                outputs = model(images)
                loss = combined_loss(outputs, heatmaps)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        print(f"Epoch [{epoch+1:3d}/{config['epochs']}] "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, best_model_path)
            print(f"  -> Saved new best model (val_loss: {val_loss:.4f})")

    print("-" * 70)
    print(f"\nTraining complete! Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {best_model_path}")

    return model, best_model_path
