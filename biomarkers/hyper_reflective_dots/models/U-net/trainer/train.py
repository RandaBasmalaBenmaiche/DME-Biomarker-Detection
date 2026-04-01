import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset.medical_dataset import create_dataloaders
from model.unet_model import get_unet_model
import os
import yaml

# -----------------------
# Loss functions
# -----------------------
def dice_loss(preds, targets, smooth=1e-6):
    """
    Dice loss for sparse heatmaps.
    preds: logits or probabilities (after sigmoid if logits=False)
    targets: ground truth heatmaps in [0,1]
    """
    preds = torch.sigmoid(preds)  # Ensure probabilities
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)
    intersection = (preds_flat * targets_flat).sum()
    return 1 - (2 * intersection + smooth) / (preds_flat.sum() + targets_flat.sum() + smooth)

def combined_loss(preds, targets, dice_weight=0.5, mse_weight=0.5):
    """
    Combined Dice + MSE loss for heatmap regression.
    """
    mse = nn.MSELoss()(torch.sigmoid(preds), targets)
    d_loss = dice_loss(preds, targets)
    return dice_weight * d_loss + mse_weight * mse

# -----------------------
# Training function
# -----------------------
def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # DataLoaders
    train_loader, val_loader, _ = create_dataloaders(
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
    model = get_unet_model(
        in_channels=int(config['in_channels']),
        out_classes=int(config['out_classes'])
    ).to(device)

    # Optimizer & Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config['lr']))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Checkpointing
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    best_model_path = os.path.join(config['checkpoint_dir'], 'best_model.pth')
    best_val_loss = float('inf')

    print(f"\nStarting training for {config['epochs']} epochs...\n")
    print("-" * 70)

    for epoch in range(int(config['epochs'])):
        # -----------------------
        # Training
        # -----------------------
        model.train()
        train_loss = 0.0

        for images, heatmaps in train_loader:
            images, heatmaps = images.to(device), heatmaps.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = combined_loss(outputs, heatmaps)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # -----------------------
        # Validation
        # -----------------------
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, heatmaps in val_loader:
                images, heatmaps = images.to(device), heatmaps.to(device)
                outputs = model(images)
                loss = combined_loss(outputs, heatmaps)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        print(f"Epoch [{epoch+1:3d}/{config['epochs']}] "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
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
            print(f"  -> Saved new best model (val_loss: {best_val_loss:.6f})")

    print("-" * 70)
    print(f"\nTraining complete! Best validation loss: {best_val_loss:.6f}")
    print(f"Best model saved to: {best_model_path}")

    return model, best_model_path