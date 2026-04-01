import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from dataset.medical_dataset import MedicalDataset, get_train_transform
from model.attention_unet_model import get_attention_unet_model
import os

def dice_coef(preds, targets, smooth=1e-6):
    """
    Computes Dice coefficient (F1 score) for binary masks.
    preds and targets should be binary tensors [B, 1, H, W]
    """
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    return (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)

def train_model(config):
    # Automatically select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Dataset and DataLoader
    train_dataset = MedicalDataset(
        config['train_images'],
        config['train_masks'],
        transform=get_train_transform()
    )
    val_dataset = MedicalDataset(
        config['val_images'],
        config['val_masks'],
        transform=get_train_transform()
    )

    train_loader = DataLoader(train_dataset, batch_size=int(config['batch_size']), shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=int(config['batch_size']), shuffle=False, num_workers=2)

    # Model
    model = get_attention_unet_model(config['in_channels'], config['out_classes']).to(device)

    # Loss
    dice_loss = smp.losses.DiceLoss(mode='binary')
    bce_loss = nn.BCEWithLogitsLoss()
    def criterion(preds, targets):
        return bce_loss(preds, targets) + dice_loss(preds, targets)

    # Optimizer + Scheduler
    optimizer = optim.Adam(model.parameters(), lr=float(config['lr']))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    best_val_dice = 0.0
    os.makedirs(config['checkpoint_dir'], exist_ok=True)

    # ----------------------- Training Loop -----------------------
    for epoch in range(int(config['epochs'])):
        # Training
        model.train()
        train_loss = 0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        val_dice = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                # Compute Dice (F1) for binary segmentation
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                val_dice += dice_coef(preds, masks).item()

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        scheduler.step(val_loss)

        print(f"Epoch [{epoch+1}/{config['epochs']}] "
              f"Train Loss: {train_loss:.4f} "
              f"Val Loss: {val_loss:.4f} "
              f"Val Dice: {val_dice:.4f} "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best model
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(
                model.state_dict(),
                os.path.join(config['checkpoint_dir'], 'best_model.pth')
            )
