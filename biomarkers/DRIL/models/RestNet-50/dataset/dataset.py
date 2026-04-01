import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(config):

    # --------- TRAIN TRANSFORMS (WITH AUGMENTATION) ---------
    train_transform = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),

        # OCT-specific augmentations
        transforms.RandomResizedCrop(
            config["image_size"],
            scale=(0.9, 1.0),
            ratio=(0.95, 1.05)
        ),
        transforms.RandomHorizontalFlip(p=0.5),

        transforms.RandomRotation(degrees=10),

        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05)
        ),

        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),

        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.4, 0.4, 0.4],
            std=[0.2, 0.2, 0.2]
        )
    ])

    # --------- VALIDATION TRANSFORMS (NO AUGMENTATION) ---------
    val_transform = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4, 0.4, 0.4],
            std=[0.2, 0.2, 0.2]
        )
    ])

    train_dataset = datasets.ImageFolder(
        root=os.path.join(config["data_dir"], "train"),
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        root=os.path.join(config["data_dir"], "val"),
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False
    )

    return train_loader, val_loader