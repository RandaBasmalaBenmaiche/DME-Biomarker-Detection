import sys
from pathlib import Path
project_root = Path("/content/RestNet-50")
sys.path.append(str(project_root))


import yaml
from dataset.dataset import get_dataloaders
from model.model import get_model
from trainer.trainer import train_model

# Load config
with open("/content/RestNet-50/configs/config.yaml") as f:
    config = yaml.safe_load(f)

# Data
train_loader, val_loader = get_dataloaders(config)

# Model
model = get_model(config["num_classes"])

# Train
train_model(model, train_loader, val_loader, config)