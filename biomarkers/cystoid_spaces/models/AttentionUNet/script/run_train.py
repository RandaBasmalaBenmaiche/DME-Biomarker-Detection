import sys
from pathlib import Path
project_root = Path("/content/U-net/U-net")
sys.path.append(str(project_root))

import yaml
from trainer.train import train_model

# Load config
with open("/content/U-net/U-net/configs/config.yaml") as f:
    config = yaml.safe_load(f)

train_model(config)
