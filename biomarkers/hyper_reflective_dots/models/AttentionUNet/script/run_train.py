import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import yaml
from trainer.train import train_model

# Load config
config_path = project_root / "configs" / "config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

train_model(config)
