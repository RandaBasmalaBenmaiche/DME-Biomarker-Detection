import sys
from pathlib import Path
project_root = Path("/content/RestNet-50")
sys.path.append(str(project_root))

import torch
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn as nn

from utils.metrics import compute_metrics


def get_test_loader(config):
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4, 0.4, 0.4],
            std=[0.2, 0.2, 0.2]
        )
    ])

    test_dataset = datasets.ImageFolder(
        root=os.path.join(config["data_dir"], "test"),
        transform=test_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False
    )

    return test_loader, test_dataset.classes


def load_model(model_path, num_classes, device):
    model_name = os.path.basename(model_path).lower()

    # ===== DETECT ARCHITECTURE =====
    if "resnet18" in model_name:
        print("→ Detected: ResNet18")
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif "resnet50" in model_name:
        print("→ Detected: ResNet50")
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif "efficientnet" in model_name:
        print("→ Detected: EfficientNet-B0")
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, num_classes
        )

    else:
        raise ValueError(f"❌ Unknown model type for: {model_name}")

    # ===== LOAD CHECKPOINT =====
    checkpoint = torch.load(model_path, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model


def test_model(model, test_loader, device):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    metrics = compute_metrics(all_preds, all_labels)
    return metrics


if __name__ == "__main__":
    config = {
        "data_dir": "/content/RestNet-50/dataset/data",
        "batch_size": 16,
        "models_dir": "/content/RestNet-50/Models/Models",  # 👈 folder with multiple .pth files
        "num_classes": 2
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loader, class_names = get_test_loader(config)
    print(f"Classes: {class_names}")

    results = []

    # ===== LOOP OVER MODELS =====
    for model_file in os.listdir(config["models_dir"]):
        if not model_file.endswith(".pth"):
            continue

        model_path = os.path.join(config["models_dir"], model_file)
        print(f"\n🔍 Evaluating: {model_file}")

        model = load_model(model_path, config["num_classes"], device)
        metrics = test_model(model, test_loader, device)

        print(
            f"Accuracy : {metrics['accuracy']:.4f} | "
            f"F1: {metrics['f1']:.4f} | "
            f"Precision: {metrics['precision']:.4f} | "
            f"Recall: {metrics['recall']:.4f}"
        )

        results.append({
            "model": model_file,
            **metrics
        })

    # ===== FIND BEST MODEL =====
    best_model = max(results, key=lambda x: x["f1"])

    print("\n🏆 ===== BEST MODEL =====")
    print(f"Model: {best_model['model']}")
    print(
        f"F1: {best_model['f1']:.4f} | "
        f"Accuracy: {best_model['accuracy']:.4f}"
    )

    import pandas as pd

df = pd.DataFrame(results)
df.to_csv("benchmark_results.csv", index=False)

print("\n📁 Results saved to benchmark_results.csv")