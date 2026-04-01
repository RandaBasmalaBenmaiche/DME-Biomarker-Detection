import torch
import os
from utils.metrics import compute_metrics

def train_model(model, train_loader, val_loader, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ===== CLASS WEIGHTS =====
    class_counts = config["class_counts"]

    weights = torch.tensor([
        sum(class_counts) / class_counts[0],
        sum(class_counts) / class_counts[1]
    ], dtype=torch.float32).to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    # ===== OPTIMIZER (ONLY TRAINABLE PARAMS) =====
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["lr"]
    )

    # ===== SAVE SETUP =====
    os.makedirs(config["save_dir"], exist_ok=True)
    best_f1 = 0.0

    for epoch in range(config["epochs"]):

        # ===== TRAIN =====
        model.train()
        train_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # ===== VALIDATION =====
        model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)

                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        metrics = compute_metrics(all_preds, all_labels)

        print(
            f"Epoch {epoch+1}/{config['epochs']} | "
            f"Loss: {train_loss:.4f} | "
            f"Acc: {metrics['accuracy']:.4f} | "
            f"F1: {metrics['f1']:.4f} | "
            f"Precision: {metrics['precision']:.4f} | "
            f"Recall: {metrics['recall']:.4f}"
        )

        # ===== SAVE BEST MODEL (BASED ON F1) =====
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "f1": best_f1
            }, os.path.join(config["save_dir"], "best_model.pth"))

            print(f"✅ New best model saved with F1: {best_f1:.4f}")

    # ===== SAVE FINAL MODEL =====
    torch.save(model.state_dict(), os.path.join(config["save_dir"], "last_model.pth"))

    print(f"🏁 Training finished. Best F1: {best_f1:.4f}")