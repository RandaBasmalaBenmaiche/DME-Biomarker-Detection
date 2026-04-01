import torch

def get_predictions(outputs):
    # If already predictions, just return them
    if outputs.ndim == 1:
        return outputs
    return torch.argmax(outputs, dim=1)


def accuracy(outputs, labels):
    preds = get_predictions(outputs)
    return (preds == labels).float().mean().item()


def precision_recall_f1(outputs, labels):
    preds = get_predictions(outputs)

    f1_scores = []
    precisions = []
    recalls = []

    for cls in [0, 1]:
        tp = ((preds == cls) & (labels == cls)).sum().item()
        fp = ((preds == cls) & (labels != cls)).sum().item()
        fn = ((preds != cls) & (labels == cls)).sum().item()

        if tp + fp == 0:
            precision = 0.0
        else:
            precision = tp / (tp + fp)

        if tp + fn == 0:
            recall = 0.0
        else:
            recall = tp / (tp + fn)

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    return {
        "precision": sum(precisions) / len(precisions),
        "recall": sum(recalls) / len(recalls),
        "f1": sum(f1_scores) / len(f1_scores),
    }


def compute_metrics(outputs, labels):
    acc = accuracy(outputs, labels)
    prf = precision_recall_f1(outputs, labels)

    return {
        "accuracy": acc,
        "precision": prf["precision"],
        "recall": prf["recall"],
        "f1": prf["f1"],
    }