import torchvision.models as models
import torch.nn as nn

def get_efficientnet_b0(num_classes):
    model = models.efficientnet_b0(pretrained=True)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last blocks of feature extractor
    for param in model.features[-1].parameters():
        param.requires_grad = True

    # Replace classifier
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model