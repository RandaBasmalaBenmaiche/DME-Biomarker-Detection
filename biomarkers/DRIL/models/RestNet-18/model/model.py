import torchvision.models as models
import torch.nn as nn

def get_model(num_classes):
    # Load pretrained ResNet18
    model = models.resnet18(pretrained=True)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last block (layer4)
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace the fully connected layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model