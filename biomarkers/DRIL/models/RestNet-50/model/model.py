import torchvision.models as models
import torch.nn as nn

def get_resnet50(num_classes):
    model = models.resnet50(pretrained=True)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last block (layer4)
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace fully connected layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model