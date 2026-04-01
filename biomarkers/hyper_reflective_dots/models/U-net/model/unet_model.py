import segmentation_models_pytorch as smp


def get_unet_model(in_channels=1, out_classes=1):
    """
    Returns a U-Net model with a pretrained ResNet34 encoder.
    Output is raw logits (sigmoid applied during inference).
    """
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=out_classes,
        activation=None  # Sigmoid applied separately for numerical stability
    )
    return model
