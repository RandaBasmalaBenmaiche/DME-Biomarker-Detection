import segmentation_models_pytorch as smp

def get_unet_model(in_channels=1, out_classes=1):
    """
    Returns a U-Net model with a pretrained encoder.
    """
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=out_classes
    )
    return model