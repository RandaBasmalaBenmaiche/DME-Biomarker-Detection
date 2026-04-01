import segmentation_models_pytorch as smp

def get_unetplusplus_model(in_channels=1, out_classes=1):
    """
    Returns a U-Net++ model with a pretrained encoder.

    U-Net++ features:
    - Nested dense skip connections between encoder and decoder
    - Reduces semantic gap between encoder and decoder features
    - Better feature fusion for complex segmentation tasks
    """
    model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=out_classes
    )
    return model
