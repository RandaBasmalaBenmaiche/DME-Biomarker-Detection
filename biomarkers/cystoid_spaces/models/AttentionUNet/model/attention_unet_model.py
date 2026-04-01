import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class AttentionGate(nn.Module):
    """
    Attention Gate for Attention U-Net.
    Filters features to highlight relevant regions for segmentation.
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g is upsampled from decoder, x is from encoder (smaller spatial size)
        # Need to upsample x to match g's spatial size
        if g.shape[2:] != x.shape[2:]:
            x = F.interpolate(x, size=g.shape[2:], mode='bilinear', align_corners=True)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        # Interpolate attention mask to match x's original size, then apply
        psi = F.interpolate(psi, size=x.shape[2:], mode='bilinear', align_corners=True)
        return x * psi


class ConvBlock(nn.Module):
    """Double convolution block"""
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class AttentionUNet(nn.Module):
    """
    Attention U-Net with ResNet34 encoder.
    Uses attention gates in skip connections to focus on relevant regions.
    """
    def __init__(self, in_channels=1, out_classes=1):
        super(AttentionUNet, self).__init__()

        # ResNet34 backbone (pretrained on ImageNet)
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

        # Encoder (downsampling path)
        # Modify first conv layer for 1-channel grayscale input
        self.enc1_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Copy weights from pretrained conv1 if in_channels==3, otherwise initialize
        if in_channels == 3:
            self.enc1_conv.weight = resnet.conv1.weight
        else:
            # Average the RGB weights to get single-channel weights
            self.enc1_conv.weight = nn.Parameter(resnet.conv1.weight.mean(dim=1, keepdim=True))

        self.enc1_bn = resnet.bn1
        self.enc1_relu = resnet.relu
        self.enc1 = nn.Sequential(self.enc1_conv, self.enc1_bn, self.enc1_relu)  # 64
        self.enc2 = nn.Sequential(resnet.layer1)  # 64
        self.enc3 = nn.Sequential(resnet.layer2)  # 128
        self.enc4 = nn.Sequential(resnet.layer3)  # 256
        self.enc5 = nn.Sequential(resnet.layer4)  # 512

        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024)

        # Decoder (upsampling path)
        self.up5 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.ag5 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.dec5 = ConvBlock(1024, 512)

        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.ag4 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.dec4 = ConvBlock(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.ag3 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.dec3 = ConvBlock(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.ag2 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.dec2 = ConvBlock(128, 64)

        # Final convolution (e1 is already at the right spatial resolution after enc1's stride=2)
        self.final = nn.Conv2d(64, out_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)      # [B, 64, H/2, W/2]
        e2 = self.enc2(e1)     # [B, 64, H/2, W/2]
        e3 = self.enc3(e2)     # [B, 128, H/4, W/4]
        e4 = self.enc4(e3)     # [B, 256, H/8, W/8]
        e5 = self.enc5(e4)     # [B, 512, H/16, W/16]

        # Bottleneck
        b = self.bottleneck(e5)  # [B, 1024, H/16, W/16]

        # Decoder with attention gates
        d5 = self.up5(b)
        e5_att = self.ag5(d5, e5)
        d5 = torch.cat([d5, e5_att], dim=1)
        d5 = self.dec5(d5)

        d4 = self.up4(d5)
        e4_att = self.ag4(d4, e4)
        d4 = torch.cat([d4, e4_att], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        e3_att = self.ag3(d3, e3)
        d3 = torch.cat([d3, e3_att], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        e2_att = self.ag2(d2, e2)
        d2 = torch.cat([d2, e2_att], dim=1)
        d2 = self.dec2(d2)

        return self.final(d2)


def get_attention_unet_model(in_channels=1, out_classes=1):
    """
    Returns an Attention U-Net model.

    Attention U-Net features:
    - Attention gates in skip connections
    - Suppresses irrelevant regions and highlights important features
    - Better at focusing on target structures of varying size/shape
    """
    model = AttentionUNet(in_channels=in_channels, out_classes=out_classes)
    return model
