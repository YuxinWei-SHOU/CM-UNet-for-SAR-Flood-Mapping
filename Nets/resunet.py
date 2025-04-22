import torch
import torch.nn as nn

# Define a convolutional block with residual connection
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Add a 1x1 convolution layer for channel matching
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn_residual = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # Main path
        residual = self.residual_conv(x)
        residual = self.bn_residual(residual)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # Add the residual connection to the output
        out += residual
        out = self.relu(out)
        return out


class UpSampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSampling, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # Bilinear interpolation
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # Convolution
            nn.BatchNorm2d(out_channels),  # Batch normalization
            nn.ReLU(inplace=True)  # Activation function
        )

    def forward(self, x):
        return self.up(x)


# Define the U-Net model
class ResUNet(nn.Module):
    def __init__(self, num_classes=1, in_channels=2):
        super(ResUNet, self).__init__()

        # Define max pooling layer
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder part
        self.encoder1 = ConvBlock(in_channels, 64)  # Output: 256x256x64
        self.encoder2 = ConvBlock(64, 128)  # Output: 128x128x128
        self.encoder3 = ConvBlock(128, 256)  # Output: 64x64x256
        self.encoder4 = ConvBlock(256, 512)  # Output: 32x32x512

        # Decoder part
        self.upconv1 = UpSampling(512, 256)  # Upsample to: 64x64x256
        self.decoder1 = ConvBlock(512, 256)  # Output: 64x64x256

        self.upconv2 = UpSampling(256, 128)  # Upsample to: 128x128x128
        self.decoder2 = ConvBlock(256, 128)  # Output: 128x128x128

        self.upconv3 = UpSampling(128, 64)  # Upsample to: 256x256x64
        self.decoder3 = ConvBlock(128, 64)  # Output: 256x256x64

        # Final output layer
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)  # Output: 256x256x1

    def forward(self, x):
        # Encoder part
        e1 = self.encoder1(x)  # Output: 256x256x64
        p1 = self.max_pool(e1)  # Output: 128x128x64

        e2 = self.encoder2(p1)  # Output: 128x128x128
        p2 = self.max_pool(e2)  # Output: 64x64x128

        e3 = self.encoder3(p2)  # Output: 64x64x256
        p3 = self.max_pool(e3)  # Output: 32x32x256

        e4 = self.encoder4(p3)  # Output: 32x32x512

        # Decoder part
        d1 = self.upconv1(e4)  # Upsample to: 64x64x256
        d1 = torch.cat((d1, e3), dim=1)  # Skip connection: 64x64x512
        d1 = self.decoder1(d1)  # Output: 64x64x256

        d2 = self.upconv2(d1)  # Upsample to: 128x128x128
        d2 = torch.cat((d2, e2), dim=1)  # Skip connection: 128x128x256
        d2 = self.decoder2(d2)  # Output: 128x128x128

        d3 = self.upconv3(d2)  # Upsample to: 256x256x64
        d3 = torch.cat((d3, e1), dim=1)  # Skip connection: 256x256x128
        d3 = self.decoder3(d3)  # Output: 256x256x64

        out = self.out_conv(d3)  # Final output: 256x256x1
        return out
