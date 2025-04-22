import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

# Define convolution block
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # Bilinear interpolation
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # Convolution
            nn.BatchNorm2d(out_channels),  # Batch normalization
            nn.ReLU(inplace=True)  # Activation function
        )

    def forward(self, x):
        return self.up(x)

# Define Patch Embedding
class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x

# Define Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, drop=0., attn_drop=0.):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, bias=qkv_bias, dropout=attn_drop)
        self.drop_path = nn.Dropout(drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0])
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# Define Vision Transformer
class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim, depth, num_heads, mlp_ratio=4.0):
        super(VisionTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, (img_size // patch_size) ** 2, embed_dim))
        self.pos_drop = nn.Dropout(p=0.)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x

# Define MambaUNet (a transformer-based U-Net architecture)
class TransUNet(nn.Module):
    def __init__(self, num_classes=1, in_channels=4, img_size=256, embed_dim=512, depth=12, num_heads=8):
        super(TransUNet, self).__init__()

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.vit = VisionTransformer(img_size=64, patch_size=16, in_channels=256, embed_dim=embed_dim, depth=depth, num_heads=num_heads)
        self.encoder1 = ConvBlock(in_channels, 64)
        self.encoder2 = ConvBlock(64, 128)
        self.encoder3 = ConvBlock(128, 256)

        self.upconv3 = UpConv(512, 256)
        self.decoder3 = ConvBlock(512, 256)

        self.upconv2 = UpConv(256, 128)
        self.decoder2 = ConvBlock(256, 128)

        self.upconv1 = UpConv(128, 64)
        self.decoder1 = ConvBlock(128, 64)


        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.encoder1(x)  # Output: 256x256x64
        p1 = self.max_pool(e1)  # Output: 128x128x64

        e2 = self.encoder2(p1)  # Output: 128x128x128
        p2 = self.max_pool(e2)  # Output: 64x64x128

        e3 = self.encoder3(p2)  # Output: 64x64x256

        # Input e3 into Vision Transformer
        vit_out = self.vit(e3)

        # Reshape output
        b, n, c = vit_out.size()
        h = w = int(n ** 0.5)
        vit_out = vit_out.view(b, c, h, w)

        vit_out = F.interpolate(vit_out, size=(32, 32), mode='bilinear', align_corners=False)

        # Decoder part
        d3 = self.upconv3(vit_out)  # Upsampling: 64x64x256 /32x32x256
        d3 = torch.cat((d3, e3), dim=1)  # Skip connection: 64x64x512 /32x32x(256+256)
        d3 = self.decoder3(d3)  # Feature extraction: 64x64x256 /32x32x256

        d2 = self.upconv2(d3)  # Upsampling: 128x128x128 /64x64x128
        d2 = torch.cat((d2, e2), dim=1)  # Skip connection: 128x128x256 /64x64x(128+128)
        d2 = self.decoder2(d2)  # Feature extraction: 128x128x128 /64x64x128

        d1 = self.upconv1(d2)  # Upsampling: 256x256x64 / 128x128x64
        d1 = torch.cat((d1, e1), dim=1)  # Skip connection: 256x256x128 / 128x128x(64+64)
        d1 = self.decoder1(d1)  # Feature extraction: 256x256x64 /128x128x64

        out = self.out_conv(d1)  # Final output: 256x256x1
        return out
