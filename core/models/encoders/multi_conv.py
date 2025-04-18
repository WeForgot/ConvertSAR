from einops.layers.torch import Rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.models.other.auxilary import DownscaleBlock, PositionalEncoding, MultiScaleBlock

class MultiScaleConvolutionalImageEncoder(nn.Module):
    def __init__(self, dim, num_heads, num_layers, num_channels, final_dim):
        super(MultiScaleConvolutionalImageEncoder, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.final_dim = final_dim

        self.conv0 = nn.Conv2d(in_channels=num_channels, out_channels=8, kernel_size=1, stride=1)
        self.bn0 = nn.BatchNorm2d(8)
        self.act0 = nn.SiLU()

        self.multiscale1 = MultiScaleBlock(8, [3, 5, 7])
        self.downscale1 = DownscaleBlock(8, 16, kernel_size=5)
        self.multiscale2 = MultiScaleBlock(16, [3, 5, 7])
        self.downscale2 = DownscaleBlock(16, 32, kernel_size=5)
        self.multiscale3 = MultiScaleBlock(32, [3, 5, 7])
        self.downscale3 = DownscaleBlock(32, 64, kernel_size=3)
        self.multiscale4 = MultiScaleBlock(64, [3, 5, 7])
        self.downscale4 = DownscaleBlock(64, 128, kernel_size=3)
        self.proj_out = nn.Conv2d(in_channels=128, out_channels=dim, kernel_size=1, stride=1)

        self.to_patches = Rearrange('b c h w -> b (h w) c')
        self.pos_embedding = PositionalEncoding(196, dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.act0(x)

        x = self.multiscale1(x)
        x = self.downscale1(x)
        x = self.multiscale2(x)
        x = self.downscale2(x)
        x = self.multiscale3(x)
        x = self.downscale3(x)
        x = self.multiscale4(x)
        x = self.downscale4(x)
        x = self.proj_out(x)

        x = self.to_patches(x)
        x = self.pos_embedding(x)
        x = self.transformer_encoder(x)
        x = self.layer_norm(x)
        return x