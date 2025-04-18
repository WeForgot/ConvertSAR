from einops.layers.torch import Rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.transforms import Resize

class PerceieverImageEncoder(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_layers: int, image_size: int, patch_size: int, final_dim: int, num_channels: int = 4, memory_tokens: int = 64):
        super(PerceieverImageEncoder, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.final_dim = final_dim

        self.memory_tokens = nn.Parameter(torch.randn(1, memory_tokens, dim))


        assert image_size % patch_size == 0, "Image size must be divisible by patch size"
        self.resize = Resize((image_size, image_size))
        self.num_patches = (image_size // patch_size) ** 2
        self.to_patches = Rearrange('b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=patch_size, p2=patch_size)
        self.patch_proj = nn.Linear(num_channels * patch_size * patch_size, dim)
        self.patch_norm = nn.LayerNorm(dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, dim))
        self.dropout = nn.Dropout(0.1)

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.resize(x)
        x = self.to_patches(x)
        x = self.patch_proj(x)
        x = self.patch_norm(x)
        x = x + self.pos_embedding
        x = self.dropout(x)
        x = self.transformer_encoder(x)
        x = self.layer_norm(x)
        return x