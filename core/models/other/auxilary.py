import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, max_len, dim):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_len, dim)
        self.register_buffer("position_ids", torch.arange(max_len).expand((1, -1)))
    
    def forward(self, x):
        seq_len = x.shape[-2]
        position_ids = self.position_ids[:, :seq_len]
        return x + self.position_embeddings(position_ids)

class GEGLU(nn.Module):
    def __init__(self, dim, bias=True):
        super(GEGLU, self).__init__()
        self.in_proj = nn.Linear(in_features = dim, out_features = dim * 2, bias = bias)
        self.ln = nn.LayerNorm(normalized_shape = dim * 2)
        self.act = nn.GELU()
    
    def forward(self, x):
        x = self.in_proj(x)
        x = self.ln(x)
        x, y = x.chunk(2, dim=-1)
        x = x * self.act(y)
        return x

class DownscaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2):
        super(DownscaleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding='valid')
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x