from einops.layers.torch import Rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as T

class DINOImageEncoder(nn.Module):
    def __init__(self, dim: int):
        super(DINOImageEncoder, self).__init__()
        self.transform = T.Compose([
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.Resize((224, 224)),
        ])
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
        self.proj = nn.Linear(384, dim)

    
    def forward(self, x, pool: bool = False):
        x = self.transform(x)
        x = self.model.forward_features(x)['x_prenorm']
        x = self.proj(x)
        return x