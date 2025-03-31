import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIPTraininer(nn.Module):
    def __init__(self, image_encoder: nn.Module, saml_encoder: nn.Module, final_dim: int = 512):
        super(CLIPTraininer, self).__init__()
        self.image_encoder = image_encoder
        self.saml_encoder = saml_encoder
        self.image_proj = nn.Linear(image_encoder.dim, final_dim)
        self.saml_proj = nn.Linear(saml_encoder.dim, final_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * 0.7)
    
    def forward(self, images, sequences, mask=None):
        image_features = self.image_encoder(images)  # no padding needed here
        saml_features = self.saml_encoder(sequences, padding_mask=~mask if mask is not None else None)
        
        # Apply masking properly before taking mean
        image_features = image_features.mean(dim=1)
        
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            saml_features = (saml_features * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            saml_features = saml_features.mean(dim=1)

        image_features = self.image_proj(image_features)
        saml_features = self.saml_proj(saml_features)

        image_features = F.normalize(image_features, dim=-1)
        saml_features = F.normalize(saml_features, dim=-1)

        logits_along_batch = (image_features @ saml_features.t()) * self.logit_scale.exp()
        return logits_along_batch