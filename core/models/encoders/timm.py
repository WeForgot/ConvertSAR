import torch.nn as nn

import timm

class TimmEncoder(nn.Module):
    def __init__(self, model_name: str, dim: int):
        super(TimmEncoder, self).__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
        )
        data_config = timm.data.resolve_model_data_config(self.model)
        self.transforms = timm.data.create_transform(**data_config, is_training=True)
        self.proj_out = nn.Linear(in_features=768, out_features=dim, bias=False)
    
    def forward(self, x):
        x = self.model.forward_features(self.transforms(x))
        x = self.proj_out(x)
        return x