import os

import torch
from PIL import Image
import timm

img = Image.open(os.path.join('output', 'base', '_Goto.png')).convert('RGB')

model = timm.create_model(
    'vit_large_patch14_dinov2.lvd142m',
    pretrained=True,
    num_classes=0,  # remove classifier nn.Linear
)
model = model.eval()

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

# or equivalently (without needing to set num_classes=0)

output = model.forward_features(transforms(img).unsqueeze(0))

torch.save(output, os.path.join('output', 'dino', '_Goto.pt'))