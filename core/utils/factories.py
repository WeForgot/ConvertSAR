import torch.nn as nn
import torch.optim as optim

from core.models.decoders import *
from core.models.encoders import *

ENCODER_REGISTRY = {
    'conv': ConvolutionalImageEncoder,
    'multi': MultiScaleConvolutionalImageEncoder,
    'vit': ViTImageEncoder,
    'saml': SAMLEncoder,
    'perc': PerceieverImageEncoder,
    'timm': TimmEncoder
}

DECODER_REGISTRY = {

    'cross': CrossAttentionDecoder,
    'rnn': RNNDecoder
}

OPTIMIZER_REGISTRY = {
    'adam': optim.Adam,
    'adamw': optim.AdamW,
    'sgd': optim.SGD,
}

def get_encoder(cfg) -> nn.Module:
    assert 'type' in cfg, "Encoder configuration must include 'type'."
    assert cfg['type'] in ENCODER_REGISTRY, f"Encoder type '{cfg['type']}' is not supported."
    assert 'args' in cfg, "Encoder configuration must include 'args'."
    encoder_cls = ENCODER_REGISTRY[cfg['type']]
    return encoder_cls(**cfg['args'])

def get_decoder(cfg) -> nn.Module:
    assert 'type' in cfg, "Decoder configuration must include 'type'."
    assert cfg['type'] in DECODER_REGISTRY, f"Decoder type '{cfg['type']}' is not supported."
    assert 'args' in cfg, "Decoder configuration must include 'args'."
    decoder_cls = DECODER_REGISTRY[cfg['type']]
    return decoder_cls(**cfg['args'])

def get_optimizer(cfg, params) -> optim.Optimizer:
    assert 'type' in cfg, "Optimizer configuration must include 'type'."
    assert cfg['type'] in OPTIMIZER_REGISTRY, f"Optimizer type '{cfg['type']}' is not supported."
    assert 'args' in cfg, "Optimizer configuration must include 'args'."
    
    optimizer_cls = OPTIMIZER_REGISTRY[cfg['type']]
    cfg['args']['lr'] = float(cfg['args']['lr'])
    cfg['args']['weight_decay'] = float(cfg['args']['weight_decay'])
    return optimizer_cls(params, **cfg['args'])