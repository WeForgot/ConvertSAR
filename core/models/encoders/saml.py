import torch
import torch.nn as nn
import torch.nn.functional as F

from core.models.other.auxilary import PositionalEncoding

class SAMLEncoder(nn.Module):
    def __init__(self, dim, num_heads, num_layers, vocab_size, vocab_dim, max_saml_layers):
        super(SAMLEncoder, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.vocab_dim = vocab_dim
        self.max_saml_layers = max_saml_layers

        self.emb_layer = nn.Embedding(vocab_size, vocab_dim)
        self.proj_in = nn.Linear(vocab_dim + 4 + 8, dim)
        self.pos_embedding = PositionalEncoding(max_saml_layers, dim)
        self.dropout = nn.Dropout(0.1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(dim)
    
    def forward(self, x, padding_mask=None):
        x, y = torch.split(x, [1, 12], dim=-1)
        x = x.long().squeeze(2)
        x = self.emb_layer(x.long())
        x = torch.cat([x, y], dim=-1)
        x = self.proj_in(x)
        x = self.pos_embedding(x)
        x = self.dropout(x)
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(x.device)
        x = self.transformer_encoder(x, mask=mask, src_key_padding_mask=padding_mask, is_causal=True)
        x = self.layer_norm(x)
        return x