import torch
import torch.nn as nn
import torch.nn.functional as F

from core.models.other.auxilary import PositionalEncoding, GEGLU

class CrossAttentionDecoder(nn.Module):
    def __init__(self, vocab_size, vocab_dim, input_dim, dim, num_layers, num_heads, max_saml_layers) -> None:
        super(CrossAttentionDecoder, self).__init__()
        self.layer_embs = nn.Embedding(vocab_size, vocab_dim)
        self.projection = nn.Linear(self.layer_embs.weight.shape[-1] + 4 + 8, dim)
        self.pos_emb = PositionalEncoding(max_saml_layers, dim)
        self.ctx_proj = nn.Linear(input_dim, dim) if input_dim != dim else nn.Identity()
        dec_layer = nn.TransformerDecoderLayer(dim, num_heads, dim_feedforward=min(4*dim, 2048), activation=F.gelu, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers)
        self.num_cls = vocab_size

        self.cls_out = nn.Sequential(
            GEGLU(dim),
            nn.Dropout(p=0.1),
            nn.Linear(dim, vocab_dim)
        )

        self.col_out = nn.Sequential(
            GEGLU(dim),
            nn.Dropout(p=0.1),
            nn.Linear(dim, 4)
        )

        self.pos_out = nn.Sequential(
            GEGLU(dim),
            nn.Dropout(p=0.1),
            nn.Linear(dim, 8)
        )
    
    def forward(self, tgt, context, padding_mask=None):
        x, y = torch.split(tgt, [1, 12], dim=-1)
        x = self.layer_embs(x.long()).squeeze(2)
        tgt = torch.cat([x, y], dim=-1)
        x = self.projection(tgt)

        x = self.pos_emb(x)

        mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1]).to(x.device)
        padding_mask = ~padding_mask if padding_mask is not None and padding_mask.dtype == torch.bool else padding_mask

        context = self.ctx_proj(context)
        x = self.decoder(x, context, tgt_mask=mask, tgt_is_causal=True, tgt_key_padding_mask=padding_mask)
        cls_out, col_out, pos_out = self.cls_out(x), self.col_out(x), self.pos_out(x)
        cls_out = torch.matmul(cls_out, self.layer_embs.weight.T)
        return cls_out, col_out, pos_out