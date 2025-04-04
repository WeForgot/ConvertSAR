import torch
import torch.nn as nn
import torch.nn.functional as F

from core.models.other.auxilary import PositionalEncoding, GEGLU

class CrossAttentionDecoder(nn.Module):
    def __init__(self, vocab_size, vocab_dim, dim, num_layers, num_heads, max_saml_layers) -> None:
        super(CrossAttentionDecoder, self).__init__()
        self.layer_embs = nn.Embedding(vocab_size, vocab_dim)
        self.projection = nn.Linear(self.layer_embs.weight.shape[-1] + 4 + 8, dim)
        self.pos_emb = PositionalEncoding(max_saml_layers, dim)
        dec_layer = nn.TransformerDecoderLayer(dim, num_heads, dim_feedforward=min(4*dim, 2048), activation=F.gelu, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers)
        self.num_cls = vocab_size

        self.cls_out = nn.Sequential(
            GEGLU(dim),
            nn.Dropout(p=0.1),
            nn.Linear(dim, self.num_cls)
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

        # Note: Our padding mask uses False for padded positions and True for valid tokens.
        # This is the reverse of the conventional approach, but it allows us to use the same mask
        # for both loss calculation and attention masking.
        padding_mask = torch.where(padding_mask == False, 0, float('-inf')) if padding_mask is not None and padding_mask.dtype == torch.bool else padding_mask

        x = self.decoder(x, context, tgt_mask=mask, tgt_is_causal=True, tgt_key_padding_mask=padding_mask)
        cls_out, col_out, pos_out = self.cls_out(x), self.col_out(x), self.pos_out(x)
        return cls_out, col_out, pos_out