import torch
import torch.nn as nn
import torch.nn.functional as F

from core.models.other.auxilary import PositionalEncoding, GEGLU

class RNNDecoder(nn.Module):
    def __init__(self, vocab_size, vocab_dim, input_dim, dim, num_layers, max_saml_layers) -> None:
        super(RNNDecoder, self).__init__()
        self.layer_embs = nn.Embedding(vocab_size, vocab_dim)
        self.projection = nn.Linear(self.layer_embs.weight.shape[-1] + 4 + 8, dim)
        self.ctx_proj = nn.Linear(input_dim, dim) if input_dim != dim else nn.Identity()
        self.pos_emb = PositionalEncoding(max_saml_layers, dim)
        self.num_cls = vocab_size

        self.rnn = nn.RNN(
            input_size=dim,
            hidden_size=dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1,
        )

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
    
    def forward(self, tgt, context, *args, **kwargs):
        context_len = context.shape[1]
        x, y = torch.split(tgt, [1, 12], dim=-1)
        x = self.layer_embs(x.long()).squeeze(2)
        tgt = torch.cat([x, y], dim=-1)
        x = self.projection(tgt)
        
        x = self.pos_emb(x)

        context = self.ctx_proj(context)
        x = torch.cat([context, x], dim=1)
        x, _ = self.rnn(x)
        x = x[:, context_len:, :]
        cls_out, col_out, pos_out = self.cls_out(x), self.col_out(x), self.pos_out(x)
        return cls_out, col_out, pos_out