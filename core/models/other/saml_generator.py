import torch
import torch.nn as nn
import torch.nn.functional as F

from core.utils.utils import top_p

class SAMLGenerator(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        super(SAMLGenerator, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, xin, img, mask):
        context = self.encoder(img)
        cls_guess, col_guess, pos_guess = self.decoder(xin, context, mask)

        return cls_guess, col_guess, pos_guess
    
    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True

    def get_trainable_params(self):
        return filter(lambda p: p.requires_grad, self.parameters())
    
    @torch.no_grad()
    def generate(self, img, vocab, device, temperature=0.1):
        layers = [[float(vocab['<SOS>'])] + ([0.0] * 12)]
        context = self.encoder(img)
        while True:
            xin = torch.tensor(layers).to(device).unsqueeze(0)
            cls_out, col_out, pos_out = self.decoder(xin, context)
            cls_out, col_out, pos_out = cls_out[:,-1,:], col_out[:,-1,:], pos_out[:,-1,:]
            best_class = top_p(cls_out)
            probs = F.softmax(best_class / temperature, dim=-1)
            cls_token = torch.multinomial(probs, num_samples=1).item()
            if cls_token == vocab['<EOS>'] or len(layers) > 225:
                break
            col_out = col_out.squeeze(0).detach().cpu().tolist()
            pos_out = pos_out.squeeze(0).detach().cpu().tolist()
            layers.append([cls_token] + col_out + pos_out)
        return layers[1:]