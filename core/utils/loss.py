import torch
import torch.nn as nn

class HomoscedasticMultiTaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # log(sigma^2) for each task
        self.log_var_col = nn.Parameter(torch.zeros(()))
        self.log_var_pos = nn.Parameter(torch.zeros(()))

    def forward(self, L_cls, L_col, L_pos):
        # classification gets weight 1 (or you can learn it too)
        loss = L_cls  
        # regression heads get scaled by 1/(2σ²) + log σ
        loss += 0.5 * torch.exp(-self.log_var_col) * L_col \
             + 0.5 * self.log_var_col
        loss += 0.5 * torch.exp(-self.log_var_pos) * L_pos \
             + 0.5 * self.log_var_pos
        return loss