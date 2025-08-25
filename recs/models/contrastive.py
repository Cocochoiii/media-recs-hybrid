import torch, torch.nn as nn, torch.nn.functional as F

class ContrastiveAligner(nn.Module):
    def __init__(self, dim=64, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.user_proj = nn.Linear(dim, dim)
        self.item_proj = nn.Linear(dim, dim)

    def forward(self, user_vecs, item_vecs):
        u = F.normalize(self.user_proj(user_vecs), dim=-1)
        v = F.normalize(self.item_proj(item_vecs), dim=-1)
        logits = u @ v.t() / self.temperature
        labels = torch.arange(u.size(0), device=u.device)
        loss = F.cross_entropy(logits, labels)
        return loss
