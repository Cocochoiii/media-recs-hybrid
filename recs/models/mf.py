import torch, torch.nn as nn, torch.optim as optim
import numpy as np
from dataclasses import dataclass

class MatrixFactorization(nn.Module):
    def __init__(self, n_users: int, n_items: int, n_factors: int = 64):
        super().__init__()
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.item_factors = nn.Embedding(n_items, n_factors)
        nn.init.normal_(self.user_factors.weight, std=0.01)
        nn.init.normal_(self.item_factors.weight, std=0.01)

    def forward(self, user_idx, item_idx):
        u = self.user_factors(user_idx)
        v = self.item_factors(item_idx)
        return (u * v).sum(dim=1)

def bpr_loss(pos_scores, neg_scores, reg=1e-4, model=None):
    # Bayesian Personalized Ranking loss
    loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
    if model is not None:
        reg_term = 0.0
        for p in model.parameters():
            reg_term = reg_term + p.norm(2)
        loss = loss + reg * reg_term
    return loss
