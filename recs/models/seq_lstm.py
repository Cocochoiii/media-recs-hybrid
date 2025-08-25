import torch, torch.nn as nn

class SequenceLSTM(nn.Module):
    def __init__(self, n_items: int, embed_dim: int = 64, hidden: int = 128):
        super().__init__()
        self.embed = nn.Embedding(n_items, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden, batch_first=True)
        self.proj = nn.Linear(hidden, n_items)

    def forward(self, seq_idxs):
        x = self.embed(seq_idxs)
        out, _ = self.lstm(x)
        logits = self.proj(out[:, -1, :])  # next-item prediction on last step
        return logits
