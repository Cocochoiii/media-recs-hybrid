# Minimal placeholder to illustrate sequence model training; fine-tune for your data.
import argparse, os
import pandas as pd, numpy as np
import torch, torch.nn as nn, torch.optim as optim
from recs.models.seq_lstm import SequenceLSTM
from recs.data import DataStore

def main(args):
    store = DataStore("data/items.csv", "data/interactions.csv")
    n_items = len(store.item_to_idx)
    model = SequenceLSTM(n_items=n_items, embed_dim=64, hidden=128)
    opt = optim.Adam(model.parameters(), lr=1e-3)

    # Build short sequences per user (toy example)
    seqs = []
    for u, df in store.interactions.groupby('user_id'):
        items = [store.item_to_idx[i] for i in df.sort_values('ts')['item_id'].tolist() if i in store.item_to_idx]
        if len(items) >= 5:
            seqs.append(items[:10])
    if not seqs:
        print("Not enough sequences; skipping.")
        return

    pad = max(len(s) for s in seqs)
    x = [s + [0]*(pad-len(s)) for s in seqs]
    x = torch.tensor(x, dtype=torch.long)

    for epoch in range(args.epochs):
        logits = model(x)
        targets = x[:, -1]
        loss = nn.CrossEntropyLoss()(logits, targets)
        opt.zero_grad(); loss.backward(); opt.step()
        print("epoch", epoch+1, "loss", float(loss))

    os.makedirs("artifacts", exist_ok=True)
    torch.save(model.state_dict(), "artifacts/seq.pt")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    args = ap.parse_args()
    main(args)
