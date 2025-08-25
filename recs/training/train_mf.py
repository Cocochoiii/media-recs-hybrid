import os, argparse, pickle, random
import pandas as pd
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from tqdm import tqdm

from recs.data import DataStore
from recs.models.mf import MatrixFactorization, bpr_loss

def build_pairs(interactions, user_to_idx, item_to_idx, neg_ratio=4):
    pairs = []
    by_user = interactions.groupby('user_id')['item_id'].apply(list).to_dict()
    all_items = list(item_to_idx.keys())
    for u, items in by_user.items():
        uidx = user_to_idx[u]
        pos_set = set(items)
        for it in items:
            iidx = item_to_idx[it]
            for _ in range(neg_ratio):
                neg = random.choice(all_items)
                while neg in pos_set:
                    neg = random.choice(all_items)
                nidx = item_to_idx[neg]
                pairs.append((uidx, iidx, nidx))
    random.shuffle(pairs)
    return pairs

def main(args):
    store = DataStore(items_path="data/items.csv", interactions_path="data/interactions.csv")
    n_users = len(store.user_to_idx)
    n_items = len(store.item_to_idx)

    model = MatrixFactorization(n_users, n_items, args.factors)
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    pairs = build_pairs(store.interactions, store.user_to_idx, store.item_to_idx, neg_ratio=args.neg_ratio)

    for epoch in range(args.epochs):
        random.shuffle(pairs)
        losses = []
        for i in tqdm(range(0, len(pairs), args.batch_size)):
            batch = pairs[i:i+args.batch_size]
            u = torch.tensor([p[0] for p in batch], dtype=torch.long)
            pos = torch.tensor([p[1] for p in batch], dtype=torch.long)
            neg = torch.tensor([p[2] for p in batch], dtype=torch.long)
            pos_scores = model(u, pos)
            neg_scores = model(u, neg)
            loss = bpr_loss(pos_scores, neg_scores, reg=1e-5, model=model)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        print(f"epoch {epoch+1}: loss={np.mean(losses):.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.out_dir, "mf.pt"))
    with open(os.path.join(args.out_dir, "mf_map.pkl"), "wb") as f:
        pickle.dump({"n_users": n_users, "n_items": n_items, "n_factors": args.factors}, f)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--factors", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--neg_ratio", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--out_dir", type=str, default="artifacts")
    args = ap.parse_args()
    main(args)
