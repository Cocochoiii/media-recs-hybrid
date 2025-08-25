import os, argparse, numpy as np, pandas as pd
from recs.models.content_encoder import ContentEncoder

def main(args):
    items = pd.read_csv("data/items.csv")
    texts = (items['title'].fillna('') + ' ' + items['desc'].fillna('')).tolist()
    enc = ContentEncoder()
    X = enc.fit(texts)
    if hasattr(X, "toarray"):
        X = X.toarray()
    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "item_embs.npy"), X)
    print(f"Saved item_embs.npy shape={X.shape}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="artifacts")
    args = ap.parse_args()
    main(args)
