import os, pickle, logging
import numpy as np
from typing import List, Dict
from sklearn.neighbors import NearestNeighbors

import torch

from .mf import MatrixFactorization
from recs.data import DataStore

logger = logging.getLogger(__name__)

class HybridRanker:
    def __init__(self, store: DataStore, mf_model, item_embs: np.ndarray):
        self.store = store
        self.mf = mf_model
        self.item_embs = item_embs
        self.nn = NearestNeighbors(metric='cosine', algorithm='brute')
        self.nn.fit(item_embs)

    @staticmethod
    def load(art_dir: str, store: DataStore):
        # Load MF
        mf_path = os.path.join(art_dir, "mf.pt")
        map_path = os.path.join(art_dir, "mf_map.pkl")
        embs_path = os.path.join(art_dir, "item_embs.npy")

        if not (os.path.exists(mf_path) and os.path.exists(map_path) and os.path.exists(embs_path)):
            logger.warning("Artifacts missing; using cold-start baseline (content-only).")
            # Cold start: content-only ranker
            from .content_encoder import ContentEncoder
            import pandas as pd, numpy as np
            texts = (store.items['title'].fillna('') + ' ' + store.items['desc'].fillna('')).tolist()
            enc = ContentEncoder()
            item_embs = enc.fit(texts)
            if hasattr(item_embs, "toarray"):
                item_embs = item_embs.toarray()
            dummy = MatrixFactorization(len(store.user_to_idx), len(store.item_to_idx), 16)
            return HybridRanker(store, dummy, np.asarray(item_embs))

        with open(map_path, "rb") as f:
            mapping = pickle.load(f)
        mf = MatrixFactorization(mapping['n_users'], mapping['n_items'], mapping['n_factors'])
        mf.load_state_dict(torch.load(mf_path, map_location="cpu"))
        mf.eval()

        item_embs = np.load(embs_path)
        return HybridRanker(store, mf, item_embs)

    def recommend(self, user_id: str, k: int = 10, alpha: float = 0.6):
        uid = self.store.user_to_idx[user_id]
        # collaborative scores: dot with item factors
        with torch.no_grad():
            uvec = self.mf.user_factors.weight[uid].detach().cpu().numpy()
            ivecs = self.mf.item_factors.weight.detach().cpu().numpy()
            cf_scores = ivecs @ uvec

        # content similarity to user's historically liked items (centroid)
        seen = list(self.store.get_user_items(user_id))
        if seen:
            idxs = [self.store.item_to_idx[i] for i in seen if i in self.store.item_to_idx]
            centroid = self.item_embs[idxs].mean(axis=0, keepdims=True)
            dists, neighbors = self.nn.kneighbors(centroid, n_neighbors=min(k*10, self.item_embs.shape[0]))
            content_scores = 1.0 - dists.flatten()
            candidates = neighbors.flatten()
            # build content score array aligned to item index
            cs = np.zeros(self.item_embs.shape[0], dtype=np.float32)
            cs[candidates] = content_scores
        else:
            cs = np.zeros(self.item_embs.shape[0], dtype=np.float32)

        # combine
        scores = alpha * cf_scores + (1 - alpha) * cs

        # filter seen
        seen_idx = set(self.store.item_to_idx[i] for i in seen if i in self.store.item_to_idx)
        ranked = [i for i in np.argsort(-scores) if i not in seen_idx][:k]
        return [{"item_id": self.store.idx_to_item[i], "score": float(scores[i])} for i in ranked]
