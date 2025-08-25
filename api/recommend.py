from typing import List, Dict
import os, json, logging, pickle
import numpy as np

from recs.data import DataStore
from recs.models.hybrid import HybridRanker

logger = logging.getLogger(__name__)

ART_DIR = os.environ.get("ARTIFACT_DIR", "artifacts")

class HybridRecommenderService:
    def __init__(self):
        self.store = DataStore(items_path="data/items.csv", interactions_path="data/interactions.csv")
        self.ranker = HybridRanker.load(ART_DIR, self.store)

    def recommend(self, user_id: str, k: int = 10, alpha: float = 0.6) -> list:
        if user_id not in self.store.user_to_idx:
            raise KeyError(f"Unknown user_id: {user_id}")
        # returns list of dicts: {item_id, score}
        return self.ranker.recommend(user_id, k=k, alpha=alpha)
