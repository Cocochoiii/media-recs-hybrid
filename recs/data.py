import pandas as pd
import numpy as np
from typing import Dict, Tuple

class DataStore:
    def __init__(self, items_path: str, interactions_path: str):
        self.items = pd.read_csv(items_path)
        self.interactions = pd.read_csv(interactions_path)

        # Create id->idx mappings
        users = sorted(self.interactions['user_id'].unique())
        items = sorted(self.items['item_id'].unique())

        self.user_to_idx = {u:i for i,u in enumerate(users)}
        self.idx_to_user = {i:u for u,i in self.user_to_idx.items()}

        self.item_to_idx = {it:i for i,it in enumerate(items)}
        self.idx_to_item = {i:it for it,i in self.item_to_idx.items()}

    def get_user_items(self, user_id: str):
        return set(self.interactions[self.interactions['user_id']==user_id]['item_id'].tolist())
