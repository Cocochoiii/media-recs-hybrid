import numpy as np
from recs.models.hybrid import HybridRanker
from recs.data import DataStore

def test_ranker_init():
    ds = DataStore("data/items.csv", "data/interactions.csv")
    # Cold start path (no artifacts): should still init
    ranker = HybridRanker.load("artifacts", ds)
    assert ranker is not None
