from recs.data import DataStore

def test_datastore_loads():
    ds = DataStore("data/items.csv", "data/interactions.csv")
    assert len(ds.items) > 0
    assert len(ds.interactions) > 0
