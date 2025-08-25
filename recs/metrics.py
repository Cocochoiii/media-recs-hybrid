import numpy as np

def precision_at_k(recommended, ground_truth, k=10):
    if not ground_truth:
        return 0.0
    rec_k = recommended[:k]
    hits = len(set(rec_k) & set(ground_truth))
    return hits / k

def recall_at_k(recommended, ground_truth, k=10):
    if not ground_truth:
        return 0.0
    rec_k = recommended[:k]
    hits = len(set(rec_k) & set(ground_truth))
    return hits / len(ground_truth)

def ndcg_at_k(recommended, ground_truth, k=10):
    rec_k = recommended[:k]
    dcg = 0.0
    for i, item in enumerate(rec_k):
        if item in ground_truth:
            dcg += 1.0 / np.log2(i + 2)
    idcg = sum(1.0/np.log2(i+2) for i in range(min(len(ground_truth), k)))
    return dcg / idcg if idcg > 0 else 0.0
