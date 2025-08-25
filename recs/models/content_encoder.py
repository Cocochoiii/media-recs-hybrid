import numpy as np
import logging

logger = logging.getLogger(__name__)

class ContentEncoder:
    def __init__(self):
        self.backend = None
        self.model = None
        self.vectorizer = None

    def fit(self, texts):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.backend = "sentence-transformers"
            return self.transform(texts)
        except Exception as e:
            logger.warning(f"SentenceTransformer unavailable, falling back to TF-IDF: {e}")
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            self.vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
            self.backend = "tfidf"
            return self.vectorizer.fit_transform(texts).astype(np.float32)

    def transform(self, texts):
        if self.backend == "sentence-transformers":
            return np.asarray(self.model.encode(texts, batch_size=64, show_progress_bar=False), dtype=np.float32)
        elif self.backend == "tfidf":
            return self.vectorizer.transform(texts).astype(np.float32)
        else:
            raise RuntimeError("ContentEncoder not fitted")
