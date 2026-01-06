import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

FAISS_INDEX_PATH = "data/vector_store/faiss.index"
CHUNKS_PATH = "data/vector_store/chunks.pkl"


class VectorSearcher:
    def __init__(self):
        self.index = faiss.read_index(FAISS_INDEX_PATH)

        with open(CHUNKS_PATH, "rb") as f:
            self.chunks = pickle.load(f)

        self.embedder = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        self.is_ip = isinstance(self.index, faiss.IndexFlatIP)

    def _score(self, raw):
        return float(raw) if self.is_ip else float(1 / (1 + raw))

    def search(self, query: str, top_k: int = 20):
        query_vec = self.embedder.encode(
            query,
            normalize_embeddings=True
        ).astype("float32")

        scores, indices = self.index.search(
            np.array([query_vec]),
            top_k
        )

        results = []

        for rank, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue

            chunk = self.chunks[idx]
            score = self._score(scores[0][rank])

            results.append({
                "score": score,
                "text": chunk["text_roman"],
                "title": chunk["title"],
                "play_url": chunk["play_url"],
            })

        return results
