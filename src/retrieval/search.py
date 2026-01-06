import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from src.utils.query_normalizer import QueryNormalizer


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

        self.normalizer = QueryNormalizer()
        self.is_ip = isinstance(self.index, faiss.IndexFlatIP)

    def _score(self, raw):
        return float(raw) if self.is_ip else float(1 / (1 + raw))

    def search(self, query: str, top_k: int = 5):
        # 🔥 NORMALIZE QUERY
        normalized_query = self.normalizer.normalize(query)

        query_vec = self.embedder.encode(
            normalized_query,
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

            prev_chunk = self.chunks[idx - 1] if idx > 0 else None
            next_chunk = self.chunks[idx + 1] if idx + 1 < len(self.chunks) else None

            results.append({
                "score": score,
                "text": chunk["text_roman"],
                "title": chunk["title"],
                "start_hhmmss": chunk["start_hhmmss"],
                "end_hhmmss": chunk["end_hhmmss"],
                "play_url": chunk["play_url"],
                "previous": prev_chunk["text_roman"] if prev_chunk else "",
                "next": next_chunk["text_roman"] if next_chunk else "",
            })

        return results


if __name__ == "__main__":
    s = VectorSearcher()
    q = "Huruf e Muqattat kya hain?"
    r = s.search(q)

    for i, x in enumerate(r, 1):
        print(f"\n--- {i} ---")
        print("SCORE:", round(x["score"], 3))
        print("TEXT :", x["text"][:200])
