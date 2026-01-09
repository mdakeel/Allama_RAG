

# import pickle
# import faiss
# import numpy as np
# from sentence_transformers import SentenceTransformer


# class FaissSearcher:
#     def __init__(
#         self,
#         index_path="data/vector_store/faiss.index",
#         chunks_path="data/vector_store/chunks.pkl",
#     ):
#         self.index = faiss.read_index(index_path)

#         with open(chunks_path, "rb") as f:
#             self.chunks = pickle.load(f)

#         # ⚠ MUST MATCH INDEX MODEL
#         self.model = SentenceTransformer(
#             "intfloat/multilingual-e5-large"
#         )

#         # safety
#         assert self.index.d == self.model.get_sentence_embedding_dimension()

#     def search(self, query: str, top_k: int = 8):
#         query = "query: " + query

#         q_emb = self.model.encode(
#             query,
#             normalize_embeddings=True
#         ).astype("float32")

#         scores, indices = self.index.search(
#             np.expand_dims(q_emb, axis=0),
#             top_k
#         )

#         results = []
#         for rank, idx in enumerate(indices[0]):
#             if idx == -1:
#                 continue

#             chunk = dict(self.chunks[idx])
#             chunk["score"] = float(scores[0][rank])
#             chunk["index"] = idx   # 🔥 THIS WAS MISSING
#             results.append(chunk)

#         return results




import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class FaissSearcher:
    def __init__(
        self,
        index_path="data/vector_store/faiss.index",
        chunks_path="data/vector_store/chunks.pkl",
        model_name="intfloat/multilingual-e5-large"
    ):
        self.index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)

        self.model = SentenceTransformer(model_name)
        assert self.index.d == self.model.get_sentence_embedding_dimension()

    def search(self, query: str, top_k: int = 15):
        query = "query: " + query
        q_emb = self.model.encode(query, normalize_embeddings=True).astype("float32")

        scores, indices = self.index.search(np.expand_dims(q_emb, axis=0), top_k)

        results = []
        for rank, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(scores[0][rank])
            chunk["index"] = idx
            results.append(chunk)

        return results
