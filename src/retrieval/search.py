# # # import pickle
# # # import faiss
# # # import numpy as np
# # # from sentence_transformers import SentenceTransformer

# # # FAISS_INDEX_PATH = "data/vector_store/faiss.index"
# # # CHUNKS_PATH = "data/vector_store/chunks.pkl"


# # # class VectorSearcher:
# # #     def __init__(self):
# # #         self.index = faiss.read_index(FAISS_INDEX_PATH)

# # #         with open(CHUNKS_PATH, "rb") as f:
# # #             self.chunks = pickle.load(f)

# # #         self.embedder = SentenceTransformer(
# # #             "sentence-transformers/all-MiniLM-L6-v2"
# # #         )

# # #         self.is_ip = isinstance(self.index, faiss.IndexFlatIP)

# # #     def _score(self, raw):
# # #         return float(raw) if self.is_ip else float(1 / (1 + raw))

# # #     def search(self, query: str, top_k: int = 20):
# # #         query_vec = self.embedder.encode(
# # #             query,
# # #             normalize_embeddings=True
# # #         ).astype("float32")

# # #         scores, indices = self.index.search(
# # #             np.array([query_vec]),
# # #             top_k
# # #         )

# # #         results = []

# # #         for rank, idx in enumerate(indices[0]):
# # #             if idx < 0 or idx >= len(self.chunks):
# # #                 continue

# # #             chunk = self.chunks[idx]
# # #             score = self._score(scores[0][rank])

# # #             results.append({
# # #                 "score": score,
# # #                 "text": chunk["text_roman"],
# # #                 "title": chunk["title"],
# # #                 "play_url": chunk["play_url"],
# # #             })

# # #         return results


# # # src/retrieval/search.py
# # import pickle
# # import faiss
# # import numpy as np
# # from sentence_transformers import SentenceTransformer


# # class FaissSearcher:
# #     def __init__(
# #         self,
# #         index_path="data/vector_store/faiss.index",
# #         chunks_path="data/vector_store/chunks.pkl",
# #         model_name="intfloat/multilingual-e5-base",
# #     ):
# #         self.index = faiss.read_index(index_path)

# #         with open(chunks_path, "rb") as f:
# #             self.chunks = pickle.load(f)

# #         self.model = SentenceTransformer(model_name)

# #     def search(self, query: str, top_k: int = 40):
# #         query = "query: " + query
# #         q_emb = self.model.encode(
# #             query,
# #             normalize_embeddings=True
# #         ).astype("float32")

# #         scores, indices = self.index.search(
# #             np.expand_dims(q_emb, axis=0),
# #             top_k
# #         )

# #         results = []
# #         for score, idx in zip(scores[0], indices[0]):
# #             if idx == -1:
# #                 continue
# #             item = dict(self.chunks[idx])
# #             item["score"] = float(score)
# #             results.append(item)

# #         return results



# import pickle
# import faiss
# import numpy as np
# from sentence_transformers import SentenceTransformer


# class FaissSearcher:
#     def __init__(
#         self,
#         index_path="data/vector_store/faiss.index",
#         chunks_path="data/vector_store/chunks.pkl",
#         model_name="intfloat/multilingual-e5-large",  # 🔥 large
#     ):
#         self.index = faiss.read_index(index_path)

#         with open(chunks_path, "rb") as f:
#             self.chunks = pickle.load(f)

#         self.model = SentenceTransformer(model_name)

#     def search(self, query: str, top_k: int = 15):
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
#         for score, idx in zip(scores[0], indices[0]):
#             if idx == -1 or score < 0.35:  # 🔒 noise cut
#                 continue

#             item = dict(self.chunks[idx])
#             item["score"] = float(score)
#             results.append(item)

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
    ):
        self.index = faiss.read_index(index_path)

        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)

        # ⚠ MUST MATCH INDEX MODEL
        self.model = SentenceTransformer(
            "intfloat/multilingual-e5-large"
        )

        # safety
        assert self.index.d == self.model.get_sentence_embedding_dimension()

    def search(self, query: str, top_k: int = 8):
        query = "query: " + query

        q_emb = self.model.encode(
            query,
            normalize_embeddings=True
        ).astype("float32")

        scores, indices = self.index.search(
            np.expand_dims(q_emb, axis=0),
            top_k
        )

        results = []
        for rank, idx in enumerate(indices[0]):
            if idx == -1:
                continue

            chunk = dict(self.chunks[idx])
            chunk["score"] = float(scores[0][rank])
            chunk["index"] = idx   # 🔥 THIS WAS MISSING
            results.append(chunk)

        return results
