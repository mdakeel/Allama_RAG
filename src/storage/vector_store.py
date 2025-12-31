import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from core.paths import data_path
from core.logging import logger

class VectorStore:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        logger.info(f"Loading embedding model: {model_name}")
        self.embedder = SentenceTransformer(model_name)
        self.index = None
        self.texts = []

    def build_index(self, transcripts_dir=data_path("transcripts")):
        files = [f for f in os.listdir(transcripts_dir) if f.endswith(".json")]
        all_embeddings = []
        for file in files:
            with open(os.path.join(transcripts_dir, file), "r", encoding="utf-8") as f:
                data = json.load(f)
                for seg in data["segments"]:
                    text = seg["text"]
                    self.texts.append(text)
                    emb = self.embedder.encode(text)
                    all_embeddings.append(emb)

        all_embeddings = np.array(all_embeddings).astype("float32")
        self.index = faiss.IndexFlatL2(all_embeddings.shape[1])
        self.index.add(all_embeddings)
        logger.info(f"Built FAISS index with {len(self.texts)} segments")

    def search(self, query: str, top_k=5):
        q_emb = self.embedder.encode(query).astype("float32").reshape(1, -1)
        D, I = self.index.search(q_emb, top_k)
        results = [self.texts[i] for i in I[0]]
        return results
