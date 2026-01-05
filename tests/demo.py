import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

INDEX_PATH = "E:/ML-Projects/Allama/data/vector_store/faiss_index"
TEXTS_PATH = "E:/ML-Projects/Allama/data/vector_store/texts.pkl"

# Load texts
with open(TEXTS_PATH, "rb") as f:
    texts = pickle.load(f)

# Load model + index
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
index = faiss.read_index(INDEX_PATH)

queries = [
    "iman walon kon hain",
    "موسیٰ اور فرعون کی کہانی",
    "story of Moses and Pharaoh"
]

for q in queries:
    print(f"\n🔎 Query: {q}")
    q_emb = model.encode([q], convert_to_numpy=True).astype("float32")
    D, I = index.search(q_emb, k=5)
    for rank, idx in enumerate(I[0]):
        if idx >= 0 and idx < len(texts):
            print(f"  Result {rank+1}: {texts[idx]} (distance={D[0][rank]:.4f})")
