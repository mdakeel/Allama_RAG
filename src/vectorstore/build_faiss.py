import pickle
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# paths
CHUNKS_PATH = "data/processed/chunks.pkl"
FAISS_INDEX_PATH = "data/vector_store/faiss.index"
CHUNKS_STORE_PATH = "data/vector_store/chunks.pkl"

# load chunks
with open(CHUNKS_PATH, "rb") as f:
    chunks = pickle.load(f)

print(f"Total chunks loaded: {len(chunks)}")

# load embedding model (CPU friendly)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embeddings = []

for chunk in tqdm(chunks, desc="Embedding chunks"):
    text = chunk["text_roman"]
    vec = model.encode(text, normalize_embeddings=True)
    embeddings.append(vec)

embeddings = np.array(embeddings).astype("float32")

print("Embedding shape:", embeddings.shape)

# build FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)  # cosine similarity
index.add(embeddings)

print("FAISS index size:", index.ntotal)

# save index
faiss.write_index(index, FAISS_INDEX_PATH)

# save chunks (metadata)
with open(CHUNKS_STORE_PATH, "wb") as f:
    pickle.dump(chunks, f)

print("FAISS index & chunks saved successfully")
