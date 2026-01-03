# src/vector_store.py
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from src.core.paths import data_path
from src.core.logging import logger
from src.utils.helpers import ensure_dir

class VectorStore:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        logger.info(f"Loading embedding model: {model_name}")
        self.embedder = SentenceTransformer(model_name)
        self.index = None
        self.texts = []

    def _split_long_text(self, raw_text: str):
        if not isinstance(raw_text, str):
            return []
        # Normalize whitespace
        text = " ".join(raw_text.split())
        # Split on Urdu full stop "۔" and also English "." and line breaks
        chunks = []
        for piece in text.replace("\n", " ").split("۔"):
            piece = piece.strip()
            if not piece:
                continue
            # Further split on '.' if present
            subpieces = [p.strip() for p in piece.split(".") if p.strip()]
            if subpieces:
                chunks.extend(subpieces)
            else:
                chunks.append(piece)
        # Optional: merge very short chunks to avoid too small embeddings
        merged = []
        buf = ""
        for c in chunks:
            if len(buf) + len(c) < 200:  # adjust threshold as needed
                buf = (buf + " " + c).strip()
            else:
                if buf:
                    merged.append(buf)
                buf = c
        if buf:
            merged.append(buf)
        return merged

    def build_index(self, transcripts_dir=data_path("transcripts")):
        files = [f for f in os.listdir(transcripts_dir) if f.endswith(".json")]
        all_embeddings = []

        for file in files:
            file_path = os.path.join(transcripts_dir, file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load {file}: {e}")
                continue

            entries = []

            # Case 1: Whisper-style or segment list with text_roman
            if "segments" in data and isinstance(data["segments"], list):
                entries = [
                    seg.get("text_roman") or seg.get("text") or ""  # prefer text_roman
                    for seg in data["segments"]
                    if isinstance(seg, dict)
                ]
                entries = [e.strip() for e in entries if e and e.strip()]

            # Case 2: List of strings under Transcripts
            elif "Transcripts" in data and isinstance(data["Transcripts"], list):
                entries = [item.strip() for item in data["Transcripts"] if isinstance(item, str) and item.strip()]

            # Case 3: Single long transcript string
            elif "transcript_roman" in data and isinstance(data["transcript_roman"], str):
                entries = self._split_long_text(data["transcript_roman"])

            else:
                logger.warning(f"Skipping {file}: no valid transcript key found")
                continue

            if not entries:
                logger.warning(f"No text entries found in {file}")
                continue

            # Batch encode for speed
            try:
                embs = self.embedder.encode(entries, convert_to_numpy=True)
            except Exception as e:
                logger.error(f"Embedding failed for {file}: {e}")
                continue

            self.texts.extend(entries)
            all_embeddings.append(embs)

        if not all_embeddings:
            logger.error("No embeddings created. Check transcript JSON format.")
            return

        all_embeddings = np.concatenate(all_embeddings, axis=0).astype("float32")
        self.index = faiss.IndexFlatL2(all_embeddings.shape[1])
        self.index.add(all_embeddings)
        logger.info(f"Built FAISS index with {len(self.texts)} segments")

    def save_index(self, path: str):
        if self.index is None:
            logger.error("Index not built. Cannot save.")
            return
        ensure_dir(os.path.dirname(path))
        faiss.write_index(self.index, path)
        logger.info(f"Saved FAISS index to {path}")

    def load_index(self, path: str):
        if not os.path.exists(path):
            logger.error(f"Index file not found at {path}")
            return
        self.index = faiss.read_index(path)
        logger.info(f"Loaded FAISS index from {path}")

    def search(self, query: str, top_k=5):
        if self.index is None:
            logger.error("Index not built or loaded. Call build_index() or load_index() first.")
            return []
        q_emb = self.embedder.encode([query], convert_to_numpy=True).astype("float32")
        D, I = self.index.search(q_emb, top_k)
        return [self.texts[i] for i in I[0] if 0 <= i < len(self.texts)]
