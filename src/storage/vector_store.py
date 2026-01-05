import os
import json
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from src.core.paths import data_path
from src.core.logging import logger
from src.utils.helpers import ensure_dir

class VectorStore:
    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        VectorStore manages transcript embeddings with FAISS.
        Default model: multilingual MiniLM (supports English, Urdu, Hindi, etc.)
        """
        logger.info(f"Loading embedding model: {model_name}")
        self.embedder = SentenceTransformer(model_name)
        self.index = None
        # entries is a list of dicts with metadata for each segment
        self.entries = []

    def _split_long_text(self, raw_text: str):
        """Split long transcript strings into manageable chunks."""
        if not isinstance(raw_text, str):
            return []
        text = " ".join(raw_text.split())  # normalize whitespace
        chunks = []
        for piece in text.replace("\n", " ").split("۔"):  # Urdu full stop
            piece = piece.strip()
            if not piece:
                continue
            subpieces = [p.strip() for p in piece.split(".") if p.strip()]
            if subpieces:
                chunks.extend(subpieces)
            else:
                chunks.append(piece)
        # Merge very short chunks
        merged, buf = [], ""
        for c in chunks:
            if len(buf) + len(c) < 200:
                buf = (buf + " " + c).strip()
            else:
                if buf:
                    merged.append(buf)
                buf = c
        if buf:
            merged.append(buf)
        return merged

    def build_index(self, transcripts_dir=data_path("transcripts")):
        """Build FAISS index from transcript JSON files."""
        files = [f for f in os.listdir(transcripts_dir) if f.endswith(".json")]
        all_embeddings = []
        all_entries = []
        for file in files:
            file_path = os.path.join(transcripts_dir, file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load {file}: {e}")
                continue

            file_entries = []
            # prefer structured segments if available
            if "segments" in data and isinstance(data["segments"], list):
                for seg in data["segments"]:
                    if not isinstance(seg, dict):
                        continue
                    text = (seg.get("text_roman") or seg.get("text") or "").strip()
                    if not text:
                        continue
                    start_sec = seg.get("start_sec")
                    start_hh = seg.get("start_hhmmss")
                    end_hh = seg.get("end_hhmmss")
                    play_url = seg.get("play_url") or data.get("play_url") or f"https://www.youtube.com/watch?v={data.get('video_id')}"
                    entry = {
                        "text_roman": text,
                        "start_sec": start_sec,
                        "start_hhmmss": start_hh,
                        "end_hhmmss": end_hh,
                        "play_url": play_url,
                        "video_id": data.get("video_id"),
                        "title": data.get("title"),
                    }
                    file_entries.append(entry)
            elif "Transcripts" in data and isinstance(data["Transcripts"], list):
                for t in data["Transcripts"]:
                    text = t.strip()
                    if text:
                        entry = {"text_roman": text, "play_url": f"https://www.youtube.com/watch?v={data.get('video_id')}", "video_id": data.get("video_id")}
                        file_entries.append(entry)
            elif "transcript_roman" in data and isinstance(data["transcript_roman"], str):
                pieces = self._split_long_text(data["transcript_roman"])
                for p in pieces:
                    entry = {"text_roman": p, "play_url": f"https://www.youtube.com/watch?v={data.get('video_id')}", "video_id": data.get("video_id")}
                    file_entries.append(entry)
            else:
                logger.warning(f"Skipping {file}: no valid transcript key found")
                continue

            if not file_entries:
                logger.warning(f"No text entries found in {file}")
                continue

            try:
                texts_only = [e["text_roman"] for e in file_entries]
                embs = self.embedder.encode(texts_only, convert_to_numpy=True)
            except Exception as e:
                logger.error(f"Embedding failed for {file}: {e}")
                continue

            all_entries.extend(file_entries)
            all_embeddings.append(embs)

        if not all_embeddings:
            logger.error("No embeddings created. Check transcript JSON format.")
            return
        all_embeddings = np.concatenate(all_embeddings, axis=0).astype("float32")
        self.index = faiss.IndexFlatL2(all_embeddings.shape[1])
        self.index.add(all_embeddings)
        self.entries = all_entries
        logger.info(f"Built FAISS index with {len(self.entries)} segments")

    #index faiss
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
    
    #text

    ...
    def save_texts(self, path: str):
        """Save transcript chunks to disk."""
        from src.utils.helpers import ensure_dir
        ensure_dir(os.path.dirname(path))
        with open(path, "wb") as f:
            pickle.dump(self.entries, f)
        logger.info(f"Saved {len(self.entries)} entries to {path}")

    def load_texts(self, path: str):
        """Load transcript chunks from disk."""
        import os
        if not os.path.exists(path):
            logger.error(f"Texts file not found at {path}")
            return
        with open(path, "rb") as f:
            self.entries = pickle.load(f)
        logger.info(f"Loaded {len(self.entries)} entries from {path}")


    def search(self, query: str, top_k=5):
        if self.index is None:
            logger.error("Index not built or loaded. Call build_index() or load_index() first.")
            return []
        q_emb = self.embedder.encode([query], convert_to_numpy=True).astype("float32")
        D, I = self.index.search(q_emb, top_k)
        results = []
        for i in I[0]:
            if 0 <= i < len(self.entries):
                results.append(self.entries[i])
        return results
