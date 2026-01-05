#!/usr/bin/env python
"""Rebuild FAISS index with proper metadata entries."""
import sys
sys.path.insert(0, 'e:/ML-Projects/Allama')

from src.storage.vector_store import VectorStore
from src.core.paths import data_path

print("Building FAISS index from transcripts...")
vs = VectorStore()
vs.build_index(transcripts_dir=data_path("transcripts"))
vs.save_index(data_path("vector_store/faiss_index"))
vs.save_texts(data_path("vector_store/texts.pkl"))
print(f"✅ Index built with {len(vs.entries)} segments")
