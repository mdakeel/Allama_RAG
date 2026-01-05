from src.storage.vector_store import VectorStore

vs = VectorStore()
vs.build_index("E:/ML-Projects/Allama/data/transcripts")
vs.save_index("E:/ML-Projects/Allama/data/vector_store/faiss_index")
vs.save_texts("E:/ML-Projects/Allama/data/vector_store/texts.pkl")
