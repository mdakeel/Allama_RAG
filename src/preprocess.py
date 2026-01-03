from src.storage.vector_store import VectorStore
from src.core.paths import data_path
from src.core.logging import logger

def main():
    logger.info("Starting preprocessing: building FAISS index from transcripts...")
    vs = VectorStore()
    vs.build_index(transcripts_dir=data_path("transcripts"))
    # Save index into vector_store folder
    vs.save_index(data_path("vector_store", "faiss_index"))
    logger.info(" Preprocessing complete. FAISS index saved.")

if __name__ == "__main__":
    main()
