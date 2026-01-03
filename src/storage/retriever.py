from src.storage.vector_store import VectorStore
from src.core.logging import logger

class Retriever:
    def __init__(self):
        self.store = VectorStore()
        self.store.build_index()

    def get_context(self, query: str, top_k=5):
        results = self.store.search(query, top_k=top_k)
        logger.info(f"Retrieved {len(results)} context segments")
        return "\n".join(results)
