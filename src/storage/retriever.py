from src.storage.vector_store import VectorStore
from src.core.paths import data_path
from src.core.logging import logger
from deep_translator import GoogleTranslator
import urllib.parse


def _make_timestamp_url(video_id: str = None, play_url: str = None, start_sec=None):
    """Build a YouTube URL with timestamp.
    
    Priority:
    1. Use existing play_url with timestamp
    2. Build from video_id
    3. Use play_url as-is
    """
    if start_sec is None:
        start_sec = 0
    
    start_sec = int(start_sec)
    
    # If we have a video_id, always build clean URL
    if video_id:
        return f"https://www.youtube.com/watch?v={video_id}&t={start_sec}s"
    
    # Otherwise, try to use/fix play_url
    if play_url:
        try:
            parsed = urllib.parse.urlparse(play_url)
            params = urllib.parse.parse_qs(parsed.query)
            
            # Remove old timestamp
            if 't' in params:
                del params['t']
            
            # Build new query string
            query_parts = []
            for key, vals in params.items():
                for val in vals:
                    query_parts.append(f"{key}={val}")
            
            # Build final URL
            base = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            if query_parts:
                query_str = "&".join(query_parts)
                return f"{base}?{query_str}&t={start_sec}s"
            else:
                return f"{base}?t={start_sec}s"
        except Exception as e:
            logger.warning(f"Failed to parse URL: {e}")
            return play_url
    
    return None


class Retriever:
    def __init__(self,
                 index_path=data_path("vector_store/faiss_index"),
                 texts_path=data_path("vector_store/texts.pkl")):
        self.vs = VectorStore()
        # attempt to load index and texts; VectorStore logs failure
        self.vs.load_index(index_path)
        self.vs.load_texts(texts_path)

    def expand_query(self, query: str):
        """Expand query with translations to improve retrieval."""
        translations = [query]
        for lang in ["ur", "hi", "en"]:
            try:
                translated = GoogleTranslator(source="auto", target=lang).translate(query)
                if translated and translated not in translations:
                    translations.append(translated)
            except Exception as e:
                logger.warning(f"Translation failed for {lang}: {e}")
        return translations

    def get_context(self, query: str, top_k=5, target_lang=None):
        """Retrieve relevant video segments from vector store.
        
        Returns:
            dict with 'context' (formatted text) and 'sources' (YouTube URLs with timestamps)
        """
        queries = self.expand_query(query)
        hits = []
        
        # Search across all query versions
        for q in queries:
            try:
                res = self.vs.search(q, top_k=top_k)
                if res:
                    hits.extend(res)
            except Exception as e:
                logger.warning(f"Search failed for '{q}': {e}")

        # Deduplicate and format results
        seen = set()
        context_blocks = []
        sources = []
        
        for h in hits:
            if not isinstance(h, dict):
                continue
            
            # Get text content
            text = h.get("text_roman", "").strip() or h.get("text", "").strip()
            if len(text) < 5:
                continue
            
            # Dedupe by video_id + start_sec
            video_id = h.get("video_id", "")
            start_sec = h.get("start_sec")
            key = (video_id, start_sec)
            
            if key in seen:
                continue
            seen.add(key)

            # Format context block with timestamps
            start_time = h.get("start_hhmmss") or "00:00"
            end_time = h.get("end_hhmmss") or ""
            
            if end_time:
                time_range = f"{start_time} – {end_time}"
            else:
                time_range = start_time
            
            block = f"📌 [{time_range}]\n{text}"
            context_blocks.append(block)

            # Build timestamped YouTube URL
            ts_url = _make_timestamp_url(
                video_id=video_id,
                play_url=h.get("play_url"),
                start_sec=start_sec
            )
            
            if ts_url:
                sources.append(ts_url)
            else:
                logger.warning(f"Could not build URL for video {video_id}")

        logger.info(f"Retrieved {len(context_blocks)} segments, {len(sources)} URLs")

        # Combine context blocks
        context_text = "\n\n".join(context_blocks)
        
        # Optionally translate context to target language
        if target_lang and context_text and target_lang != "roman":
            try:
                context_text = GoogleTranslator(source="auto", target=target_lang).translate(context_text)
            except Exception as e:
                logger.warning(f"Context translation to {target_lang} failed: {e}")

        return {
            "context": context_text,
            "sources": list(dict.fromkeys(sources))  # Dedupe while preserving order
        }
