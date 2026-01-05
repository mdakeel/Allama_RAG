from src.chat.language_detect import detect_language
from src.storage.retriever import Retriever
from src.core.logging import logger
from src.chat.model_loader import load_model
from deep_translator import GoogleTranslator


class ChatModel:
    def __init__(self, llm=None):
        # llm: an object with generate(prompt) -> str
        self.llm = llm or load_model()
        self.retriever = Retriever()
    
    # Multilingual templates for "no result" messages
    NO_RESULT_MESSAGES = {
        "ur": "❌ متاسف ہے کہ اس موضوع پر video transcripts میں معلومات نہیں مل سکیں۔",
        "hi": "❌ खेद है कि इस विषय पर video transcripts में जानकारी नहीं मिली।",
        "en": "❌ Unfortunately, no relevant information found in video transcripts about this topic.",
        "roman": "❌ Maafi chaahta hoon, is topic par video transcripts mein koi information nahi mili."
    }

    def answer(self, query: str, top_k: int = 5):
        """Generate AI answer ONLY from retrieved video segments.
        
        Pipeline:
        1. Detect query language
        2. Retrieve relevant video segments (FAISS search)
        3. Extract context from transcripts
        4. Generate answer in SAME language as query
        5. Return answer + real video links with timestamps
        """
        query_lang = detect_language(query)
        logger.info(f"Detected query language: {query_lang}")

        # Retrieve context from ACTUAL video transcripts only
        data = self.retriever.get_context(query, top_k=top_k, target_lang=None)
        context = data.get("context", "") or ""
        sources = data.get("sources", []) or []
        
        logger.info(f"Retrieved {len(sources)} video segments")

        if not sources or not context.strip():
            no_result_msg = self.NO_RESULT_MESSAGES.get(query_lang, self.NO_RESULT_MESSAGES["en"])
            return {"answer": no_result_msg, "sources": []}

        # Clean and prepare context
        clean_context = self._clean_context(context)
        
        # Generate answer from transcript context
        answer = self._generate_answer_from_context(clean_context, query, query_lang)
        
        # Format answer with styling
        formatted = self._format_answer(answer, query_lang)
        
        # Add real video sources with timestamps
        top_sources = sources[:5]
        if top_sources:
            formatted += self._format_video_sources(top_sources, query_lang)

        return {"answer": formatted, "sources": top_sources}
    
    def _clean_context(self, context: str) -> str:
        """Remove timestamp markers from context while preserving content."""
        import re
        # Remove timestamp patterns like 📌 [HH:MM – HH:MM]
        clean = re.sub(r'📌\s*\[[\d:–\s]+\]\s*', '', context)
        clean = re.sub(r'\[\d{1,2}:\d{2}\s*–\s*\d{1,2}:\d{2}\]', '', clean)
        # Remove excessive newlines
        clean = re.sub(r'\n\n+', '\n', clean)
        return clean.strip()
    
    def _generate_answer_from_context(self, context: str, query: str, lang: str) -> str:
        """Generate answer from transcript context in the same language as query.
        
        Strategy:
        1. Try LLM generation with context
        2. If LLM fails or answer is weak, extract directly from context
        3. Translate to query language if needed
        """
        answer = ""
        
        # Try LLM-based generation first
        try:
            answer = self._llm_generate(context, query, lang)
            if answer and len(answer) > 20:
                logger.info(f"LLM generated answer ({len(answer)} chars)")
                return answer
        except Exception as e:
            logger.warning(f"LLM generation failed: {e}, falling back to extraction")
        
        # Fallback: extract meaningful sentences from context
        answer = self._extract_from_context(context)
        if answer and len(answer) > 10:
            logger.info(f"Extracted answer from context ({len(answer)} chars)")
            return answer
        
        # Last resort: return first meaningful line
        return self._get_first_meaningful_line(context)
    
    def _llm_generate(self, context: str, query: str, lang: str) -> str:
        """Call LLM with context and query to generate answer."""
        try:
            # Prepare a better prompt
            prompt = self._build_prompt(context, query, lang)
            
            # Call model with reasonable limits (without unsupported parameters)
            answer = self.llm.generate(prompt, max_length=200)
            answer = answer.strip()
            
            # Validate answer quality
            if answer and len(answer) > 15:
                # Remove common prefixes
                for prefix in ["Answer:", "Jawab:", "Response:", "A:", "Q:"]:
                    if answer.lower().startswith(prefix.lower()):
                        answer = answer[len(prefix):].strip()
                
                # Check for too much repetition
                words = answer.split()
                unique_words = len(set(words))
                if unique_words > len(words) * 0.25:  # At least 25% unique words
                    return answer
        except Exception as e:
            logger.warning(f"LLM failed: {e}")
        
        return ""
    
    def _build_prompt(self, context: str, query: str, lang: str) -> str:
        """Build an optimized prompt for the LLM."""
        # Limit context length to avoid token overflow
        context = context[:1500]  # Max 1500 chars of context
        
        if lang == "ur":
            prompt = f"""یہ معلومات پڑھو:
{context}

سوال: {query}

اوپر دی گئی معلومات سے جواب دو:"""
        elif lang == "hi":
            prompt = f"""यह जानकारी पढ़ें:
{context}

सवाल: {query}

ऊपर दी गई जानकारी से जवाब दें:"""
        elif lang == "roman":
            prompt = f"""Yeh info padho:
{context}

Sawal: {query}

Upar di gayi info se jawab do:"""
        else:  # English
            prompt = f"""Read this information:
{context}

Question: {query}

Answer based on the information above:"""
        
        return prompt
    
    def _extract_from_context(self, context: str) -> str:
        """Extract meaningful content directly from context."""
        lines = context.split('\n')
        
        # Collect non-empty, meaningful lines
        meaningful = []
        for line in lines:
            line = line.strip()
            # Skip empty, too short, or timestamp-only lines
            if not line or len(line) < 30:
                continue
            if line.startswith('[') or line.endswith(']'):
                continue
            meaningful.append(line)
        
        if not meaningful:
            return ""
        
        # Combine first 2-3 meaningful lines
        combined = " ".join(meaningful[:3])
        
        # Truncate at sentence boundary
        for separator in ['۔', '.', '!', '?']:
            if separator in combined:
                combined = combined[:combined.rfind(separator) + 1]
                break
        
        return combined[:300].strip()  # Max 300 chars
    
    def _get_first_meaningful_line(self, context: str) -> str:
        """Get the first substantive line from context."""
        for line in context.split('\n'):
            line = line.strip()
            if line and len(line) > 30 and not line.startswith('['):
                return line[:250]
        return "Video mein relevant information hai."
    
    def _format_video_sources(self, sources: list, lang: str) -> str:
        """Format video sources as markdown links."""
        if lang == "ur":
            header = "\n\n**📺 ویڈیو کے ذرائع:**"
        elif lang == "hi":
            header = "\n\n**📺 वीडियो स्रोत:**"
        elif lang == "roman":
            header = "\n\n**📺 Video Sources:**"
        else:
            header = "\n\n**📺 Video Sources:**"
        
        lines = [header]
        for i, url in enumerate(sources, 1):
            if lang == "ur":
                lines.append(f"  {i}. [🎬 ویڈیو دیکھیں]({url})")
            elif lang == "hi":
                lines.append(f"  {i}. [🎬 वीडियो देखें]({url})")
            else:
                lines.append(f"  {i}. [🎬 Watch video]({url})")
        
        return "\n".join(lines)
    
    def _format_answer(self, text: str, lang: str) -> str:
        """Add formatting and language-appropriate styling."""
        if not text:
            if lang == "ur":
                return "📖 ویڈیو میں متعلقہ معلومات ہے۔"
            elif lang == "hi":
                return "📖 वीडियो में प्रासंगिक जानकारी है।"
            else:
                return "📖 Relevant information found in video."
        
        text = text.strip()
        
        # Add opening emoji if missing
        if text and not text.startswith(('📌', '📖', '💡', '✨', '🎯', '❓', '**')):
            text = f"📖 {text}"
        
        # Make first sentence bold for better readability
        if '. ' in text or '۔' in text or '। ' in text:
            separator = '۔' if '۔' in text else ('। ' if '। ' in text else '. ')
            parts = text.split(separator, 1)
            if len(parts[0]) < 180:  # Only if not too long
                parts[0] = f"**{parts[0]}**"
                text = separator.join(parts)
        
        return text
