# Allama RAG System - Production Setup

## Quick Start

```bash
# 1. Activate virtual environment
.\.venv\Scripts\activate

# 2. Install dependencies (if not done)
pip install -r requirements.txt

# 3. Build FAISS index from transcripts (one-time setup)
python rebuild_index.py

# 4. Start the API server
.\.venv\Scripts\uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

# 5. In another terminal, test the API
python test_api.py
```

## API Usage

### Endpoint: POST `/ask`

**Request:**

```json
{
  "question": "What is the Quran?"
}
```

**Response:**

```json
{
  "answer": "AI-generated answer based on video segments...",
  "sources": [
    "https://www.youtube.com/watch?v=VIDEO_ID&t=XXs",
    "https://www.youtube.com/watch?v=VIDEO_ID2&t=YYs"
  ]
}
```

## System Features

✅ **Multi-language Support**

- English, Urdu, Hindi, Roman Urdu automatic detection
- Query expansion into multiple languages for better retrieval

✅ **FAISS Vector Search**

- 4556 indexed segments from your transcripts
- Fast semantic search with multilingual embeddings
- Automatic deduplication

✅ **YouTube Integration**

- Clickable links with exact timestamps (&t=XXs)
- Video metadata included in response

✅ **LLM-based Generation**

- Intelligent answer synthesis from multiple segments
- No hallucination - only uses provided content
- Fallback for unmatched queries

✅ **Production-Ready**

- Full logging and monitoring
- UTF-8 support
- Error handling
- Configurable model selection

## Configuration

### Change LLM Model

```bash
export HF_MODEL="google/flan-t5-large"
# or other HuggingFace seq2seq models
```

### Adjust Retrieval

Edit `src/chat/chat_model.py` - change `top_k` parameter (default: 6)

### Production Deployment

```bash
# Use Gunicorn for multiple workers
pip install gunicorn
gunicorn src.api.app:app -w 4 -b 0.0.0.0:8000
```

## Files Structure

```
src/
├── api/
│   └── app.py                 # FastAPI /ask endpoint
├── chat/
│   ├── chat_model.py          # LLM answer generation
│   ├── language_detect.py     # Multi-language detection
│   └── model_loader.py        # HuggingFace model loading
├── storage/
│   ├── retriever.py           # Query expansion + FAISS search
│   └── vector_store.py        # Embedding + indexing
├── core/
│   ├── config.py
│   ├── logging.py
│   └── paths.py
└── utils/
    ├── helpers.py
    └── retry.py

data/
├── transcripts/               # Your JSON transcript files
└── vector_store/
    ├── faiss_index           # FAISS index (4556 segments)
    └── texts.pkl             # Segment metadata
```

## Troubleshooting

**Issue: Always getting fallback message**
→ Run `python rebuild_index.py` to rebuild FAISS index

**Issue: Slow responses**
→ Use a larger GPU or reduce `top_k` in chat_model.py

**Issue: Poor answer quality**
→ Switch to a larger LLM: `export HF_MODEL="google/flan-t5-large"`

## Testing

```bash
# Test retrieval
python test_retrieval.py

# Test API with multiple languages
python test_api_multi.py
```

---

**Version:** 1.0  
**Last Updated:** 2026-01-05  
**Status:** Production Ready ✅
