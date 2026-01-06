import os
import logging
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from src.chat.model_loader import load_model
from src.retrieval.search import VectorSearcher
from src.reasoning.evidence_builder import build_evidence_text
from langdetect import detect

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="Islamic Transcript QA API")

# Load LLM model once
MODEL_NAME = os.getenv("HF_MODEL", "google/flan-t5-small")
logger.info(f"Loading model {MODEL_NAME}...")
llm = load_model(MODEL_NAME)

# Load VectorSearcher
logger.info("Loading VectorSearcher...")
searcher = VectorSearcher()

# Request/response models
class QARequest(BaseModel):
    question: str
    top_k: int = 5

class QAResponse(BaseModel):
    answer: str
    sources: List[Dict]

# Helper to detect language
def detect_language(text: str) -> str:
    try:
        lang = detect(text)
        if lang not in ["en", "hi", "ur"]:
            # Arabic/Urdu script → ur
            if any("\u0600" <= ch <= "\u06FF" for ch in text):
                lang = "ur"
            # Devanagari → hi
            elif any("\u0900" <= ch <= "\u097F" for ch in text):
                lang = "hi"
            else:
                lang = "en"
        return lang
    except Exception as e:
        logger.warning(f"Language detection failed: {e}")
        return "en"

# Build prompt for LLM
def build_prompt(question: str, evidence_text: str, lang: str) -> str:
    # The prompt adapts answer language automatically
    prompt = f"""
You are a knowledgeable Islamic AI assistant.

RULES:
- Answer ONLY in {lang}
- Use ONLY the reference content below
- Add light explanation for clarity
- Do NOT hallucinate
- Write clear, structured text
- Use headings, **bold**, and emojis

REFERENCE CONTENT:
{evidence_text}

QUESTION:
{question}

FINAL ANSWER:
"""
    return prompt

@app.post("/ask", response_model=QAResponse)
def ask_question(req: QARequest):
    lang = detect_language(req.question)
    
    # Retrieve top_k chunks
    results = searcher.search(req.question, top_k=req.top_k)
    
    if not results:
        return QAResponse(
            answer="⚠️ This question is not directly covered in the available video segments.",
            sources=[]
        )
    
    # Build evidence-only text
    evidence = build_evidence_text(results)
    
    # Build prompt for LLM
    prompt = build_prompt(req.question, evidence["evidence_text"], lang)
    
    # Generate answer
    answer = llm.generate(prompt, max_length=512)
    
    return QAResponse(answer=answer, sources=evidence["references"])
