from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from src.chat.chat_model import ChatModel
from src.chat.model_loader import load_model
from src.core.logging import logger

app = FastAPI(title="Allama RAG API")


class AskRequest(BaseModel):
    question: str


@app.on_event("startup")
def startup_event():
    logger.info("Starting Allama RAG API")
    # load model and retriever lazily via ChatModel
    global chat
    try:
        llm = load_model()
        chat = ChatModel(llm=llm)
        logger.info("Model and retriever initialized")
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")


@app.post("/ask")
def ask(req: AskRequest):
    q = req.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="question required")
    try:
        resp = chat.answer(q)
        # Ensure UTF-8 safe output
        answer = resp.get("answer", "")
        sources = resp.get("sources", [])
        return {"answer": answer, "sources": sources}
    except Exception as e:
        logger.error(f"/ask failed: {e}")
        raise HTTPException(status_code=500, detail="internal error")
