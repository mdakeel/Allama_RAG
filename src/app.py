import logging
import sys
from fastapi import FastAPI
from pydantic import BaseModel
from src.chat.model_loader import ModelLoader
from src.chat.chat_model import ChatModel

sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])

app = FastAPI(
    title="Islamic RAG API",
    description="Ask questions based on YouTube video transcripts",
    version="1.0.0"
)

model_loader = ModelLoader()
chat = ChatModel(model_loader)

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    sources: list[str] | None = []

@app.post("/ask", response_model=AnswerResponse)
def ask_question(req: QuestionRequest):
    result = chat.answer(req.question)
    if isinstance(result, dict):
        return result
    return {"answer": str(result), "sources": []}
