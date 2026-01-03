import os
from src.chat.model_loader import ChatModel
from src.storage.retriever import Retriever

def test_chat_pipeline():
    model = ChatModel()
    retriever = Retriever()
    query = "بنی اسرائیل کے بارے میں قرآن کیا کہتا ہے؟"
    context = retriever.get_context(query)
    assert isinstance(context, str)
    assert len(context) > 0

    prompt = f"User asked: {query}\n\nContext:\n{context}\n\nAnswer in Urdu."
    response = model.generate(prompt, max_length=128)
    assert isinstance(response, str)
    assert len(response) > 0
