
# src/api/main.py
from src.retrieval.search import FaissSearcher
from src.reasoning.gpt2_reasoner import GPT2Reasoner

def answer_question(question: str, top_k: int = 10):
    # 🔹 Initialize reasoner (model load होगा flant5_reasoner.py में)
    llm = GPT2Reasoner()

    # 🔹 Retrieve context
    searcher = FaissSearcher()
    results = searcher.search(question, top_k=top_k)

    if not results: context = "No lecture excerpts available." 
    else: context = "\n".join( r.get("text") or r.get("text_roman") or "" for r in results if (r.get("text") or r.get("text_roman") or "").strip() )
    # 🔹 Build prompt
    prompt = f"""
Question: {question}

Context:
{context}

Task:
Answer the question based on the above context.
Answer concisely in English in 3–4 sentences.
Avoid vague language.
Explain clearly in English in 3–4 sentences.
Do NOT refer to videos or speakers.
Do Not repeart text from the context.
If context is empty or irrelevant, politely say: "No relevant lecture content was found."

Answer: 
"""


    return llm.generate(prompt, max_new_tokens=220)

if __name__ == "__main__":
    q = input("Enter your question: ")
    print("\nFINAL ANSWER:\n")
    print(answer_question(q))
