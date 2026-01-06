from src.retrieval.search import VectorSearcher
from src.reasoning.flant5_reasoner import FlanT5Reasoner


def answer_question(question: str) -> str:
    searcher = VectorSearcher()
    reasoner = FlanT5Reasoner()

    results = searcher.search(question, top_k=25)

    if not results:
        return "No relevant lecture evidence was found."

    # 🔹 Merge ALL relevant text (trim safely)
    evidence_blocks = []
    for r in results:
        txt = r["text"].strip()
        if len(txt) > 400:
            txt = txt[:400]
        evidence_blocks.append(txt)

    merged_evidence = "\n".join(evidence_blocks)

    final_prompt = f"""
You are a knowledgeable Islamic teacher.

Below is lecture content extracted from multiple Quran lessons.
Understand it carefully and answer the question.

Lecture Content:
{merged_evidence}

Rules:
- Answer ONLY in English
- Explain clearly and calmly
- Do NOT mention videos or speakers
- Do NOT quote long text
- Write at least 8–10 sentences

Question:
{question}

Answer:
"""

    return reasoner.generate(final_prompt, max_new_tokens=300).strip()


if __name__ == "__main__":
    q = "Huruf-e-Muqattaat kya hain? Unka maqsad kya bataya gaya hai?"
    print("\nFINAL ANSWER:\n")
    print(answer_question(q))
