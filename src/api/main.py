
# flatet5 code 

from src.retrieval.search import FaissSearcher
from src.reasoning.flant5_reasoner import FlanT5Reasoner


def build_context(results, all_chunks, window=2, max_chars=800):
    collected = []
    used = set()
    total = 0

    for r in results:
        idx = r.get("index")
        if idx is None:
            continue

        for i in range(idx - window, idx + window + 1):
            if i < 0 or i >= len(all_chunks):
                continue
            if i in used:
                continue

            text = all_chunks[i].get("text") or all_chunks[i].get("text_roman")
            if not text:
                continue

            used.add(i)
            collected.append(text.strip())
            total += len(text)

            if total >= max_chars:
                return "\n".join(collected)

    return "\n".join(collected)


def answer_question(question: str, top_k: int = 8):
    llm = FlanT5Reasoner()
    searcher = FaissSearcher()

    results = searcher.search(question, top_k=top_k)

    context = build_context(
        results,
        searcher.chunks,
        window=2
    )

    if not context.strip():
        return "No relevant lecture content was found."

    prompt = f"""
You are an Islamic scholar AI.

Rules:
- Use ONLY the given context.
- Write the answer in the SAME language as the question.
- there not limit the answer length.
- Provide DETAILED explanations with examples.
- Answer MUST setisfy the question directly.
- Do NOT repeat the question.
- Do NOT copy text from the context verbatim.
- Do NOT repeat any phrase or sentence.
- Avoid generic statements like 'pillars of Islam' or 'foundation of faith'.
- Provide a clear, structured explanation that feels complete and satisfying.
- Do NOT mention the rules in your answer.
- Do NoT give promt in the answer.
- If context is empty or irrelevant, say exactly: "No relevant lecture content was found."




Context:
{context}

Question:
{question}

Answer (strictly follow the rules above):
""".strip()


    return llm.generate(prompt)


if __name__ == "__main__":
    q = input("Enter your question: ")
    print("\nFINAL ANSWER:\n")
    print(answer_question(q))


