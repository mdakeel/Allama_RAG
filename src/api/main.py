
# # src/api/main.py
# from src.retrieval.search import FaissSearcher
# from src.reasoning.gpt2_reasoner import GPT2Reasoner

# def answer_question(question: str, top_k: int = 10):
#     # 🔹 Initialize reasoner (model load होगा flant5_reasoner.py में)
#     llm = GPT2Reasoner()

#     # 🔹 Retrieve context
#     searcher = FaissSearcher()
#     results = searcher.search(question, top_k=top_k)

#     if not results: context = "No lecture excerpts available." 
#     else: context = "\n".join( r.get("text") or r.get("text_roman") or "" for r in results if (r.get("text") or r.get("text_roman") or "").strip() )
#     # 🔹 Build prompt
#     prompt = f"""
# Question: {question}

# Context:
# {context}

# Task:
# Answer the question based on the above context.
# Answer concisely in English in 3–4 sentences.
# Avoid vague language.
# Explain clearly in English in 3–4 sentences.
# Do NOT refer to videos or speakers.
# Do Not repeart text from the context.
# If context is empty or irrelevant, politely say: "No relevant lecture content was found."

# Answer: 
# """


#     return llm.generate(prompt, max_new_tokens=220)

# if __name__ == "__main__":
#     q = input("Enter your question: ")
#     print("\nFINAL ANSWER:\n")
#     print(answer_question(q))



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
- Answer MUST be between 4 and 6 sentences. Never shorter.
- Do NOT repeat the question.
- Do NOT copy text from the context verbatim.
- Do NOT repeat any phrase or sentence.
- Avoid generic statements like 'pillars of Islam' or 'foundation of faith'.
- Provide a clear, structured explanation that feels complete and satisfying.
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
