
# # flatet5 code 

# from src.retrieval.search import FaissSearcher
# from src.reasoning.mt5_reasoner import MT5Reasoner


# def build_context(results, all_chunks, window=2, max_chars=500):
#     collected = []
#     used = set()
#     total = 0

#     for r in results:
#         idx = r.get("index")
#         if idx is None:
#             continue

#         for i in range(idx - window, idx + window + 1):
#             if i < 0 or i >= len(all_chunks):
#                 continue
#             if i in used:
#                 continue

#             text = all_chunks[i].get("text") or all_chunks[i].get("text_roman")
#             if not text:
#                 continue

#             used.add(i)
#             collected.append(text.strip())
#             total += len(text)

#             if total >= max_chars:
#                 return "\n".join(collected)

#     return "\n".join(collected)


# def answer_question(question: str, top_k: int = 20):
#     llm = MT5Reasoner()
#     searcher = FaissSearcher()

#     results = searcher.search(question, top_k=top_k)
#     print("results:", results)

#     context = build_context(
#         results,
#         searcher.chunks,
#         window=2
#     )
#     print("\nContext:\n", context)
#     if not context.strip():
#         return "No relevant lecture content was found."

#     prompt = f"""
# You are an Islamic scholar AI.
# Use ONLY the given context to answer.
# Always write the answer in English.
# Give a detailed and structured explanation with examples.

# Context:
# {context}

# Question:
# {question}

# Answer:
# """.strip()





#     return llm.generate(prompt)


# if __name__ == "__main__":
#     q = input("Enter your question: ")
#     print("\nFINAL ANSWER:\n")
#     print(answer_question(q))





from src.retrieval.search import FaissSearcher
from src.reasoning.mt5_reasoner import MT5Reasoner

def build_context(results, all_chunks, window=2, max_chars=900):
    collected, used, total = [], set(), 0
    for r in results:
        idx = r.get("index")
        if idx is None:
            continue
        for i in range(idx - window, idx + window + 1):
            if i < 0 or i >= len(all_chunks) or i in used:
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

def answer_question(question: str, top_k: int = 20):
    llm = MT5Reasoner()
    searcher = FaissSearcher()

    results = searcher.search(question, top_k=top_k)
    print("results\n:", results[:5])  # debug

    context = build_context(results, searcher.chunks, window=2)
    print("\nContext:\n", context[:500])

    if not context.strip():
        return "No relevant lecture content was found."

    # English enforced prompt
    prompt = f"""
سوال کا جواب صرف نیچے دیے گئے کانٹیکسٹ سے دیں۔
جواب اردو میں دیں اور تفصیلی وضاحت کریں۔

کانٹیکسٹ:
{context}

سوال:
{question}

جواب:
""".strip()


    print("Prompt tokens:", len(llm.tokenizer.encode(prompt))) 
    print("\nPrompt sent to model:\n", prompt)

    return llm.generate(prompt)

if __name__ == "__main__":
    q = input("Enter your question: ").strip()
    print("\nFINAL ANSWER:\n")
    print(answer_question(q))
