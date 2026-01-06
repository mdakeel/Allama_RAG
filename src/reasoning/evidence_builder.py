# src/reasoning/evidence_builder.py

from src.retrieval.search import VectorSearcher
from src.reasoning.gpt2_reasoner import GPT2Reasoner

SCORE_THRESHOLD = 0.4


def build_evidence_answer(question: str, top_k: int = 5):
    print("=== Evidence Builder Started ===")

    # 1. Query (NO normalization)
    query = question

    print("\nOriginal Query:")
    print(query)

    # 2. Vector Search
    searcher = VectorSearcher()
    results = searcher.search(query, top_k=top_k)

    if not results:
        return "No evidence found.", []

    evidence_blocks = []
    references = []
    scores = []

    # 3. Collect evidence (DIRECT STRUCTURE)
    for res in results:
        text = res.get("text", "")
        score = float(res.get("score", 0.0))

        if not text.strip():
            continue

        evidence_blocks.append(text)
        scores.append(score)

        references.append({
            "title": res.get("title", "Unknown"),
            "time": f"{res.get('start_hhmmss', '')}–{res.get('end_hhmmss', '')}",
            "url": res.get("play_url", "")
        })

    if not evidence_blocks:
        return "No usable evidence found.", []

    max_score = max(scores) if scores else 0.0

    # 4. Reasoning (GPT-2, CPU SAFE)
    reasoner = GPT2Reasoner()
    answer = reasoner.build_answer(
        question=question,
        evidence=evidence_blocks,
        score=max_score
    )

    return answer, references


# -------------------------
# CLI TEST
# -------------------------
if __name__ == "__main__":
    q = (
        "What are Huruf-e-Muqatta'at "
        "(the disconnected letters like Alif Lam Meem) "
        "that appear at the beginning of some Quran chapters?"
    )

    ans, refs = build_evidence_answer(q)

    print("\n=== FINAL ANSWER ===\n")
    print(ans)

    print("\n=== REFERENCES ===\n")
    for r in refs:
        print(f"- {r['title']} [{r['time']}]")
        print(f"  {r['url']}")
