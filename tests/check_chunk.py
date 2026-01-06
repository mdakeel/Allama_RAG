# import pickle

# CHUNKS_PATH = "data/vector_store/chunks.pkl"

# with open(CHUNKS_PATH, "rb") as f:
#     chunks = pickle.load(f)

# print(f"Total chunks: {len(chunks)}\n")

# # First 3 chunks dekhne ke liye
# for i, chunk in enumerate(chunks[:3], 1):
#     print(f"--- Chunk {i} ---")
#     print("Title      :", chunk.get("title"))
#     print("Start      :", chunk.get("start_hhmmss"))
#     print("End        :", chunk.get("end_hhmmss"))
#     print("Text       :", chunk.get("text_roman")[:200], "...")  # pehle 200 chars
#     print("URL        :", chunk.get("play_url"))
#     print()


from src.retrieval.search import VectorSearcher

searcher = VectorSearcher()
query = "وض بلہِ ونشیطانِ رجیم بسم اللہِ رحمان الرحیم"

results = searcher.search(query, top_k=3)

for i, r in enumerate(results, 1):
    print(f"\n--- Result {i} ---")
    print("Score :", round(r['score'], 3))
    print("Title :", r['title'])
    print("Text  :", r['text'][:200], "...")
    print("URL   :", r['play_url'])
