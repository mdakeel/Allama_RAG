from src.preprocess.preprocess import preprocess_all

chunks = preprocess_all("data/transcripts")

print("Total chunks:", len(chunks))
print(chunks[0])
print(chunks[1])
