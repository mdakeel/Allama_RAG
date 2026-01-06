from src.embeddings.embedder import TextEmbedder

def test_embedding():
    texts = [
        "اللہ ایک ہے اور وہی معبود ہے",
        "Bani Israel ko Allah ne fazilat di"
    ]

    embedder = TextEmbedder()
    embeddings = embedder.embed_texts(texts)

    print("Total embeddings:", len(embeddings))
    print("Embedding shape:", embeddings[0].shape)
    print("Sample vector (first 5 values):", embeddings[0][:5])

if __name__ == "__main__":
    test_embedding()
