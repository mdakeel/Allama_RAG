from chat.model_loader import ChatModel
from chat.language_detect import detect_language
from storage.retriever import Retriever

def main():
    model = ChatModel()
    retriever = Retriever()

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        lang = detect_language(user_input)
        context = retriever.get_context(user_input)

        prompt = f"User asked in {lang}: {user_input}\n\nRelevant context:\n{context}\n\nAnswer in the same language."
        response = model.generate(prompt)
        print(f"Model: {response}")

if __name__ == "__main__":
    main()
