# from huggingface_hub import InferenceClient
# import os
# from dotenv import load_dotenv

# load_dotenv()
# hf_token = os.getenv("HF_TOKEN")

# # Qwen model (multilingual, Urdu/English strong)
# client = InferenceClient("Qwen/Qwen2-7B-Instruct", token=hf_token)

# response = client.chat_completion(
#     messages=[{"role": "user", "content": "Explain huruf e muqattat"}],
#     max_tokens=300
# )

# print(response)


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

model_id = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

question = "Explain huruf e muqattat"
context = "Huruf-e-Muqattaʿat are unique letter combinations that appear at the beginning of certain chapters (Surahs) in the Quran. Their exact meanings are not definitively known, but they are believed to hold special significance and are considered a miraculous aspect of the Quranic text."
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

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


Question:
{question}

Answer (strictly follow the rules above):
""".strip()
result = pipe(prompt, max_new_tokens=200)

print(result[0]["generated_text"])
