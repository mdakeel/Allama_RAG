import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class FlanT5Reasoner:
    def __init__(self, model_name: str = "google/flan-t5-base"):
        # 🔹 Model load यहीं होगा
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 200) -> str:
        # 🔹 Encode prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        # 🔹 Generate answer
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_length=60,              # multi-sentence guarantee
                num_beams=4,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
                early_stopping=True
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
