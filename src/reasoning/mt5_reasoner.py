import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class MT5Reasoner:
    def __init__(self, model_name: str = "google/mt5-base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Multilingual model (handles Urdu/Hindi/English/Roman Urdu)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.max_input_tokens = 512
        self.max_new_tokens = 250

    def _safe_trim(self, text: str) -> str:
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= self.max_input_tokens:
            return text
        return self.tokenizer.decode(tokens[:self.max_input_tokens], skip_special_tokens=True)

    def generate(self, prompt: str, min_length: int = 60) -> str:
        prompt = self._safe_trim(prompt)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_input_tokens).to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                min_length=min_length,
                num_beams=4,
                repetition_penalty=1.2,
                no_repeat_ngram_size=2,
                early_stopping=True
            )

        return self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
