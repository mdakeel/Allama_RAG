import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class FlanT5Reasoner:
    def __init__(self, model_name="google/flan-t5-large"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 300) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=2,
                do_sample=False,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3,
                early_stopping=True
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
