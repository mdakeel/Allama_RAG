import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class MBartTranslator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = "facebook/mbart-large-50-many-to-many-mmt"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id).to(self.device)
        self.model.eval()

    def translate_to_english(self, text: str) -> str:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
        translated_tokens = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tokenizer.lang_code_to_id["en_XX"],
            max_length=512
        )
        return self.tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
