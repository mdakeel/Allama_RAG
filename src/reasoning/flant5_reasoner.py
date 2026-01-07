import torch
import warnings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

warnings.filterwarnings("ignore")


class FlanT5Reasoner:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(
            "google/flan-t5-large"
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            "google/flan-t5-large",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)

        self.model.eval()

        self.max_input_tokens = 512
        self.max_new_tokens = 200

    def _safe_trim(self, text: str) -> str:
        tokens = self.tokenizer.encode(text, add_special_tokens=False)

        if len(tokens) <= self.max_input_tokens:
            return text

        # keep instructions + question
        head = tokens[: int(self.max_input_tokens * 0.4)]
        tail = tokens[-int(self.max_input_tokens * 0.6):]

        return self.tokenizer.decode(
            head + tail,
            skip_special_tokens=True
        )

    def generate(self, prompt: str) -> str:
        prompt = self._safe_trim(prompt)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.max_input_tokens,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs, 
                max_new_tokens=self.max_new_tokens, 
                min_length=120, # 🔥 ensures multi-sentence 
                num_beams=4, # beam search for quality 
                repetition_penalty=2.0, # 🔥 strong penalty against repeats 
                no_repeat_ngram_size=3, # 🔥 prevents phrase repetition 
                early_stopping=True
            )

        return self.tokenizer.decode(
            output[0],
            skip_special_tokens=True
        ).strip()
