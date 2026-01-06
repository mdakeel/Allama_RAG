import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class QueryNormalizer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            self.tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ur")
        except Exception as e:
            msg = str(e).lower()
            if "sentencepiece" in msg:
                raise ImportError("The tokenizer requires the 'sentencepiece' package. Install it with: pip install sentencepiece") from e
            raise

        self.model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-ur").to(self.device)

    def normalize(self, query: str) -> str:
        tokens = self.tokenizer(
            query,
            return_tensors="pt",
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                **tokens,
                max_length=64
            )

        return self.tokenizer.decode(
            out[0],
            skip_special_tokens=True
        )
