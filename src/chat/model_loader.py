import os
import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

logger = logging.getLogger("allama")

class ModelLoader:
    def __init__(self, model_name=None):
        self.model_name = model_name or os.getenv("HF_MODEL", "google/flan-t5-small")
        device = 0 if torch.cuda.is_available() else -1
        logger.info(f"Loading Seq2Seq model: {self.model_name} on device {device}")

        # Load tokenizer and Seq2Seq model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

        # Pipeline for text2text generation
        self.pipe = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device
        )

    def generate(self, prompt: str, max_length=256):
        output = self.pipe(prompt, max_length=max_length)
        return output[0]["generated_text"]

def load_model(model_name=None):
    return ModelLoader(model_name)
