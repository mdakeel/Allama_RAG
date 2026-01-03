from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from src.core.config import SETTINGS
from src.core.logging import logger

class ChatModel:
    def __init__(self):
        model_name = SETTINGS["app"]["model_name"]
        logger.info(f"Loading model: {model_name}")

        # Load tokenizer + model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",   # auto-select GPU/CPU
            torch_dtype="auto"   # efficient precision
        )

        # Create pipeline
        self.chat_pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )

    def generate(self, prompt: str, max_length: int = 512) -> str:
        logger.info(f"Generating response for: {prompt[:50]}...")
        output = self.chat_pipeline(
            prompt,
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )[0]["generated_text"]
        return output
