import os
import logging
from src.core.logging import logger

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
    import torch
except ImportError as e:
    logger.warning(f"Transformers not installed: {e}")


class ModelLoader:
    """
    Production-ready model loader for HuggingFace models.
    Supports both seq2seq (text2text) and causalLM (text-generation) models.
    Automatically selects GPU if available.
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or os.getenv("HF_MODEL", "google/flan-t5-small")
        device = 0 if torch.cuda.is_available() else -1
        logger.info(f"Loading model: {self.model_name} on device {device}")

        self.pipeline_type = None
        self.pipe = None

        # Try text2text first (good for instruction-following)
        try:
            self.pipe = pipeline("text2text-generation", model=self.model_name, device=device)
            self.pipeline_type = "text2text-generation"
            logger.info("Using text2text-generation pipeline")
        except Exception:
            # fallback to text-generation (causalLM)
            try:
                self.pipe = pipeline("text-generation", model=self.model_name, device=device)
                self.pipeline_type = "text-generation"
                logger.info("Using text-generation pipeline")
            except Exception as e:
                logger.error(f"Failed to load model pipeline: {e}")
                raise RuntimeError(f"Cannot load model {self.model_name}") from e

    def generate(self, prompt: str, max_length: int = 512, temperature: float = 0.7, top_p: float = 0.9) -> str:
        """
        Generate text from prompt using the loaded model pipeline.

        Args:
            prompt: str - input text prompt
            max_length: int - max tokens to generate
            temperature: float - sampling temperature
            top_p: float - nucleus sampling probability

        Returns:
            str - generated text
        """
        if not self.pipe:
            logger.error("Pipeline is not loaded")
            return ""

        try:
            output = self.pipe(
                prompt,
                max_length=max_length,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                truncation=True
            )
            # pipeline output can be a list of dicts
            if isinstance(output, list) and output:
                first = output[0]
                return first.get("generated_text") or first.get("text") or ""
            return str(output)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return ""


def load_model(model_name: str = None) -> ModelLoader:
    """
    Convenience function to load the model.
    """
    return ModelLoader(model_name)
