import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

_MODEL = None
_TOKENIZER = None


def load_phi2(model_name="microsoft/phi-2"):
    global _MODEL, _TOKENIZER

    if _MODEL is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        _TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        _MODEL = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)

        _MODEL.eval()

    return _MODEL, _TOKENIZER
