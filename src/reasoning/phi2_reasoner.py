import torch
from src.reasoning.model_loader import load_phi2


class Phi2Reasoner:
    def __init__(self):
        self.model, self.tokenizer = load_phi2()
        self.device = next(self.model.parameters()).device

    def build_prompt(self, question, evidence, score):
        if score >= 0.6:
            mode = "LECTURE_FOUND"
        elif score >= 0.4:
            mode = "WEAK_LECTURE"
        else:
            mode = "NO_LECTURE"

        return f"""
You are an honest Islamic AI assistant.

MODE: {mode}

RULES:
- If MODE is NO_LECTURE → say clearly no lecture found
- Never fake references
- If reasoning yourself, say so

QUESTION:
{question}

EVIDENCE:
{evidence}
"""

    def generate(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=350,
                temperature=0.6,
                top_p=0.9,
                do_sample=True
            )

        return self.tokenizer.decode(output[0], skip_special_tokens=True)
