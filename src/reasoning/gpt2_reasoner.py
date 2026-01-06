# src/reasoning/gpt2_reasoner.py , light model for cpu usage or 8gb ram

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class GPT2Reasoner:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.model = AutoModelForCausalLM.from_pretrained(
            "gpt2",
            torch_dtype=torch.float32
        ).to(self.device)

        self.model.eval()

    def build_answer(self, question: str, evidence: list[str], score: float) -> str:
        evidence_text = "\n".join(evidence[:3])

        prompt = f"""
You are a knowledgeable Islamic scholar.

Question:
{question}

Evidence from lectures:
{evidence_text}

Instructions:
- Answer clearly
- If evidence is weak, say so politely
- Do NOT invent scholars or quotes

Answer:
"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=800
        ).to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=180,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )

        text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return text.split("Answer:")[-1].strip()
