# src/reasoning/gpt2_reasoner.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class GPT2Reasoner:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        # Use dtype instead of deprecated torch_dtype
        self.model = AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float32).to(self.device)

        # Set pad token to EOS (GPT-2 has no PAD by default)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()
        self.max_positions = getattr(self.model.config, "n_positions", 1024)

    def _token_len(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _trim_to_budget(self, prefix: str, evidence_texts: list[str], meta_lines: list[str], max_new_tokens: int) -> tuple[str, list[str], list[str]]:
        # Reserve room for generation and a safety margin
        safety = 32
        budget = self.max_positions - max_new_tokens - safety
        # Start with prefix only
        prefix_tokens = self._token_len(prefix)
        if prefix_tokens >= budget:
            # Hard trim the prefix itself if needed
            # Keep last budget tokens of the prefix to preserve question and instructions tail
            enc = self.tokenizer.encode(prefix, add_special_tokens=False)
            trimmed = self.tokenizer.decode(enc[-budget:], skip_special_tokens=True)
            return trimmed, [], []

        # Iteratively add meta + evidence until budget is hit
        kept_meta, kept_evi = [], []
        current = prefix_tokens

        def add_block(block: str):
            nonlocal current
            # Soft cap each block to avoid long paragraphs
            block = block.strip()
            if len(block) > 1500:
                block = block[:1500]
            tok = self._token_len(block)
            if current + tok <= budget:
                current += tok
                return block, True
            return "", False

        # Add meta lines first (they're short and useful)
        for m in meta_lines:
            b, ok = add_block(m)
            if not ok:
                break
            kept_meta.append(b)

        # Add evidence excerpts
        for e in evidence_texts:
            b, ok = add_block(e)
            if not ok:
                break
            kept_evi.append(b)

        # Rebuild prefix with kept meta/evidence
        meta_block = "\n".join(kept_meta)
        evi_block = "\n".join(kept_evi)
        full = f"{prefix}\n\nSource context (titles, timestamps, links):\n{meta_block}\n\nLecture excerpts:\n{evi_block}"
        # If even assembling increased length too much, fallback to prefix-only
        if self._token_len(full) > budget:
            full = prefix
            kept_meta, kept_evi = [], []

        return full, kept_evi, kept_meta

    def generate(self, prompt: str, max_new_tokens: int = 200):
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,          # extra guard
            max_length=self.max_positions - max_new_tokens - 4
        ).to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,       # deterministic
                temperature=None,      # don't pass temperature when not sampling
                top_p=None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        if "Answer:" in text:
            return text.split("Answer:", 1)[-1].strip()
        return text.strip()

    def build_answer(self, question: str, evidence_texts: list[str], meta_lines: list[str], disclaimer: str | None = None, max_new_tokens: int = 200):
        prefix = f"""
You are an Islamic scholar who explains clearly and concisely.
Respond ONLY in English. Use simple language. Do not copy long text verbatim. Summarize and explain.

Question:
{question}

Instructions:
- Read all excerpts together and explain in your own words.
- If the evidence does not explicitly cover the question, say so politely.
- Do NOT invent scholars, quotes, or sources.
- Prefer short, clear paragraphs.
- If relevant, reference the provided title and timestamp briefly (e.g., "Lecture X, 12:03–14:10").

Answer:
""".strip()

        if disclaimer:
            prefix = f"{disclaimer}\n\n{prefix}"

        safe_prompt, kept_evi, kept_meta = self._trim_to_budget(prefix, evidence_texts[:30], meta_lines[:30], max_new_tokens=max_new_tokens)
        return self.generate(safe_prompt, max_new_tokens=max_new_tokens)
