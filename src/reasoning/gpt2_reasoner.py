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

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()

        # 🔴 HARD GPT-2 LIMIT
        self.max_positions = self.model.config.n_positions  # = 1024

    def _token_len(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _hard_trim(self, text: str, max_new_tokens: int) -> str:
        """
        Ensure input NEVER exceeds GPT-2 context window
        """
        max_input = self.max_positions - max_new_tokens - 8
        tokens = self.tokenizer.encode(text, add_special_tokens=False)

        if len(tokens) > max_input:
            tokens = tokens[-max_input:]  # keep last part (question + instructions)

        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def generate(self, prompt: str, max_new_tokens: int = 200) -> str:
        # 🔴 ABSOLUTE SAFETY TRIM
        prompt = self._hard_trim(prompt, max_new_tokens)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return text.strip()
