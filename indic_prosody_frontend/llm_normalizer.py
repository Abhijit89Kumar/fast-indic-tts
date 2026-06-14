"""
llm_normalizer.py
=================
LLM-driven Text-Normalization (TN) front-end for code-mixed Indic TTS, built on
**Sarvam-1 (2B)** (`sarvamai/sarvam-1`).

IMPORTANT DESIGN NOTE (and a key interview talking point)
---------------------------------------------------------
`sarvam-1` is a *base* (pretraining) decoder LM — NOT instruction-tuned. Verbose
"You are a deterministic TN engine ..." system prompts do NOT work: the model
ignores the instructions and emits garbled text. The correct lever for a base
model is **few-shot in-context learning (ICL)**: present a clean, consistent
`Input:/Output:` exemplar pattern and let the model *continue the pattern*.

So this module:
  * uses a compact task header + a curated block of few-shot exemplars,
  * keeps decoding greedy/deterministic,
  * stops the completion at the first newline (one line in, one line out),
  * uses exemplars that are DISJOINT from `data/testset.json` (no eval leakage).

The exemplars teach four behaviours rules cannot do: acronym->phonetic letters,
identifier->digit-by-digit, code-mix locale preservation, and context-aware
number reading.

Run:
    python llm_normalizer.py
    python llm_normalizer.py "your code-mixed sentence"
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass

# Windows consoles default to cp1252 and cannot print Devanagari; force UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

MODEL_ID = "sarvamai/sarvam-1"

# ---------------------------------------------------------------------------
# Few-shot exemplars. DELIBERATELY DISJOINT from data/testset.json so the
# evaluation measures generalization, not memorization. Each line teaches a
# distinct normalization behaviour for code-mixed Hindi/English.
# ---------------------------------------------------------------------------
TASK_HEADER = (
    "Convert each code-mixed Hindi-English sentence into clean spoken form for "
    "a text-to-speech system. Keep Hindi and English words as they are. Spell "
    "acronyms as letters (PM -> pee-em). Read ID and phone digits one by one. "
    "Expand times, dates, currency, units, and percentages into words."
)

FEWSHOT = [
    ("Subah 7:15 AM ki train pakadni hai.",
     "Subah seven fifteen ay-em ki train pakadni hai."),
    ("ATM se Rs 10000 nikaale aaj.",
     "ay-tee-em se ten thousand rupees nikaale aaj."),
    ("Mera roll number 21CS3045 hai.",
     "Mera roll number two one see-es three zero four five hai."),
    ("Aaj 40% discount chal raha hai sale mein.",
     "Aaj forty percent discount chal raha hai sale mein."),
    ("Call me at 90123-45678 kabhi bhi.",
     "Call me at nine zero one two three four five six seven eight kabhi bhi."),
    ("Uska weight 72.5 kg hai abhi.",
     "Uska weight seventy two point five kilogram hai abhi."),
    ("Form 26th January tak jama karo.",
     "Form twenty sixth January tak jama karo."),
    ("NASA ne naya mission launch kiya.",
     "en-ay-es-ay ne naya mission launch kiya."),
    ("Yeh ghar 1990 mein bana tha.",
     "Yeh ghar nineteen ninety mein bana tha."),
    ("Match 5:45 PM se shuru hoga.",
     "Match five forty five pee-em se shuru hoga."),
    ("Mere paas Rs 2.5 lakh hain savings mein.",
     "Mere paas two point five lakh rupees hain savings mein."),
    ("SBI ka customer care 1800 number hai.",
     "es-bee-aai ka customer care one eight zero zero number hai."),
]


# Zero-shot template used by the FINE-TUNED model (no exemplars needed). Shared
# verbatim with train_lora.py so training and inference formats never drift.
ZS_PROMPT = "Input: {x}\nOutput:"


def build_prompt_zeroshot(user_text: str) -> str:
    """Minimal prompt for the LoRA fine-tuned model: it has learned the mapping,
    so no header or exemplars are required (faster, shorter context)."""
    return ZS_PROMPT.format(x=user_text.strip())


def build_prompt(user_text: str) -> str:
    """Assemble the few-shot ICL prompt for the BASE Sarvam-1 model."""
    lines = [TASK_HEADER, ""]
    for src, tgt in FEWSHOT:
        lines.append(f"Input: {src}")
        lines.append(f"Output: {tgt}")
        lines.append("")
    lines.append(f"Input: {user_text.strip()}")
    lines.append("Output:")
    return "\n".join(lines)


def extract_output(completion: str) -> str:
    """Take the first line of the continuation (one line in -> one line out).

    The base model continues by hallucinating further Input:/Output: pairs, so
    we keep only text up to the first newline / next 'Input:' marker.
    """
    text = completion.strip("\n")
    # Cut at the first newline or a hallucinated next example.
    for stop in ("\nInput:", "\nOutput:", "\n"):
        idx = text.find(stop)
        if idx != -1:
            text = text[:idx]
    return text.strip().strip('"').strip()


@dataclass
class SarvamNormalizer:
    """Lazy wrapper around Sarvam-1 for deterministic TN.

    Two modes:
      * base model  -> few-shot ICL prompt   (adapter_path=None, few_shot=True)
      * LoRA model  -> zero-shot prompt       (adapter_path set, few_shot=False)
    """

    model_id: str = MODEL_ID
    max_new_tokens: int = 96
    adapter_path: str = None        # path to a trained LoRA adapter (optional)
    few_shot: bool = True           # auto-disabled when an adapter is loaded
    _tok: object = None
    _model: object = None

    def load(self) -> "SarvamNormalizer":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        if self.adapter_path:
            # Attach the fine-tuned LoRA adapter and switch to zero-shot prompts.
            from peft import PeftModel
            self._model = PeftModel.from_pretrained(self._model,
                                                    self.adapter_path)
            self.few_shot = False
        self._model.eval()
        return self

    def normalize(self, text: str) -> str:
        """Normalize one sentence. Greedy/deterministic decoding."""
        import torch

        if self._model is None:
            raise RuntimeError("Call .load() before .normalize().")

        prompt = build_prompt(text) if self.few_shot \
            else build_prompt_zeroshot(text)
        inputs = self._tok(prompt, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            gen = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,          # greedy -> deterministic
                num_beams=1,
                repetition_penalty=1.05,
                pad_token_id=self._tok.eos_token_id,
            )
        new_tokens = gen[0][inputs["input_ids"].shape[1]:]
        completion = self._tok.decode(new_tokens, skip_special_tokens=True)
        return extract_output(completion)


def main() -> None:
    text = (sys.argv[1] if len(sys.argv) > 1
            else "Mera flight ticket PNR-8392 hai, aur departure 4:30 PM ko hai.")
    print("=" * 72)
    print("SARVAM-1 LLM NORMALIZER  (few-shot in-context learning)")
    print("=" * 72)
    print(f"MODEL : {MODEL_ID}")
    print(f"INPUT : {text}")
    try:
        norm = SarvamNormalizer().load()
        print(f"OUTPUT: {norm.normalize(text)}")
    except Exception as exc:
        print(f"\n[error] model run failed: {exc!r}")
        print("\n[info] Engineered few-shot prompt that would be sent:\n")
        print(build_prompt(text))


if __name__ == "__main__":
    main()
