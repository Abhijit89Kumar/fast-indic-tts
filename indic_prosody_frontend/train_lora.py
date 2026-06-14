"""
train_lora.py
=============
LoRA fine-tune Sarvam-1 (2B) into a code-mixed Text-Normalization transducer.

Why LoRA: the base model can't be prompted into reliable TN (see llm_normalizer
docstring). Full fine-tuning a 2B model is wasteful for a narrow transduction
task; LoRA trains <1% of params, fits easily on one A100, and yields a tiny
(~tens of MB) adapter — the efficient, production-realistic choice.

Training format is IDENTICAL to inference (`ZS_PROMPT` from llm_normalizer), and
we mask the prompt tokens in the loss so the model is trained ONLY to produce
the normalized output, not to echo the input.

Run (on the A100 pod):
    python train_lora.py --data data/train.jsonl --out sarvam_tn_lora \
        --epochs 3 --bs 16
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)

from llm_normalizer import MODEL_ID, ZS_PROMPT


class TNDataset(Dataset):
    """Tokenize (input -> output) pairs; mask the prompt span in the labels."""

    def __init__(self, path: str, tok, max_len: int = 128):
        self.rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                prompt = ZS_PROMPT.format(x=ex["input"])
                target = " " + ex["output"] + tok.eos_token
                p_ids = tok(prompt, add_special_tokens=True).input_ids
                t_ids = tok(target, add_special_tokens=False).input_ids
                ids = (p_ids + t_ids)[:max_len]
                # -100 => ignored by cross-entropy: train only on the target.
                labels = ([-100] * len(p_ids) + t_ids)[:max_len]
                self.rows.append({"input_ids": ids, "labels": labels})

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def make_collator(pad_id: int):
    def collate(batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            pad = maxlen - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [pad_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            attn.append([1] * len(b["input_ids"]) + [0] * pad)
        return {"input_ids": torch.tensor(input_ids),
                "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(attn)}
    return collate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train.jsonl")
    ap.add_argument("--out", default="sarvam_tn_lora")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=128,
                    help="token cap per example (raise for long G2P targets)")
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map={"": 0})
    model.config.use_cache = False

    # LoRA on all attention + MLP projections (standard for Llama-family).
    lora = LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = TNDataset(args.data, tok, max_len=args.max_len)
    print(f"[info] training examples: {len(ds)}")

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=2,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        bf16=True,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=25,
        save_strategy="no",
        report_to="none",
        dataloader_num_workers=2,
    )

    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=make_collator(tok.pad_token_id))
    trainer.train()

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"[info] saved LoRA adapter -> {args.out}")


if __name__ == "__main__":
    main()
