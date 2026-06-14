"""
run_tn_eval.py
==============
Full Text-Normalization evaluation: runs the labeled test set
(`data/testset.json`) through three systems and scores each with WER/CER/EM,
plus a per-category breakdown that pinpoints WHERE the LLM beats rules.

Systems:
  * naive        — indic-numtowords digit-strip baseline
  * competitive  — engineered regex rule system (fair opponent)
  * sarvam       — Sarvam-1 (2B) LLM normalizer

Outputs:
  * console tables (corpus-level + per-category)
  * results/tn_results.json   (raw per-sentence outputs for inspection)
  * results/tn_summary.json   (aggregate + per-category metrics)

Run (on the GPU pod):
    python run_tn_eval.py
    python run_tn_eval.py --no-llm     # baselines only (fast, no model load)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Windows consoles default to cp1252; force UTF-8 for Devanagari output.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from baseline_rules import competitive_normalize, naive_normalize
from metrics import aggregate, per_category

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "testset.json")
RESULTS_DIR = os.path.join(HERE, "results")


def load_testset() -> list:
    with open(DATA, encoding="utf-8") as f:
        return json.load(f)["samples"]


def print_corpus_table(by_system: dict) -> None:
    print("\n" + "=" * 72)
    print("CORPUS-LEVEL  (lower WER/CER better; higher EM better)")
    print("=" * 72)
    for name, agg in by_system.items():
        print("  " + agg.as_row(name))


def print_category_table(cat_metrics: dict, systems: list) -> None:
    print("\n" + "=" * 72)
    print("PER-CATEGORY WER (%)  — where the LLM earns its keep")
    print("=" * 72)
    cats = sorted({c for s in systems for c in cat_metrics[s]})
    header = f"  {'category':24s}" + "".join(f"{s:>13s}" for s in systems)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for cat in cats:
        row = f"  {cat:24s}"
        for s in systems:
            agg = cat_metrics[s].get(cat)
            row += f"{(agg.wer*100 if agg else float('nan')):>12.1f}%"
        print(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true",
                    help="skip all LLM systems (baselines only)")
    ap.add_argument("--base-llm", action="store_true",
                    help="evaluate base Sarvam-1 with few-shot ICL")
    ap.add_argument("--adapter", default=None,
                    help="path to LoRA adapter -> evaluate fine-tuned Sarvam-1")
    ap.add_argument("--limit", type=int, default=0, help="cap #samples (debug)")
    args = ap.parse_args()

    samples = load_testset()
    if args.limit:
        samples = samples[:args.limit]
    refs = [s["gold"] for s in samples]
    cats = [s["category"] for s in samples]

    systems = {}  # name -> list[str] hypotheses

    # --- rule baselines ---------------------------------------------------
    systems["naive"] = [naive_normalize(s["text"]) for s in samples]
    systems["competitive"] = [competitive_normalize(s["text"]) for s in samples]

    # --- Sarvam-1 systems -------------------------------------------------
    latencies = {}  # system -> mean ms/sentence

    def run_llm(label: str, norm) -> None:
        print(f"[info] running '{label}' over {len(samples)} sentences ...")
        outs, t0 = [], time.time()
        for i, s in enumerate(samples, 1):
            outs.append(norm.normalize(s["text"]))
            if i % 10 == 0:
                print(f"  ... {i}/{len(samples)}")
        latencies[label] = (time.time() - t0) / len(samples) * 1000
        systems[label] = outs
        print(f"[info] {label} mean latency: {latencies[label]:.0f} ms/sentence")

    if not args.no_llm:
        from llm_normalizer import SarvamNormalizer
        if args.base_llm:
            print("[info] loading base Sarvam-1 (few-shot ICL) ...")
            run_llm("sarvam_fewshot", SarvamNormalizer().load())
        if args.adapter:
            print(f"[info] loading LoRA-fine-tuned Sarvam-1 from {args.adapter} ...")
            run_llm("sarvam_lora",
                    SarvamNormalizer(adapter_path=args.adapter).load())

    # --- score ------------------------------------------------------------
    by_system = {name: aggregate(hyps, refs) for name, hyps in systems.items()}
    cat_metrics = {
        name: per_category([
            {"category": c, "hyp": h, "ref": r}
            for c, h, r in zip(cats, hyps, refs)
        ])
        for name, hyps in systems.items()
    }

    print_corpus_table(by_system)
    print_category_table(cat_metrics, list(systems.keys()))

    # --- persist ----------------------------------------------------------
    os.makedirs(RESULTS_DIR, exist_ok=True)
    raw = []
    for i, s in enumerate(samples):
        raw.append({
            "id": s["id"], "category": s["category"], "input": s["text"],
            "gold": s["gold"],
            **{name: systems[name][i] for name in systems},
        })
    with open(os.path.join(RESULTS_DIR, "tn_results.json"), "w",
              encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)

    summary = {
        "corpus": {name: vars(agg) for name, agg in by_system.items()},
        "per_category_wer": {
            name: {c: agg.wer for c, agg in cm.items()}
            for name, cm in cat_metrics.items()
        },
        "llm_mean_latency_ms": latencies,
    }
    with open(os.path.join(RESULTS_DIR, "tn_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[info] wrote results/ -> tn_results.json, tn_summary.json")


if __name__ == "__main__":
    main()
