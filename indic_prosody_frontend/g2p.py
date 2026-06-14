"""
g2p.py
======
Grapheme-to-Phoneme (G2P) stage + the assembled end-to-end front-end.

Two things:
  1. EVALUATE the Sarvam-1 G2P LoRA (distilled from espeak-ng) on held-out text,
     reporting PER (Phoneme Error Rate = edit distance over IPA symbols / #gold).
  2. DEMONSTRATE the full front-end pipeline on real code-mixed input:
         raw text  --[Sarvam-1 TN LoRA]-->  normalized spoken form
                   --[Sarvam-1 G2P LoRA]-->  IPA phonemes  (TTS-ready)
     with the espeak-ng IPA shown alongside as the reference.

Run (on the pod):
    python g2p.py --eval  --g2p-adapter sarvam_g2p_lora
    python g2p.py --pipeline --tn-adapter sarvam_tn_lora --g2p-adapter sarvam_g2p_lora
"""

from __future__ import annotations

import argparse
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from metrics import _levenshtein, _normalize_for_scoring  # reuse edit distance


def per(hyp: str, ref: str) -> float:
    """Phoneme Error Rate: char-level edit distance over IPA, ignoring spaces."""
    h = list(hyp.replace(" ", ""))
    r = list(ref.replace(" ", ""))
    if not r:
        return 0.0 if not h else 1.0
    return _levenshtein(h, r) / len(r)


def espeak_g2p(texts, lang="en-us"):
    """Reference G2P via espeak-ng. Returns list of IPA strings."""
    from phonemizer.backend import EspeakBackend
    backend = EspeakBackend(lang, with_stress=False,
                            language_switch="remove-flags")
    return [s.strip() for s in backend.phonemize(texts, strip=True)]


def run_eval(g2p_adapter: str, test_path: str) -> None:
    from llm_normalizer import SarvamNormalizer

    rows = [json.loads(l) for l in open(test_path, encoding="utf-8")]
    inputs = [r["input"] for r in rows]
    gold = [r["output"] for r in rows]  # espeak IPA reference

    print(f"[info] loading Sarvam-1 G2P LoRA from {g2p_adapter} ...")
    g2p = SarvamNormalizer(adapter_path=g2p_adapter, max_new_tokens=160).load()
    hyps = [g2p.normalize(t) for t in inputs]

    pers = [per(h, g) for h, g in zip(hyps, gold)]
    mean_per = sum(pers) / len(pers)
    exact = sum(1 for h, g in zip(hyps, gold)
                if h.replace(" ", "") == g.replace(" ", "")) / len(pers)

    print("\n" + "=" * 60)
    print(f"SARVAM-1 G2P  vs  espeak-ng reference   (n={len(rows)})")
    print("=" * 60)
    print(f"  PER (phoneme error rate): {mean_per*100:.2f}%")
    print(f"  exact phoneme match     : {exact*100:.2f}%")
    print("\nEXAMPLES:")
    for r, h in list(zip(rows, hyps))[:5]:
        print(f"  TEXT : {r['input']}")
        print(f"  GOLD : {r['output']}")
        print(f"  PRED : {h}\n")


def run_pipeline(tn_adapter: str, g2p_adapter: str) -> None:
    """Show the complete raw -> TN -> G2P front-end on code-mixed samples."""
    from llm_normalizer import SarvamNormalizer

    samples = [
        "Mera flight ticket PNR-8392 hai, aur departure 4:30 PM ko hai.",
        "IRCTC se ticket book karo, fare Rs 1250 hai.",
        "Meeting 9:05 AM par hai, 3rd floor conference room mein.",
    ]
    print("[info] loading TN + G2P adapters ...")
    tn = SarvamNormalizer(adapter_path=tn_adapter).load()
    g2p = SarvamNormalizer(adapter_path=g2p_adapter, max_new_tokens=160).load()

    print("\n" + "=" * 72)
    print("FULL FRONT-END:  raw -> TN(Sarvam) -> G2P(Sarvam)  [espeak ref]")
    print("=" * 72)
    out = []
    for s in samples:
        norm = tn.normalize(s)
        ipa = g2p.normalize(norm)
        ref = espeak_g2p([norm])[0]
        print(f"\nRAW : {s}")
        print(f"NORM: {norm}")
        print(f"IPA : {ipa}")
        print(f"REF : {ref}")
        out.append({"raw": s, "normalized": norm, "ipa_sarvam": ipa,
                    "ipa_espeak": ref})
    os.makedirs("results", exist_ok=True)
    with open("results/frontend_pipeline.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n[info] wrote results/frontend_pipeline.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--pipeline", action="store_true")
    ap.add_argument("--g2p-adapter", default="sarvam_g2p_lora")
    ap.add_argument("--tn-adapter", default="sarvam_tn_lora")
    ap.add_argument("--test", default="data/g2p/g2p_test.jsonl")
    args = ap.parse_args()

    if args.eval:
        run_eval(args.g2p_adapter, args.test)
    if args.pipeline:
        run_pipeline(args.tn_adapter, args.g2p_adapter)
    if not (args.eval or args.pipeline):
        print("specify --eval and/or --pipeline")


if __name__ == "__main__":
    main()
