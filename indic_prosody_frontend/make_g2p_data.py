"""
make_g2p_data.py
================
Build a Grapheme-to-Phoneme (G2P) corpus by *distilling espeak-ng* into
(text -> IPA) pairs, then split train/test. We fine-tune Sarvam-1 on this so a
single LLM front-end performs BOTH text-normalization and G2P (a unified,
production-attractive design vs. chaining separate tools).

Source text = the normalized (spoken-form) outputs from the TN corpus
(`data/train.jsonl`), so G2P is trained on exactly the distribution the TN stage
emits. espeak-ng provides the reference IPA (the de-facto open phonemizer).

Code-switching note: per-span language identification for phonemization is a
known hard problem. To keep the reference labels clean we phonemize the
English-spoken (Latin-script) normalized lines with the en-us voice and hold out
Devanagari-carrier lines (a documented limitation, not hidden).

Run (on the pod, espeak-ng required):
    python make_g2p_data.py --src data/train.jsonl --out data/g2p
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/train.jsonl")
    ap.add_argument("--out", default="data/g2p")
    ap.add_argument("--max", type=int, default=5000)
    ap.add_argument("--test-frac", type=float, default=0.1)
    args = ap.parse_args()

    from phonemizer.backend import EspeakBackend

    # Collect unique Latin-script normalized lines (skip Devanagari carriers).
    texts = []
    seen = set()
    with open(args.src, encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)["output"].strip()
            if _DEVANAGARI.search(t) or t in seen:
                continue
            seen.add(t)
            texts.append(t)
            if len(texts) >= args.max:
                break

    # Batch-phonemize with espeak-ng (en-us). One IPA string per input line.
    backend = EspeakBackend("en-us", with_stress=False,
                            language_switch="remove-flags")
    ipa = backend.phonemize(texts, strip=True)

    pairs = [{"input": t, "output": p.strip()}
             for t, p in zip(texts, ipa) if p.strip()]

    n_test = max(1, int(len(pairs) * args.test_frac))
    test, train = pairs[:n_test], pairs[n_test:]

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "g2p_train.jsonl"), "w",
              encoding="utf-8") as f:
        for p in train:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out, "g2p_test.jsonl"), "w",
              encoding="utf-8") as f:
        for p in test:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"train={len(train)}  test={len(test)}  -> {args.out}")
    print("\nSAMPLES:")
    for p in train[:5]:
        print(f"  IN : {p['input']}\n  IPA: {p['output']}\n")


if __name__ == "__main__":
    main()
