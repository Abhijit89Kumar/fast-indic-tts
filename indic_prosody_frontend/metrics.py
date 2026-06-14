"""
metrics.py
==========
Evaluation metrics for Text Normalization quality. Pure-Python (no GPU / deps),
so it runs anywhere and is trivially auditable — important because these are the
numbers that go on the CV and get defended in the interview.

We report, against the hand-authored gold spoken-form:
  * WER  — Word Error Rate   (edit distance over whitespace tokens / #gold words)
  * CER  — Character Error Rate (edit distance over chars / #gold chars)
  * EM   — Exact Match rate   (fraction of sentences normalized exactly)

WER/CER are the standard TTS-front-end metrics: a normalizer that produces the
right *spoken words* is what matters, so word-level edit distance is the primary
score and character-level catches near-misses.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


def _normalize_for_scoring(s: str) -> str:
    """Light, defensible canonicalization before scoring.

    We do NOT want to reward/penalize cosmetic differences (case, punctuation,
    repeated spaces) — only the spoken-word content. We lowercase, strip
    punctuation except intra-word hyphens (pee-en-aar is one spoken token), and
    collapse whitespace. Unicode is NFC-normalized so Devanagari compares cleanly.
    """
    s = unicodedata.normalize("NFC", s).lower().strip()
    # Keep word chars, Devanagari, spaces and hyphens; drop other punctuation.
    s = re.sub(r"[^\wऀ-ॿ\s-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _levenshtein(a: list, b: list) -> int:
    """Classic O(len(a)*len(b)) edit distance between two token sequences."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1,        # deletion
                           cur[j - 1] + 1,     # insertion
                           prev[j - 1] + cost))  # substitution
        prev = cur
    return prev[-1]


def wer(hyp: str, ref: str) -> float:
    """Word Error Rate of hypothesis vs reference (0 = perfect)."""
    r = _normalize_for_scoring(ref).split()
    h = _normalize_for_scoring(hyp).split()
    if not r:
        return 0.0 if not h else 1.0
    return _levenshtein(h, r) / len(r)


def cer(hyp: str, ref: str) -> float:
    """Character Error Rate of hypothesis vs reference (0 = perfect)."""
    r = list(_normalize_for_scoring(ref).replace(" ", ""))
    h = list(_normalize_for_scoring(hyp).replace(" ", ""))
    if not r:
        return 0.0 if not h else 1.0
    return _levenshtein(h, r) / len(r)


def exact_match(hyp: str, ref: str) -> bool:
    """Whether hyp equals ref after scoring-canonicalization."""
    return _normalize_for_scoring(hyp) == _normalize_for_scoring(ref)


@dataclass
class Aggregate:
    """Corpus-level summary for one system."""
    n: int
    wer: float
    cer: float
    em: float

    def as_row(self, name: str) -> str:
        return (f"{name:16s}  WER={self.wer*100:6.2f}%  "
                f"CER={self.cer*100:6.2f}%  EM={self.em*100:6.2f}%  (n={self.n})")


def aggregate(hyps: list, refs: list) -> Aggregate:
    """Mean WER/CER and exact-match rate over a parallel hyp/ref corpus."""
    assert len(hyps) == len(refs) and hyps, "non-empty parallel lists required"
    n = len(hyps)
    w = sum(wer(h, r) for h, r in zip(hyps, refs)) / n
    c = sum(cer(h, r) for h, r in zip(hyps, refs)) / n
    e = sum(1 for h, r in zip(hyps, refs) if exact_match(h, r)) / n
    return Aggregate(n=n, wer=w, cer=c, em=e)


def per_category(rows: list) -> dict:
    """Group results by category and aggregate each.

    `rows`: list of dicts with keys 'category', 'hyp', 'ref'.
    Returns {category: Aggregate}.
    """
    buckets: dict = {}
    for r in rows:
        buckets.setdefault(r["category"], []).append((r["hyp"], r["ref"]))
    return {cat: aggregate([h for h, _ in pairs], [g for _, g in pairs])
            for cat, pairs in buckets.items()}


if __name__ == "__main__":
    # Self-test: identical -> 0 error; one-word swap -> 1/len WER.
    ref = "pee-en-aar eight three nine two"
    assert wer(ref, ref) == 0.0 and exact_match(ref, ref)
    hyp = "pee-en-aar eight three nine ZERO"
    print(f"demo WER={wer(hyp, ref):.3f}  CER={cer(hyp, ref):.3f}  "
          f"EM={exact_match(hyp, ref)}")
    print("metrics self-test OK")
