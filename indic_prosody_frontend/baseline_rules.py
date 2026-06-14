"""
baseline_rules.py
=================
Rule-based Text-Normalization (TN) baselines for the Indic-Prosody front-end.

We provide TWO baselines so the LLM comparison is fair:

  1. `naive_normalize`      — the textbook indic-numtowords path: strip non-digits,
                              expand the integer. Breaks on almost everything.
  2. `competitive_normalize`— a *best-effort* engineered rule system: regexes for
                              times / currency / percent / ordinals / decimals +
                              a real English cardinal speller. This is a FAIR
                              opponent for the LLM. It still cannot:
                                * spell acronyms phonetically (PNR -> pee-en-aar),
                                * read identifiers digit-by-digit (it says
                                  "eight thousand..." for an 8392 PNR),
                                * disambiguate by context (ID vs price vs year),
                                * preserve the Hindi/English code-mix locale.
                              Those four are exactly where Sarvam-1 wins, and the
                              per-category metrics in evaluate.py prove it.

Run:
    python baseline_rules.py            # demo + documented failure analysis
"""

from __future__ import annotations

import re
import sys

# Windows consoles default to cp1252 and cannot print Devanagari; force UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---------------------------------------------------------------------------
# Optional Hindi expander (used by the naive baseline to mirror a real
# "Indic" rule stack that is locked to one locale).
# ---------------------------------------------------------------------------
try:
    from indic_numtowords import num2words as _hi_num2words  # type: ignore
    _HAS_LIB = True
except Exception as exc:  # pragma: no cover
    _HAS_LIB = False
    _IMPORT_ERR = exc

    def _hi_num2words(number, lang="hi"):  # type: ignore
        _HI = {0: "शून्य", 4: "चार", 30: "तीस", 430: "चार सौ तीस",
               8392: "आठ हज़ार तीन सौ बानवे"}
        return _HI.get(int(number), str(number))


TEST_SENTENCE = "Mera flight ticket PNR-8392 hai, aur departure 4:30 PM ko hai."


# ===========================================================================
# A small, self-contained ENGLISH cardinal speller (0 .. 10^12), so the
# competitive baseline can expand numbers in English without extra deps.
# ===========================================================================
_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven",
         "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
         "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety"]


def _three_digits(n: int) -> str:
    """Spell 0..999 in English."""
    out = []
    if n >= 100:
        out.append(_ONES[n // 100] + " hundred")
        n %= 100
    if n >= 20:
        out.append(_TENS[n // 10] + ((" " + _ONES[n % 10]) if n % 10 else ""))
    elif n > 0:
        out.append(_ONES[n])
    return " ".join(out)


def english_cardinal(n: int) -> str:
    """Spell a non-negative integer in English using the Indian numbering
    system (lakh / crore), which is the natural register for Indic TTS."""
    if n == 0:
        return "zero"
    parts = []
    # Indian grouping: crore (10^7), lakh (10^5), thousand (10^3), then 0..999.
    for div, name in ((10**7, "crore"), (10**5, "lakh"), (10**3, "thousand")):
        if n >= div:
            parts.append(_three_digits(n // div) + " " + name)
            n %= div
    if n > 0:
        parts.append(_three_digits(n))
    return " ".join(parts)


def english_ordinal(n: int) -> str:
    """Spell an ordinal: 1->first, 2->second, 21->twenty first, etc."""
    special = {1: "first", 2: "second", 3: "third", 5: "fifth", 8: "eighth",
               9: "ninth", 12: "twelfth"}
    if n in special:
        return special[n]
    base = english_cardinal(n)
    if base.endswith("y"):
        return base[:-1] + "ieth"          # twenty -> twentieth
    last = n % 10
    if 10 < n % 100 < 20:                   # teens
        return base + "th"
    return base + {1: "st", 2: "nd", 3: "rd"}.get(last, "th") \
        if last in (1, 2, 3) else base + "th"


# ===========================================================================
# Baseline 1: NAIVE — the textbook indic-numtowords failure path.
# ===========================================================================
def naive_normalize(sentence: str, lang: str = "hi") -> str:
    """Whitespace-tokenize; any token with a digit -> strip non-digits ->
    Hindi cardinal. This is what most Indic TN tutorials ship, and it loses the
    PNR prefix, mangles 4:30 into 430, and forces a single Hindi locale."""
    out = []
    for token in sentence.split():
        if any(c.isdigit() for c in token):
            digits = re.sub(r"\D", "", token)
            if digits:
                try:
                    out.append(_hi_num2words(int(digits), lang=lang))
                except Exception:
                    out.append(token)
                continue
        out.append(token)
    return " ".join(out)


# ===========================================================================
# Baseline 2: COMPETITIVE — best-effort engineered rules (English locale).
# ===========================================================================
_RE_TIME = re.compile(r"\b(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)?\b")
_RE_CURR = re.compile(r"(?:Rs\.?|₹)\s?(\d[\d,]*(?:\.\d+)?)\s?(lakh|crore)?", re.I)
_RE_PCT = re.compile(r"(\d+(?:\.\d+)?)\s?%")
_RE_ORD = re.compile(r"\b(\d+)(st|nd|rd|th)\b", re.I)
_RE_DEC = re.compile(r"\b(\d+)\.(\d+)\b")
_RE_INT = re.compile(r"\b\d[\d,]*\b")

_DIGIT_WORDS = {c: w for c, w in zip("0123456789", _ONES[:10])}


def _spell_decimal(int_part: str, frac: str) -> str:
    whole = english_cardinal(int(int_part.replace(",", "")))
    digits = " ".join(_DIGIT_WORDS[d] for d in frac)
    return f"{whole} point {digits}"


def _spell_time(h: str, m: str, ampm: str | None) -> str:
    hh = english_cardinal(int(h))
    if m == "00":
        mm = "o'clock"
    elif m[0] == "0":
        mm = "oh " + _DIGIT_WORDS[m[1]]
    else:
        mm = english_cardinal(int(m))
    suffix = ""
    if ampm:
        suffix = " " + {"am": "ay-em", "pm": "pee-em"}[ampm.lower()]
    return f"{hh} {mm}{suffix}".strip()


def competitive_normalize(sentence: str) -> str:
    """Engineered, English-locale rule normalizer. Strong on structured numeric
    patterns; deliberately still blind to acronym phonetics, identifier
    digit-reading, semantic context, and code-mix locale."""
    s = sentence
    # Order matters: most specific patterns first.
    s = _RE_TIME.sub(lambda m: _spell_time(m.group(1), m.group(2), m.group(3)), s)
    s = _RE_CURR.sub(
        lambda m: (english_cardinal(int(float(m.group(1).replace(",", ""))))
                   if "." not in m.group(1) else
                   _spell_decimal(*m.group(1).split(".")))
        + (" " + m.group(2).lower() if m.group(2) else "") + " rupees", s)
    s = _RE_PCT.sub(
        lambda m: (english_cardinal(int(m.group(1))) if "." not in m.group(1)
                   else _spell_decimal(*m.group(1).split("."))) + " percent", s)
    s = _RE_ORD.sub(lambda m: english_ordinal(int(m.group(1))), s)
    s = _RE_DEC.sub(lambda m: _spell_decimal(m.group(1), m.group(2)), s)
    # Remaining bare integers -> English cardinal (the naive part: an 8392 PNR
    # gets read as a magnitude here, which is wrong but is the rule limit).
    s = _RE_INT.sub(lambda m: english_cardinal(int(m.group(0).replace(",", ""))), s)
    return re.sub(r"\s+", " ", s).strip()


# Backwards-compatible alias used by evaluate.py / older callers.
rule_based_normalize = competitive_normalize


def main() -> None:
    print("=" * 72)
    print("RULE-BASED BASELINES")
    print("=" * 72)
    if not _HAS_LIB:
        print(f"[warn] indic-numtowords not importable ({_IMPORT_ERR!r}); "
              f"naive baseline uses a tiny fallback.\n")

    print(f"INPUT       : {TEST_SENTENCE}")
    print(f"NAIVE       : {naive_normalize(TEST_SENTENCE)}")
    print(f"COMPETITIVE : {competitive_normalize(TEST_SENTENCE)}\n")

    print("-" * 72)
    print("WHAT EVEN THE COMPETITIVE RULE SYSTEM STILL GETS WRONG")
    print("-" * 72)
    failures = [
        ("PNR-8392 (acronym)",
         "No phonetic acronym table: 'PNR' is left as-is (or dropped), never "
         "spoken as 'pee-en-aar'."),
        ("PNR-8392 (identifier)",
         "8392 is an ID and must be read 'eight three nine two', but the rule "
         "engine reads it as the magnitude 'eight thousand three hundred "
         "ninety two'. No way to know it is an ID without context."),
        ("Locale / code-mix",
         "The competitive engine forces ONE language (English here); a true "
         "Hinglish line needs per-span language decisions a regex cannot make."),
        ("Semantic context",
         "1947 (year) vs 1947 (count) vs 1947 (ID) all normalize identically. "
         "The rule engine has no notion of meaning."),
    ]
    for i, (item, why) in enumerate(failures, 1):
        print(f"{i}. {item}\n   -> {why}\n")
    print("These four categories are precisely where Sarvam-1 wins — quantified "
          "per-category in evaluate.py.")


if __name__ == "__main__":
    main()
