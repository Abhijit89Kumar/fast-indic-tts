"""
make_tn_data.py
===============
Synthetic, *correct-by-construction* training corpus for code-mixed Hindi/English
Text Normalization. Used to LoRA fine-tune Sarvam-1.

WHY SYNTHETIC?
--------------
There is no large public corpus of code-mixed Hinglish TN pairs. But TN is a
structured transduction: given a value we can generate BOTH its written surface
form and its correct spoken form deterministically. So we template thousands of
(input, gold) pairs whose labels are correct by construction:

  * acronyms  -> phonetic letter names (PNR -> pee-en-aar), via an explicit dict
  * IDs/OTP/phone/PIN -> digit-by-digit
  * times / dates / currency / units / percent / ordinals / years -> word forms
  * embedded in randomized romanized-Hindi carrier sentences (code-mixing)

The model thus learns the four behaviours rules cannot generalize: acronym
phonetics, identifier digit-reading, code-mix locale preservation, and
context-aware number reading.

IMPORTANT (eval integrity): the held-out evaluation set (data/testset.json) is
HAND-AUTHORED with different carrier sentences. Phenomena overlap (that is the
task) but exact sentences do not, so eval measures generalization, not recall.

Run:
    python make_tn_data.py --n 6000 --seed 0 --out data/train.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from baseline_rules import english_cardinal, english_ordinal

# ---------------------------------------------------------------------------
# Spoken letter names for generic acronym spelling (covers any unseen acronym).
# ---------------------------------------------------------------------------
LETTER = {
    "A": "ay", "B": "bee", "C": "see", "D": "dee", "E": "ee", "F": "ef",
    "G": "jee", "H": "aitch", "I": "aai", "J": "jay", "K": "kay", "L": "el",
    "M": "em", "N": "en", "O": "oh", "P": "pee", "Q": "cue", "R": "aar",
    "S": "es", "T": "tee", "U": "yoo", "V": "vee", "W": "double-yoo",
    "X": "eks", "Y": "why", "Z": "zed",
}


def spell_letters(acr: str) -> str:
    """Generic letter-by-letter spelling for any acronym."""
    return "-".join(LETTER[c] for c in acr.upper() if c in LETTER)


# Acronyms with an explicit spoken form. Most are letter-spelled; a few that are
# conventionally pronounced as words (RAM, PAN, COVID) are given as words to
# match real Indian pronunciation.
ACR_SPELL = ["PNR", "OTP", "IRCTC", "SBI", "RBI", "UPI", "GST", "ATM", "ID",
             "CPU", "GPU", "SP", "CM", "PM", "AM", "DM", "FIR", "IFSC", "NEFT",
             "RTGS", "EMI", "KYC", "USB", "HR", "IT", "AC", "TV", "AI", "ML",
             "API", "URL", "PDF", "ORD", "IPL", "ICU", "MBA", "CEO", "CFO",
             "HDFC", "ICICI", "SIM", "BSNL", "DTH", "GPS", "SMS", "PIN"]
ACR_WORD = {"RAM": "ram", "PAN": "pan", "COVID": "kovid", "NASA": "nasa",
            "WIFI": "wai-fai", "AADHAAR": "aadhaar"}


def acr_spoken(acr: str) -> str:
    if acr in ACR_WORD:
        return ACR_WORD[acr]
    return spell_letters(acr)


def digits_by_digit(s: str) -> str:
    """'8392' -> 'eight three nine two' (the identifier reading)."""
    names = ["zero", "one", "two", "three", "four", "five", "six", "seven",
             "eight", "nine"]
    return " ".join(names[int(c)] for c in s if c.isdigit())


# ---------------------------------------------------------------------------
# Slot generators: each returns (surface, spoken).
# ---------------------------------------------------------------------------
def g_acronym_id(rng):
    acr = rng.choice(ACR_SPELL)
    n = rng.randint(3, 8)
    digits = "".join(str(rng.randint(0, 9)) for _ in range(n))
    sep = rng.choice(["-", " ", ""])
    return f"{acr}{sep}{digits}", f"{acr_spoken(acr)} {digits_by_digit(digits)}"


def g_acronym_word(rng):
    acr = rng.choice(ACR_SPELL + list(ACR_WORD))
    return acr, acr_spoken(acr)


def g_otp(rng):
    n = rng.choice([4, 6])
    digits = "".join(str(rng.randint(0, 9)) for _ in range(n))
    return f"OTP {digits}", f"oh-tee-pee {digits_by_digit(digits)}"


def g_phone(rng):
    digits = "".join(str(rng.randint(0, 9)) for _ in range(10))
    surface = rng.choice([digits, digits[:5] + "-" + digits[5:],
                          "+91 " + digits])
    spoken = ("nine one " if surface.startswith("+91") else "") + \
             digits_by_digit(digits)
    return surface, spoken


def g_pincode(rng):
    digits = "".join(str(rng.randint(0, 9)) for _ in range(6))
    return digits, digits_by_digit(digits)


def g_time(rng):
    h = rng.randint(1, 12)
    m = rng.randint(0, 59)
    ampm = rng.choice(["AM", "PM"])
    surface = f"{h}:{m:02d} {ampm}"
    if m == 0:
        mm = "o'clock"
    elif m < 10:
        mm = "oh " + digits_by_digit(f"{m:02d}")[len("zero "):]  # "oh five"
    else:
        mm = english_cardinal(m)
    spoken = f"{english_cardinal(h)} {mm} {'ay-em' if ampm=='AM' else 'pee-em'}"
    return surface, spoken


def g_currency(rng):
    scale = rng.choice(["", "", "lakh", "crore"])
    if scale:
        val = rng.choice([1, 2, 3, 5, 10, 25, 50])
        dec = rng.choice(["", ".5", ".25"])
        surface = f"Rs {val}{dec} {scale}"
        num = english_cardinal(val) + (
            " point five" if dec == ".5" else
            " point two five" if dec == ".25" else "")
        spoken = f"{num} {scale} rupees"
    else:
        val = rng.randint(10, 99999)
        surface = f"Rs {val}"
        spoken = f"{english_cardinal(val)} rupees"
    return surface, spoken


def g_percent(rng):
    if rng.random() < 0.5:
        val = rng.randint(1, 100)
        return f"{val}%", f"{english_cardinal(val)} percent"
    whole = rng.randint(0, 99)
    frac = rng.randint(1, 99)
    return (f"{whole}.{frac}%",
            f"{english_cardinal(whole)} point {digits_by_digit(f'{frac:02d}')}"
            f" percent")


def g_decimal_unit(rng):
    unit_s, unit_w = rng.choice([("kg", "kilogram"), ("km", "kilometre"),
                                 ("g", "gram"), ("GB", "gigabyte"),
                                 ("MB", "megabyte"), ("litre", "litre"),
                                 ("cm", "centimetre"), ("m", "metre")])
    whole = rng.randint(1, 500)
    if rng.random() < 0.5:
        frac = rng.randint(1, 9)
        return (f"{whole}.{frac} {unit_s}",
                f"{english_cardinal(whole)} point {digits_by_digit(str(frac))}"
                f" {unit_w}")
    return f"{whole} {unit_s}", f"{english_cardinal(whole)} {unit_w}"


def g_ordinal_date(rng):
    day = rng.randint(1, 28)
    month = rng.choice(["January", "February", "March", "April", "May", "June",
                        "July", "August", "September", "October", "November",
                        "December"])
    suf = {1: "st", 2: "nd", 3: "rd"}.get(day % 10 if not 10 < day < 20 else 0,
                                          "th")
    return f"{day}{suf} {month}", f"{english_ordinal(day)} {month}"


def year_to_words(y: int) -> str:
    """Read a 4-digit year in the natural 'nineteen ninety' register."""
    hi, lo = y // 100, y % 100
    if lo == 0:
        return f"{english_cardinal(hi)} hundred"
    if lo < 10:
        return f"{english_cardinal(hi)} oh {english_cardinal(lo)}"
    return f"{english_cardinal(hi)} {english_cardinal(lo)}"


def g_year(rng):
    y = rng.randint(1950, 2030)
    return str(y), year_to_words(y)


_MONTHS = ["January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December"]


def g_slashdate(rng):
    """DD/MM/YYYY (or '-') -> 'fifteenth August twenty twenty four'."""
    d = rng.randint(1, 28)
    m = rng.randint(1, 12)
    y = rng.randint(2015, 2030)
    sep = rng.choice(["/", "-"])
    surface = f"{d:02d}{sep}{m:02d}{sep}{y}"
    return surface, f"{english_ordinal(d)} {_MONTHS[m-1]} {year_to_words(y)}"


def indian_comma(n: int) -> str:
    """Format an int with Indian grouping: 1250000 -> '12,50,000'."""
    s = str(n)
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    parts.insert(0, head)
    return ",".join(parts) + "," + tail


def g_comma_number(rng):
    """Indian comma-grouped large number -> cardinal (lakh/crore)."""
    val = rng.randint(100000, 99999999)
    return indian_comma(val), english_cardinal(val)


def g_product(rng):
    """Product/version numbers are CARDINALS, not years: 'Android 14'."""
    name = rng.choice(["iPhone", "Android", "Galaxy", "Windows", "Pixel",
                       "iOS", "OnePlus", "Note"])
    n = rng.randint(2, 22)
    extra = rng.choice(["", "", " Pro", " Max", " Plus", " Pro Max", " Ultra"])
    return f"{name} {n}{extra}", f"{name} {english_cardinal(n)}{extra}"


def g_currency_dev(rng):
    """Currency written with Devanagari रुपये (kept verbatim)."""
    val = rng.randint(100, 99999)
    return f"{val} रुपये", f"{english_cardinal(val)} रुपये"


def g_count(rng):
    val = rng.randint(2, 5000)
    return str(val), english_cardinal(val)


SLOTS = [g_acronym_id, g_acronym_id, g_otp, g_phone, g_pincode, g_time,
         g_time, g_currency, g_currency, g_percent, g_decimal_unit,
         g_ordinal_date, g_year, g_acronym_word, g_count, g_slashdate,
         g_comma_number, g_product, g_product]

# Romanized-Hindi (code-mixed) carrier templates with one {X} slot. The Hindi
# carrier words must be preserved verbatim in the gold; only {X} is normalized.
CARRIERS = [
    "Mera {X} hai abhi.",
    "Kal {X} ki baat hui thi.",
    "Usne mujhe {X} bataya phone par.",
    "{X} ko note kar lo please.",
    "Office mein {X} ka kaam pending hai.",
    "Bhai {X} jaldi bhej do.",
    "Aaj ki meeting {X} important hai.",
    "Maine {X} already confirm kar diya.",
    "Yeh {X} sahi hai kya check karo.",
    "Driver ko {X} address de dena.",
    "Ticket par {X} likha hua tha.",
    "Mummy ne {X} ke baare mein poocha.",
    "Customer ka {X} verify karna hai.",
    "Form mein {X} fill karna zaroori hai.",
    "{X} wala order cancel kar do.",
    "Sir, {X} ki detail mail kar dijiye.",
    "Hostel ka {X} abhi tak nahi aaya.",
    "Exam ke liye {X} yaad rakhna.",
]

# Devanagari-script Hindi carriers (code-mixing where the carrier is in Hindi
# script but the normalized number/acronym is spoken in English, matching real
# Indian TTS convention, e.g. "मेरी सैलरी fifty thousand रुपये").
DEVA_CARRIERS = [
    "मेरा {X} है अभी।",
    "कल {X} की बात हुई थी।",
    "उसने मुझे {X} बताया फोन पर।",
    "{X} को नोट कर लो प्लीज़।",
    "ऑफिस में {X} का काम बाकी है।",
    "आज की मीटिंग {X} ज़रूरी है।",
    "फॉर्म में {X} भरना ज़रूरी है।",
    "{X} वाला ऑर्डर कैंसिल कर दो।",
    "मुझे {X} याद रखना है।",
    "सर, {X} की डिटेल भेज दीजिए।",
]
# Slots safe to drop into a Devanagari carrier (English-spoken values; plus a
# Devanagari-rupees currency variant).
DEVA_SLOTS = [g_acronym_id, g_otp, g_phone, g_pincode, g_time, g_percent,
              g_year, g_count, g_currency_dev, g_acronym_word, g_slashdate,
              g_comma_number, g_product]


def make_example(rng):
    """Build one (input, output) pair. ~30% Devanagari carrier, ~30% two-slot."""
    devanagari = rng.random() < 0.30
    carrier = rng.choice(DEVA_CARRIERS if devanagari else CARRIERS)
    slots = DEVA_SLOTS if devanagari else SLOTS
    joiner = " और " if devanagari else " aur "
    if rng.random() < 0.30:
        s1, w1 = rng.choice(slots)(rng)
        s2, w2 = rng.choice(slots)(rng)
        surface = carrier.format(X=f"{s1}{joiner}{s2}")
        spoken = carrier.format(X=f"{w1}{joiner}{w2}")
    else:
        s, w = rng.choice(slots)(rng)
        surface = carrier.format(X=s)
        spoken = carrier.format(X=w)
    return {"input": surface, "output": spoken}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                  "data", "train.jsonl"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    seen = set()
    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        while n < args.n:
            ex = make_example(rng)
            key = ex["input"]
            if key in seen:
                continue
            seen.add(key)
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} unique pairs -> {args.out}")
    # Show a few samples for a sanity check.
    rng2 = random.Random(args.seed + 1)
    print("\nSAMPLES:")
    for _ in range(6):
        ex = make_example(rng2)
        print(f"  IN : {ex['input']}\n  OUT: {ex['output']}\n")


if __name__ == "__main__":
    main()
