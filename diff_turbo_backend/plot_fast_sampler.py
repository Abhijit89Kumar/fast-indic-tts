"""
plot_fast_sampler.py
====================
Render the Project-1 headline figure from fast_sampler_results.json:
  (left)  quality vs latency   — mel MAE (vs Euler-200 ref) against fused latency
  (right) quality vs NFE       — mel MAE against #function evaluations
showing the 2nd-order Heun solver reaching Euler-50 quality at far lower
latency/NFE. Saves fast_sampler_curve.png.

Run:  python plot_fast_sampler.py
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    with open(os.path.join(HERE, "fast_sampler_results.json")) as f:
        data = json.load(f)
    fused = data["fused"]
    base_mae = next(r["mel_mae"] for r in data["eager"]
                    if r["sampler"] == "euler" and r["nfe"] == 50)

    def series(name):
        rows = sorted([r for r in fused if r["sampler"] == name],
                      key=lambda r: r["nfe"])
        return rows

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for name, color, marker in [("euler", "#d62728", "o"),
                                ("heun", "#1f77b4", "s")]:
        rows = series(name)
        lat = [r["latency_ms"] for r in rows]
        mae = [r["mel_mae"] for r in rows]
        nfe = [r["nfe"] for r in rows]
        ax1.plot(lat, mae, marker=marker, color=color, label=name, lw=2)
        ax2.plot(nfe, mae, marker=marker, color=color, label=name, lw=2)
        for x, y, n in zip(lat, mae, nfe):
            ax1.annotate(f"{n}", (x, y), fontsize=8,
                         textcoords="offset points", xytext=(4, 4))

    ax1.axhline(base_mae, ls="--", color="gray",
                label="Euler-50 (default) MAE")
    ax1.axvline(305, ls=":", color="green", alpha=0.6)
    ax1.set_xlabel("fused sampler latency (ms)")
    ax1.set_ylabel("mel MAE vs Euler-200 reference")
    ax1.set_title("Quality vs Speed (annotated = NFE)")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.axhline(base_mae, ls="--", color="gray", label="Euler-50 (default) MAE")
    ax2.set_xlabel("number of function evaluations (NFE)")
    ax2.set_ylabel("mel MAE vs Euler-200 reference")
    ax2.set_title("Quality vs NFE")
    ax2.legend(); ax2.grid(alpha=0.3)

    fig.suptitle("Diff-Turbo: Grad-TTS reverse diffusion is oversampled at 50 "
                 "steps — the ~16-NFE knee is ~3x faster at near-reference quality",
                 fontweight="bold")
    fig.tight_layout()
    out = os.path.join(HERE, "fast_sampler_curve.png")
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
