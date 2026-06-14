"""
profile_gradtts.py
==================
Amdahl's-law profile of Grad-TTS reverse diffusion: what fraction of the
synthesis wall-clock is actually the bandwidth-bound element-wise work our
Triton kernels target (Mish + the SDE Euler step) vs the compute-bound
Conv2d / GroupNorm / attention?

It explains why fusing the element-wise ops yields a large kernel-level speedup
(see benchmark.py) but a small end-to-end speedup in this model: the U-Net is
compute-bound, dominated by attention.

Run (on the pod):
    python profile_gradtts.py --gradtts /path/Grad-TTS --timesteps 50
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gradtts_triton import encode_text, load_models, synth_mel

# Coarse op buckets keyed by substrings of the aten op name.
BUCKETS = {
    "conv (Conv2d)": ["conv", "cudnn"],
    "groupnorm": ["group_norm", "native_group_norm"],
    "attention/einsum/matmul": ["bmm", "matmul", "einsum", "mm", "softmax"],
    "elementwise: mish (tanh/softplus/mul)": ["tanh", "softplus", "mul",
                                              "silu", "sigmoid"],
    "elementwise: add/sub": ["add", "sub"],
    "elementwise: other (copy/cat/etc)": ["copy", "cat", "stack", "to_copy",
                                          "contiguous", "expand"],
}


def bucketize(name: str) -> str:
    low = name.lower()
    for bucket, keys in BUCKETS.items():
        if any(k in low for k in keys):
            return bucket
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gradtts", required=True)
    ap.add_argument("--timesteps", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    bundle = load_models(args.gradtts)
    x, x_len = encode_text(bundle, "Diffusion based text to speech accelerated "
                                   "with custom Triton kernels.")

    # Warmup.
    for _ in range(2):
        synth_mel(bundle, x, x_len, args.timesteps, args.temperature, args.seed)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        synth_mel(bundle, x, x_len, args.timesteps, args.temperature, args.seed)
        torch.cuda.synchronize()

    # Aggregate self CUDA time by bucket.
    totals: dict = {}
    grand = 0.0
    for evt in prof.key_averages():
        t = evt.self_device_time_total or 0.0  # microseconds
        if t <= 0:
            continue
        totals[bucketize(evt.key)] = totals.get(bucketize(evt.key), 0.0) + t
        grand += t

    print("=" * 64)
    print(f"GRAD-TTS SYNTHESIS PROFILE  ({args.timesteps} reverse steps)")
    print("=" * 64)
    print(f"{'bucket':44s}{'CUDA ms':>10s}{'%':>8s}")
    print("-" * 64)
    for bucket, t in sorted(totals.items(), key=lambda kv: -kv[1]):
        print(f"{bucket:44s}{t/1000:>10.2f}{100*t/grand:>7.1f}%")
    print("-" * 64)
    ew = sum(t for b, t in totals.items() if b.startswith("elementwise"))
    print(f"{'TOTAL element-wise (our kernels target this)':44s}"
          f"{ew/1000:>10.2f}{100*ew/grand:>7.1f}%")
    print(f"\nAmdahl ceiling: even an INFINITELY fast element-wise path caps "
          f"end-to-end speedup at {100*ew/grand:.1f}% of runtime.\n"
          f"=> Grad-TTS reverse diffusion is COMPUTE-bound (conv/attention); "
          f"that is why kernel-level 2-3x gains do not translate end-to-end.")


if __name__ == "__main__":
    main()
