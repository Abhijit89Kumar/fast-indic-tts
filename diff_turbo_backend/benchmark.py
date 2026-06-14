"""
benchmark.py
============
Latency + VRAM benchmark for the Diff-Turbo back-end. Two views:

  (A) KERNEL micro-benchmark — the direct, unambiguous measure of the fusion.
      Compares eager `Mish(a+b)+c` (which PyTorch runs as ~3 separate
      bandwidth-bound CUDA kernels) against the single fused Triton kernel.
      This is the element-wise math of the diffusion reverse-step ODE solver.

  (B) BLOCK end-to-end — eager vs Triton-fused residual block, run 50x to
      simulate a 50-step reverse-diffusion loop. Here two Conv1d ops sit on
      cuDNN and dominate, so the *net* speedup is smaller and only materializes
      once the audio is long enough (T >= ~1024 frames) that the element-wise
      traffic stops being a rounding error next to the convs. At very short T
      the block is conv-bound and fusion can even be net-negative — we report
      it directly rather than only reporting the favourable size.

All timings use CUDA events with warmup; VRAM via max_memory_allocated.

Run:
    python benchmark.py
    python benchmark.py --steps 50 --batch 16 --frames 1024
"""

from __future__ import annotations

import argparse
import statistics

import torch

from triton_fused_sde import (
    EagerResidualBlock,
    TritonFusedResidualBlock,
    _eager_mish,
    fused_residual_add_mish,
    _HAS_TRITON,
)


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------
def _time(fn, steps: int, device: str):
    """Time `fn` for `steps` iterations; return (latencies_ms, peak_mem_MiB)."""
    latencies = []
    for _ in range(15):           # warmup: cuDNN autotune + Triton JIT cache
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    for _ in range(steps):
        if device == "cuda":
            s, e = (torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True))
            s.record(); fn(); e.record()
            torch.cuda.synchronize()
            latencies.append(s.elapsed_time(e))
        else:
            import time
            t0 = time.perf_counter(); fn()
            latencies.append((time.perf_counter() - t0) * 1e3)

    peak = (torch.cuda.max_memory_allocated() / (1024 ** 2)
            if device == "cuda" else float("nan"))
    return latencies, peak


def _stats(lat):
    s = sorted(lat)
    pct = lambda q: s[min(len(s) - 1, int(q * len(s)))]
    return {"mean": statistics.mean(lat), "p50": pct(0.5),
            "p90": pct(0.9), "total": sum(lat)}


def _delta(eager_lat, fused_lat, eager_mem, fused_mem, device, steps):
    es, fs = _stats(eager_lat), _stats(fused_lat)
    print(f"    {'Eager':14s} mean={es['mean']:.4f}ms  p90={es['p90']:.4f}ms  "
          f"total({steps})={es['total']:.2f}ms  peak={eager_mem:.2f}MiB")
    print(f"    {'Triton fused':14s} mean={fs['mean']:.4f}ms  p90={fs['p90']:.4f}ms  "
          f"total({steps})={fs['total']:.2f}ms  peak={fused_mem:.2f}MiB")
    speedup = (es["mean"] - fs["mean"]) / es["mean"] * 100.0
    sx = es["mean"] / fs["mean"] if fs["mean"] > 0 else float("nan")
    print(f"    -> latency speedup: {speedup:+.2f}%  ({sx:.2f}x)")
    if device == "cuda":
        memred = (eager_mem - fused_mem) / eager_mem * 100.0
        print(f"    -> VRAM reduction : {memred:+.2f}%  "
              f"({eager_mem:.2f} -> {fused_mem:.2f} MiB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--channels", type=int, default=80, help="mel channels")
    ap.add_argument("--frames", type=int, default=1024,
                    help="audio frames T (full-utterance scale; try 128 to see "
                         "the conv-bound regime)")
    ap.add_argument("--time-dim", type=int, default=128)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 72)
    print(f"DIFF-TURBO BENCHMARK  (device={device}, triton_active={_HAS_TRITON})")
    print("=" * 72)
    print(f"steps={args.steps}  shape=(B={args.batch}, C={args.channels}, "
          f"T={args.frames})")
    if not _HAS_TRITON:
        print("[warn] Triton/CUDA inactive -> 'fused' == eager fallback; "
              "speedup will read ~0%. Run on a CUDA box (pip install "
              "triton-windows on Windows) for real numbers.")
    print()

    torch.manual_seed(0)
    B, C, T, TD = args.batch, args.channels, args.frames, args.time_dim

    # ---- (A) KERNEL micro-benchmark -------------------------------------
    print("(A) KERNEL  —  Mish(a+b)+c   [eager ~3 kernels vs 1 fused]")
    a = torch.randn(B, C, T, device=device)
    b = torch.randn(B, C, T, device=device)
    c = torch.randn(B, C, T, device=device)
    k_eager, m_eager = _time(lambda: _eager_mish(a + b) + c, args.steps, device)
    k_fused, m_fused = _time(lambda: fused_residual_add_mish(a, b, c),
                             args.steps, device)
    _delta(k_eager, k_fused, m_eager, m_fused, device, args.steps)
    print()

    # ---- (B) BLOCK end-to-end -------------------------------------------
    print("(B) BLOCK  —  full residual block x50 (conv1 + tail + conv2)")
    eager = EagerResidualBlock(C, TD).to(device).eval()
    fused = TritonFusedResidualBlock(C, TD).to(device).eval().load_from(eager)
    x = torch.randn(B, C, T, device=device)
    t_emb = torch.randn(B, TD, device=device)
    with torch.no_grad():
        b_eager, bm_eager = _time(lambda: eager(x, t_emb), args.steps, device)
        b_fused, bm_fused = _time(lambda: fused(x, t_emb), args.steps, device)
    _delta(b_eager, b_fused, bm_eager, bm_fused, device, args.steps)

    print("=" * 72)
    print("Note: the KERNEL block isolates what the fusion actually optimizes "
          "(bandwidth-bound element-wise ops). The BLOCK number is end-to-end "
          "and is gated by the cuDNN convs; it turns clearly positive at "
          "full-utterance frame counts (T>=1024) and is conv-bound at tiny T.")


if __name__ == "__main__":
    main()
