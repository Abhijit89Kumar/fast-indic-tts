"""
torch_profiler_script.py
========================
Profile the *eager* PyTorch diffusion residual block to expose where the
reverse-sampling loop spends its time and memory bandwidth — the empirical
justification for the Triton fusion in `triton_fused_sde.py`.

We build a dummy U-Net residual block (1-D convs + Mish/ReLU + element-wise
adds), run it under `torch.profiler.profile`, and dump:
  * a sorted op table (self CUDA time),
  * peak memory stats,
  * a Chrome trace (`trace.json`) you can open in chrome://tracing.

The key thing to look for in the table: the element-wise ops (`aten::add`,
`aten::mish`/`silu`, `aten::mul`) each appear as separate kernels — exactly the
launches we collapse with Triton.

Run:
    python torch_profiler_script.py
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, record_function


class DummyUNetResidualBlock(nn.Module):
    """A representative Grad-TTS-style residual block for profiling.

    Mirrors the real workload: two Conv1d stages, a broadcast time-embedding
    add, Mish + ReLU activations, and a residual skip — all the element-wise
    traffic that dominates a memory-bound diffusion step.
    """

    def __init__(self, channels: int = 80, time_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, channels)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)                         # compute-bound
        t = self.time_proj(t_emb).unsqueeze(-1)   # (B, C, 1) broadcast bias
        h = h + t                                 # element-wise add  (kernel)
        h = h * torch.tanh(F.softplus(h))         # Mish              (kernels)
        h = self.conv2(h)                         # compute-bound
        h = F.relu(h)                             # element-wise      (kernel)
        return h + x                              # residual add      (kernel)


def main(channels: int = 80, frames: int = 128, batch: int = 16,
         time_dim: int = 128, warmup: int = 5, active: int = 20) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 72)
    print(f"TORCH PROFILER  —  eager diffusion residual block  (device={device})")
    print("=" * 72)
    print(f"shape: (B={batch}, C={channels}, T={frames})  "
          f"[C={channels} mel channels = standard audio frame width]\n")

    block = DummyUNetResidualBlock(channels, time_dim).to(device).eval()
    x = torch.randn(batch, channels, frames, device=device)
    t_emb = torch.randn(batch, time_dim, device=device)

    # Warm up cuDNN autotuner / allocator so the trace isn't polluted.
    with torch.no_grad():
        for _ in range(warmup):
            block(x, t_emb)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with torch.no_grad():
        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,       # trace allocator / bandwidth pressure
            with_stack=False,
        ) as prof:
            for _ in range(active):
                with record_function("residual_block_step"):
                    block(x, t_emb)
            if device == "cuda":
                torch.cuda.synchronize()

    sort_key = "self_cuda_time_total" if device == "cuda" else "self_cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_key, row_limit=15))

    if device == "cuda":
        peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"\nPeak CUDA memory allocated: {peak:.2f} MiB")

    out_dir = os.path.dirname(os.path.abspath(__file__))
    trace_path = os.path.join(out_dir, "trace.json")
    prof.export_chrome_trace(trace_path)
    print(f"Chrome trace written -> {trace_path}  (open in chrome://tracing)")
    print("\nTakeaway: each of aten::add / mish(tanh+softplus+mul) / relu is a "
          "separate element-wise kernel — these are what triton_fused_sde.py "
          "collapses into one HBM round-trip.")


if __name__ == "__main__":
    main()
