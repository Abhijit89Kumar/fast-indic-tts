"""
triton_fused_sde.py
===================
Custom **fused Triton kernels** for the Grad-TTS reverse-diffusion (SDE/ODE)
sampling loop — the core optimization of the Diff-Turbo back-end.

WHY FUSE?
---------
The reverse sampler calls a U-Net `O(n_timesteps)` times (often 50–100). Inside
each U-Net residual block the dominant *memory-bound* pattern is:

        h = Conv1d(x)                 # compute-bound, leave to cuDNN
        h = h + time_emb              # element-wise  ┐ each of these is a
        h = Mish(h)                   # element-wise  ┤ separate CUDA kernel in
        out = h + residual           # element-wise  ┘ eager PyTorch

Every element-wise op in eager mode launches its own kernel and makes a full
round-trip to HBM (read inputs, write output). For a tensor of shape
(B, 80, T) that is 3 extra global-memory read/write sweeps per block, per step,
per layer. These ops are *bandwidth-bound*, so fusing them into ONE kernel that
reads once and writes once is close to a 3x reduction in memory traffic for the
element-wise portion.

WHAT WE FUSE
------------
    fused_add_mish(a, b)            -> Mish(a + b)              (2 ops, 1 kernel)
    fused_residual_add_mish(a,b,c)  -> Mish(a + b) + c         (3 ops, 1 kernel)

Mish(x) = x * tanh(softplus(x)),  softplus(x) = ln(1 + e^x)

These are the exact element-wise math of the Grad-TTS residual block, so the
fused kernels are drop-in replacements that preserve numerics (verified bitwise-
close in `verify_kernel.py`).

This module degrades gracefully: if Triton/CUDA is unavailable the public
functions fall back to eager PyTorch so the rest of the repo still imports.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = torch.cuda.is_available()
except Exception:  # pragma: no cover - environment dependent
    _HAS_TRITON = False


# =============================================================================
# Triton kernels
# =============================================================================
if _HAS_TRITON:

    @triton.jit
    def _fused_add_mish_kernel(
        a_ptr,          # *Pointer* to first input (e.g. conv output)
        b_ptr,          # *Pointer* to second input (e.g. time embedding/bias)
        out_ptr,        # *Pointer* to output
        n_elements,     # total number of elements (flattened)
        BLOCK_SIZE: tl.constexpr,
    ):
        """out = Mish(a + b), computed element-wise over a flat 1-D view.

        The diffusion feature maps are contiguous (B, C, T) tensors. We treat
        them as a flat 1-D array because the operation is purely element-wise —
        this gives perfectly *coalesced* loads/stores (consecutive threads read
        consecutive addresses), which is what makes a bandwidth-bound kernel hit
        peak HBM throughput.
        """
        # Each program (CUDA block) owns one contiguous BLOCK_SIZE-wide chunk.
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        # Guard the tail block: n_elements is rarely a multiple of BLOCK_SIZE.
        mask = offsets < n_elements

        a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
        b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
        x = a + b  # fused element-wise addition (op 1)

        # --- Numerically-stable Mish (op 2) -----------------------------------
        # softplus(x) = ln(1 + e^x). The naive form overflows for large x, so we
        # use the stable identity: softplus(x) = max(x,0) + ln(1 + e^(-|x|)).
        sp = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(x)))
        # Mish(x) = x * tanh(softplus(x))
        out = x * (2.0 / (1.0 + tl.exp(-2.0 * sp)) - 1.0)  # tanh via exp form

        tl.store(out_ptr + offsets, out, mask=mask)

    @triton.jit
    def _fused_residual_add_mish_kernel(
        a_ptr, b_ptr, c_ptr, out_ptr, n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """out = Mish(a + b) + c  — fuses the full residual-block tail.

        a = conv output, b = broadcast time embedding (pre-expanded), c = skip
        connection. Three element-wise ops collapsed into one HBM round-trip.
        """
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
        b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
        c = tl.load(c_ptr + offsets, mask=mask, other=0.0)

        x = a + b
        sp = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(x)))
        mish = x * (2.0 / (1.0 + tl.exp(-2.0 * sp)) - 1.0)
        out = mish + c

        tl.store(out_ptr + offsets, out, mask=mask)

    @triton.jit
    def _fused_mish_kernel(x_ptr, out_ptr, n_elements,
                           BLOCK_SIZE: tl.constexpr):
        """out = Mish(x) = x * tanh(softplus(x)).

        Grad-TTS implements Mish as `x * torch.tanh(F.softplus(x))`, which eager
        PyTorch executes as THREE separate bandwidth-bound kernels (softplus,
        tanh, mul). This single kernel does it in one HBM round-trip. Mish is
        called in every Block of the U-Net (many times per estimator call, x50
        diffusion steps), so this is the highest-frequency fusion in the model.
        """
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        sp = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(x)))  # softplus
        out = x * (2.0 / (1.0 + tl.exp(-2.0 * sp)) - 1.0)           # * tanh(sp)
        tl.store(out_ptr + offsets, out, mask=mask)

    @triton.jit
    def _sde_step_kernel(xt_ptr, mu_ptr, score_ptr, mask_ptr, out_ptr,
                         coeff, n_elements, BLOCK_SIZE: tl.constexpr):
        """Fused Grad-TTS reverse SDE/ODE Euler step (deterministic case):

            dxt = 0.5 * (mu - xt - score) * coeff      # coeff = noise_t * h
            xt_new = (xt - dxt) * mask

        Eager PyTorch runs this as ~6 element-wise kernels (two subtracts, three
        scalar multiplies, one mask multiply), each a full read/write of the
        (B, 80, T) mel tensor — and the reverse loop runs it `n_timesteps` (~50)
        times. We collapse it to ONE kernel. `coeff` is a scalar because the
        timestep `t` is identical across the batch at each step, so noise_t*h
        does not vary per element."""
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        xt = tl.load(xt_ptr + offsets, mask=mask, other=0.0)
        mu = tl.load(mu_ptr + offsets, mask=mask, other=0.0)
        sc = tl.load(score_ptr + offsets, mask=mask, other=0.0)
        m = tl.load(mask_ptr + offsets, mask=mask, other=0.0)
        dxt = 0.5 * (mu - xt - sc) * coeff
        out = (xt - dxt) * m
        tl.store(out_ptr + offsets, out, mask=mask)


# =============================================================================
# Block-size selection
# =============================================================================
# WHY BLOCK_SIZE = 1024?
#   * It must be a power of two so `tl.arange` maps cleanly onto warps (32
#     lanes); 1024 = 32 warps, a full, well-occupied CUDA block.
#   * The element-wise op is MEMORY-bound, not compute-bound, so we size the
#     block to maximize in-flight memory transactions, not register reuse.
#   * Audio mel feature maps are (B, C=80 or 100, T). After flattening, a
#     typical tensor is B*80*T elements — for B=16, T=128 that is ~163k
#     elements, i.e. ~160 blocks of 1024: plenty to saturate every SM.
#   * We deliberately DO NOT tile to the 80/100 channel dimension. Because the
#     op is element-wise we flatten instead, so the awkward, non-power-of-two
#     channel count (80, 100) never forces ragged, uncoalesced access. The only
#     ragged block is the final tail, handled by `mask`.
#   * 1024 keeps register pressure low enough for high occupancy while still
#     amortizing kernel-launch overhead across enough work per program.
_BLOCK_SIZE = 1024


def _grid(n_elements):
    """1-D launch grid: one program per BLOCK_SIZE-wide chunk (ceil-div)."""
    return (triton.cdiv(n_elements, _BLOCK_SIZE),)


# =============================================================================
# Public API (Triton-backed with eager fallback)
# =============================================================================
def _eager_mish(x: torch.Tensor) -> torch.Tensor:
    """Reference Mish in pure PyTorch (matches torch.nn.functional.mish)."""
    return x * torch.tanh(torch.nn.functional.softplus(x))


def fused_add_mish(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute Mish(a + b). Uses the Triton kernel on CUDA, else eager."""
    if not _HAS_TRITON or not a.is_cuda:
        return _eager_mish(a + b)

    a = a.contiguous()
    b = b.contiguous()
    out = torch.empty_like(a)
    n = out.numel()
    _fused_add_mish_kernel[_grid(n)](a, b, out, n, BLOCK_SIZE=_BLOCK_SIZE)
    return out


def fused_residual_add_mish(a: torch.Tensor, b: torch.Tensor,
                            c: torch.Tensor) -> torch.Tensor:
    """Compute Mish(a + b) + c. Triton-fused on CUDA, eager otherwise."""
    if not _HAS_TRITON or not a.is_cuda:
        return _eager_mish(a + b) + c

    a, b, c = a.contiguous(), b.contiguous(), c.contiguous()
    out = torch.empty_like(a)
    n = out.numel()
    _fused_residual_add_mish_kernel[_grid(n)](a, b, c, out, n,
                                              BLOCK_SIZE=_BLOCK_SIZE)
    return out


def fused_mish(x: torch.Tensor) -> torch.Tensor:
    """Compute Mish(x) in one fused kernel (eager fallback off-CUDA)."""
    if not _HAS_TRITON or not x.is_cuda:
        return _eager_mish(x)
    x = x.contiguous()
    out = torch.empty_like(x)
    n = out.numel()
    _fused_mish_kernel[_grid(n)](x, out, n, BLOCK_SIZE=_BLOCK_SIZE)
    return out


def fused_sde_step(xt: torch.Tensor, mu: torch.Tensor, score: torch.Tensor,
                   mask: torch.Tensor, coeff: float) -> torch.Tensor:
    """Fused Grad-TTS reverse Euler step: (xt - 0.5*(mu-xt-score)*coeff)*mask.

    `mask` is broadcast (B,1,T) over (B,80,T) in Grad-TTS; we expand it to the
    full shape once (one cheap copy) so the kernel stays a simple 1-D map. Even
    with that copy this is 1-2 kernels vs ~6 eager ones, run 50x."""
    if not _HAS_TRITON or not xt.is_cuda:
        return (xt - 0.5 * (mu - xt - score) * coeff) * mask

    xt = xt.contiguous()
    mu = mu.contiguous()
    score = score.contiguous()
    mask_full = mask.expand_as(xt).contiguous()
    out = torch.empty_like(xt)
    n = out.numel()
    _sde_step_kernel[_grid(n)](xt, mu, score, mask_full, out,
                               float(coeff), n, BLOCK_SIZE=_BLOCK_SIZE)
    return out


# =============================================================================
# Residual blocks: eager reference vs. Triton-fused
# =============================================================================
class EagerResidualBlock(torch.nn.Module):
    """Plain-PyTorch Grad-TTS-style 1-D residual block (the baseline).

    Structure mirrors Grad-TTS `ResnetBlock`: two Conv1d stages, a broadcast
    time embedding added after the first conv, Mish activations, and a residual
    skip. Every element-wise step is a separate eager CUDA kernel.
    """

    def __init__(self, channels: int, time_dim: int):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(channels, channels, 3, padding=1)
        self.conv2 = torch.nn.Conv1d(channels, channels, 3, padding=1)
        # Projects the diffusion-step embedding to a per-channel bias.
        self.time_proj = torch.nn.Linear(time_dim, channels)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # (B, C, T) conv; (B, C, 1) broadcast time bias.
        h = self.conv1(x)
        t = self.time_proj(t_emb).unsqueeze(-1)          # (B, C, 1)
        t = t.expand_as(h)                               # materialize for fusion
        # --- element-wise tail (the part we will fuse) ---
        h = _eager_mish(h + t)                           # add + Mish
        h = self.conv2(h)
        out = h + x                                       # residual add
        return out


class TritonFusedResidualBlock(torch.nn.Module):
    """Identical math to `EagerResidualBlock`, but the element-wise tail is
    executed by fused Triton kernels.

    The two convolutions stay on cuDNN (they are compute-bound and already
    optimal); only the bandwidth-bound element-wise ops are fused.
    """

    def __init__(self, channels: int, time_dim: int):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(channels, channels, 3, padding=1)
        self.conv2 = torch.nn.Conv1d(channels, channels, 3, padding=1)
        self.time_proj = torch.nn.Linear(time_dim, channels)

    def load_from(self, other: EagerResidualBlock) -> "TritonFusedResidualBlock":
        """Copy weights from an eager block so outputs are directly comparable."""
        self.conv1.load_state_dict(other.conv1.state_dict())
        self.conv2.load_state_dict(other.conv2.state_dict())
        self.time_proj.load_state_dict(other.time_proj.state_dict())
        return self

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        t = self.time_proj(t_emb).unsqueeze(-1).expand_as(h)
        # Fused add + Mish (single kernel, single HBM round-trip).
        h = fused_add_mish(h, t)
        h = self.conv2(h)
        # Fused residual add could be folded too; kept explicit for clarity and
        # because `x` here is the block input, not a third conv output.
        out = h + x
        return out


if __name__ == "__main__":
    # Smoke test: prove the public ops run on whatever device is available.
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] Triton active: {_HAS_TRITON} | device: {dev}")
    a = torch.randn(2, 80, 128, device=dev)
    b = torch.randn(2, 80, 128, device=dev)
    c = torch.randn(2, 80, 128, device=dev)
    print("fused_add_mish        ->", tuple(fused_add_mish(a, b).shape))
    print("fused_residual_add_mish ->", tuple(fused_residual_add_mish(a, b, c).shape))
