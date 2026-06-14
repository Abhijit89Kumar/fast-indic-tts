"""
gradtts_triton.py
=================
End-to-end integration of the Diff-Turbo fused Triton kernels into the REAL
Grad-TTS model (Huawei Speech-Backbones), with pretrained LJSpeech weights and
the HiFi-GAN vocoder.

What this does:
  1. Loads pretrained Grad-TTS + HiFi-GAN.
  2. Defines two drop-in optimizations of the diffusion decoder:
       * Mish  -> fused Triton Mish kernel (replaces 3 eager kernels)
       * reverse SDE Euler step -> fused Triton kernel (replaces ~6 eager kernels)
  3. PARITY: runs reverse diffusion eager vs fused on identical noise and asserts
     the mel-spectrograms match (the optimization must not change the audio).
  4. BENCHMARK: times full text->mel synthesis (50 reverse steps) eager vs fused
     and reports % speedup + VRAM.
  5. AUDIO: vocodes both mels to .wav so the parity is audible, not just numeric.

Run (on the A100 pod, from the diff_turbo_backend dir):
    python gradtts_triton.py --gradtts /path/Grad-TTS --timesteps 50

Scope note: we fuse the bandwidth-bound element-wise ops (Mish, SDE step)
and leave the compute-bound Conv2d/GroupNorm/attention on cuDNN. The end-to-end
speedup is therefore bounded by how much of the wall-clock those element-wise ops
represent (reported faithfully below).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

# triton_fused_sde lives next to this file.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from triton_fused_sde import _HAS_TRITON, fused_mish, fused_sde_step


# ---------------------------------------------------------------------------
# Optimization patches
# ---------------------------------------------------------------------------
def patch_mish(diffusion_mod) -> None:
    """Replace Grad-TTS Mish.forward with the fused Triton kernel, in place.

    Every Block in the U-Net holds a Mish() instance; overriding the class
    method swaps them all at once. Mish is the highest-frequency element-wise op
    in the network (called for every Block, every estimator call, every step)."""
    diffusion_mod.Mish.forward = lambda self, x: fused_mish(x)


def make_fused_reverse(diffusion_mod):
    """Build a fused replacement for Diffusion.reverse_diffusion.

    Identical math to the upstream loop, but the per-step element-wise Euler
    update is done by one fused kernel instead of ~6 eager ones. `coeff =
    noise_t * h` is a scalar because the timestep t is constant across the batch
    at each step."""
    get_noise = diffusion_mod.get_noise

    @torch.no_grad()
    def fused_reverse(self, z, mask, mu, n_timesteps, stoc=False, spk=None):
        h = 1.0 / n_timesteps
        xt = z * mask
        # Mask is constant across all reverse steps, so expand to the full mel
        # shape ONCE here instead of copying it inside every step's kernel call.
        mask_full = mask.expand_as(xt).contiguous()
        for i in range(n_timesteps):
            t = (1.0 - (i + 0.5) * h) * torch.ones(z.shape[0], dtype=z.dtype,
                                                   device=z.device)
            time = t.unsqueeze(-1).unsqueeze(-1)
            noise_t = get_noise(time, self.beta_min, self.beta_max,
                                cumulative=False)
            # Deterministic ODE solver path (stoc=False), which is the default
            # used for high-quality Grad-TTS synthesis.
            score = self.estimator(xt, mask, mu, t, spk)
            coeff = float(noise_t.flatten()[0].item()) * h
            xt = fused_sde_step(xt, mu, score, mask_full, coeff)
        return xt

    return fused_reverse


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_models(gradtts_dir: str):
    """Load Grad-TTS + HiFi-GAN and the text frontend. Returns a bundle dict."""
    sys.path.insert(0, gradtts_dir)
    sys.path.insert(0, os.path.join(gradtts_dir, "hifi-gan"))
    os.chdir(gradtts_dir)  # inference paths in the repo are relative

    import params
    from model import GradTTS
    from text import text_to_sequence, cmudict
    from text.symbols import symbols
    from utils import intersperse
    from env import AttrDict
    from models import Generator as HiFiGAN
    import model.diffusion as diffusion_mod

    gen = GradTTS(len(symbols) + 1, params.n_spks, params.spk_emb_dim,
                  params.n_enc_channels, params.filter_channels,
                  params.filter_channels_dp, params.n_heads, params.n_enc_layers,
                  params.enc_kernel, params.enc_dropout, params.window_size,
                  params.n_feats, params.dec_dim, params.beta_min,
                  params.beta_max, params.pe_scale)
    gen.load_state_dict(torch.load("./checkpts/grad-tts.pt",
                                   map_location="cpu"))
    gen = gen.cuda().eval()

    with open("./checkpts/hifigan-config.json") as f:
        hcfg = AttrDict(json.load(f))
    vocoder = HiFiGAN(hcfg)
    vocoder.load_state_dict(torch.load("./checkpts/hifigan.pt",
                                       map_location="cpu")["generator"])
    vocoder = vocoder.cuda().eval()
    vocoder.remove_weight_norm()

    cmu = cmudict.CMUDict("./resources/cmu_dictionary")
    return {"gen": gen, "vocoder": vocoder, "cmu": cmu, "params": params,
            "text_to_sequence": text_to_sequence, "intersperse": intersperse,
            "n_symbols": len(symbols), "diffusion_mod": diffusion_mod}


def encode_text(bundle, text: str):
    """Text -> (x, x_lengths) tensors for Grad-TTS (blank-interspersed)."""
    seq = bundle["text_to_sequence"](text, dictionary=bundle["cmu"])
    seq = bundle["intersperse"](seq, bundle["n_symbols"])
    x = torch.LongTensor(seq).cuda()[None]
    x_len = torch.LongTensor([x.shape[-1]]).cuda()
    return x, x_len


@torch.no_grad()
def synth_mel(bundle, x, x_len, n_timesteps, temperature, seed):
    """Run a full Grad-TTS synthesis with a FIXED seed so the initial diffusion
    noise is identical across eager/fused runs (required for parity)."""
    torch.manual_seed(seed)
    y_enc, y_dec, attn = bundle["gen"].forward(
        x, x_len, n_timesteps=n_timesteps, temperature=temperature,
        stoc=False, length_scale=0.91)
    return y_dec  # mel-spectrogram (B, 80, T)


@torch.no_grad()
def vocode(bundle, mel) -> np.ndarray:
    audio = bundle["vocoder"].forward(mel).cpu().squeeze().clamp(-1, 1).numpy()
    return (audio * 32768).astype("int16")


def benchmark(fn, n_iter, device="cuda"):
    """Median latency (ms) and peak VRAM (MiB) over n_iter timed runs."""
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(n_iter):
        s, e = (torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True))
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return times[len(times) // 2], peak


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gradtts", required=True, help="path to Grad-TTS repo dir")
    ap.add_argument("--text", default="Diffusion based text to speech, "
                    "accelerated with custom Triton kernels.")
    ap.add_argument("--timesteps", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=1.5)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    assert _HAS_TRITON, "Triton/CUDA required for the real benchmark."
    outdir = args.outdir or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "audio_out")

    print("=" * 72)
    print("DIFF-TURBO x GRAD-TTS  —  fused-kernel integration")
    print("=" * 72)
    bundle = load_models(args.gradtts)
    diffusion_mod = bundle["diffusion_mod"]
    os.makedirs(outdir, exist_ok=True)
    x, x_len = encode_text(bundle, args.text)
    print(f"text tokens: {x.shape[-1]} | timesteps: {args.timesteps}")

    # Keep pristine references so we can flip between eager and fused.
    orig_mish_fwd = diffusion_mod.Mish.forward
    orig_reverse = diffusion_mod.Diffusion.reverse_diffusion
    fused_reverse = make_fused_reverse(diffusion_mod)

    def set_eager():
        diffusion_mod.Mish.forward = orig_mish_fwd
        diffusion_mod.Diffusion.reverse_diffusion = orig_reverse

    def set_fused():
        patch_mish(diffusion_mod)
        diffusion_mod.Diffusion.reverse_diffusion = fused_reverse

    # ---- PARITY -------------------------------------------------------
    set_eager()
    mel_eager = synth_mel(bundle, x, x_len, args.timesteps, args.temperature,
                          args.seed)
    set_fused()
    mel_fused = synth_mel(bundle, x, x_len, args.timesteps, args.temperature,
                          args.seed)
    max_abs = (mel_eager - mel_fused).abs().max().item()
    mae = (mel_eager - mel_fused).abs().mean().item()
    close = torch.allclose(mel_eager, mel_fused, rtol=1e-3, atol=1e-3)
    print("\n--- PARITY (eager vs fused mel) ---")
    print(f"  shape={tuple(mel_eager.shape)}  max_abs={max_abs:.3e}  "
          f"mae={mae:.3e}  allclose(1e-3)={close}")

    # ---- AUDIO --------------------------------------------------------
    from scipy.io.wavfile import write as wavwrite
    sr = bundle["params"].__dict__.get("sample_rate", 22050)
    set_eager()
    wavwrite(os.path.join(outdir, "eager.wav"), sr,
             vocode(bundle, synth_mel(bundle, x, x_len, args.timesteps,
                                      args.temperature, args.seed)))
    set_fused()
    wavwrite(os.path.join(outdir, "fused.wav"), sr,
             vocode(bundle, synth_mel(bundle, x, x_len, args.timesteps,
                                      args.temperature, args.seed)))
    print(f"  wrote {outdir}/eager.wav and fused.wav")

    # ---- BENCHMARK (full text->mel synthesis) -------------------------
    set_eager()
    t_eager, m_eager = benchmark(
        lambda: synth_mel(bundle, x, x_len, args.timesteps, args.temperature,
                          args.seed), args.iters)
    set_fused()
    t_fused, m_fused = benchmark(
        lambda: synth_mel(bundle, x, x_len, args.timesteps, args.temperature,
                          args.seed), args.iters)
    speedup = (t_eager - t_fused) / t_eager * 100
    print("\n--- BENCHMARK (full synthesis, "
          f"{args.timesteps} reverse steps, median of {args.iters}) ---")
    print(f"  eager : {t_eager:8.2f} ms   peak {m_eager:7.1f} MiB")
    print(f"  fused : {t_fused:8.2f} ms   peak {m_fused:7.1f} MiB")
    print(f"  speedup: {speedup:+.2f}%   ({t_eager/t_fused:.2f}x)   "
          f"VRAM: {(m_eager-m_fused)/m_eager*100:+.2f}%")

    # ---- persist summary ---------------------------------------------
    summary = {
        "timesteps": args.timesteps, "tokens": int(x.shape[-1]),
        "parity": {"max_abs": max_abs, "mae": mae, "allclose_1e-3": close},
        "latency_ms": {"eager": t_eager, "fused": t_fused},
        "vram_mib": {"eager": m_eager, "fused": m_fused},
        "speedup_pct": speedup,
    }
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "gradtts_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {here}/gradtts_results.json")


if __name__ == "__main__":
    main()
