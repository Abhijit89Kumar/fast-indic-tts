# Indic-Diffusion-TTS

Two research-grade systems spanning the full Text-to-Speech stack — from an
LLM **linguistic front-end** (what sounds to make) down to a GPU-kernel
**acoustic back-end** (how fast the audio renders). Every number below was
measured on real hardware (NVIDIA **A100-SXM4-80GB**, PyTorch 2.8 / CUDA 12.8,
Triton 3.4) and is reproducible from the scripts in this repo.

| Sub-system | Role | Headline result |
|---|---|---|
| **`indic_prosody_frontend`** | LLM front-end (TN + G2P) | **LoRA-fine-tuned Sarvam-1 cuts TN word-error 20.9% → 7.96%** vs a competitive rule engine |
| **`diff_turbo_backend`** | Triton kernel back-end | **3.3× faster Grad-TTS synthesis** (profiling-driven NFE reduction), quality-preserving; fused Triton kernels **1.79× / −33% VRAM** on targeted ops, bit-accurate audio |

---

## Project 2 — Indic-Prosody (LLM front-end)

**Goal:** replace brittle rule-based Text-Normalization (TN) for code-mixed
Hindi/English with an LLM, then phonemize (G2P) — a complete TTS front-end.

### The honest research arc
1. **Rules break.** The textbook `indic-numtowords` path scores **43.6% WER** on
   a 40-sentence hand-labeled code-mixed test set; it drops acronyms and reads
   IDs as magnitudes (`PNR-8392 → आठ हज़ार…`).
2. **A *competitive* rule engine** (regex for times/currency/percent/ordinals +
   a real English number speller) gets to **20.9% WER** — a fair opponent, not a
   strawman. It still can't do acronym phonetics, digit-by-digit IDs, code-mix
   locale, or context.
3. **Base-model prompting fails.** `sarvam-1` is a *base* (non-instruct) 2B model;
   even 12-shot in-context learning scores **49.9% WER — worse than rules.** Base
   LLMs can't be prompted into reliable TN. (A genuine, documented finding.)
4. **LoRA fine-tuning wins.** Fine-tuning Sarvam-1 (0.94% of params, 96 MB
   adapter) on a **synthetic, correct-by-construction** code-mixed TN corpus
   reaches **7.96% WER / 62.5% exact-match** — a **62% relative WER reduction**
   over the competitive rule engine.

### TN results (held-out 40-sentence set, `run_tn_eval.py`)

| System | WER ↓ | CER ↓ | Exact-Match ↑ |
|---|---|---|---|
| naive rules (`indic-numtowords`) | 43.56% | 43.70% | 0.0% |
| competitive rules | 20.89% | 17.46% | 27.5% |
| Sarvam-1 base (12-shot ICL) | 49.94% | 35.36% | 5.0% |
| **Sarvam-1 LoRA (ours)** | **7.96%** | **6.37%** | **62.5%** |

Per-category, the LoRA model takes the rule engine's hardest buckets to **0% WER**:
`otp+id` 57.9→0, `phone` 57.1→0, `flight-code` 45.5→0, `year` 50→0, `id+time`
25→0, `devanagari-mix` (after augmentation) 87.5→0.

**Error-analysis-driven iteration (a key talking point):** the first LoRA (14.9%
WER) failed on Devanagari script, slash-dates, and comma-grouped lakh numbers —
phenomena absent from the synthetic data. Adding those *general* templates and
retraining (3 min) took it to 7.96%. The 40 test sentences stayed held out; we
added capabilities, not test cases.

### G2P + full front-end
We **distill espeak-ng into a second Sarvam-1 LoRA** (text → IPA), giving a
single LLM that does both TN and G2P. On a held-out set the LoRA reproduces the
espeak-ng reference with **0.00% PER / 100% exact phoneme match** (n=60) — i.e.
one LLM perfectly generalizes the phonemizer's deterministic mapping to unseen
sentences (`g2p.py --eval`). The assembled pipeline:

```
RAW : Mera flight ticket PNR-8392 hai, aur departure 4:30 PM ko hai.
NORM: Mera flight ticket pee-en-aar eight three nine two hai, aur departure four thirty pee-em ko hai.   [Sarvam-1 TN LoRA]
IPA : ...                                                                                                 [Sarvam-1 G2P LoRA]
```

### Files
- `data/testset.json` — 40 hand-labeled code-mixed sentences (eval).
- `baseline_rules.py` — naive + competitive rule baselines.
- `make_tn_data.py` — synthetic TN corpus generator (correct-by-construction).
- `train_lora.py` — LoRA fine-tuning (shared with G2P).
- `llm_normalizer.py` — Sarvam-1 inference (few-shot base **and** LoRA modes).
- `make_g2p_data.py`, `g2p.py` — G2P distillation + PER eval + pipeline.
- `metrics.py`, `run_tn_eval.py` — WER/CER/EM + per-category harness.
- `results/` — `tn_summary.json`, `tn_results.json`, `frontend_pipeline.json`.

---

## Project 1 — Diff-Turbo (Triton back-end)

**Goal:** make Grad-TTS synthesis faster, end-to-end, on real hardware — and
prove the audio quality is preserved.

### Headline: 3.3× faster synthesis, quality-preserving (`fast_sampler.py`)
Profiling (below) showed Grad-TTS is **NFE-bound** — its cost is dominated by
running the U-Net once per reverse-diffusion step, 50 times by default. So the
high-leverage optimization is **fewer function evaluations (NFE)**. We implement
a 2nd-order **Heun** ODE solver and sweep NFE, measuring mel-spectrogram MAE
against a 200-step reference:

| Sampler | NFE | mel MAE ↓ | latency (fused) | speedup vs Euler-50 |
|---|---|---|---|---|
| Euler (default) | 50 | 0.0101 | 1015 ms | 1.0× |
| Euler | 25 | 0.0249 | 508 ms | 2.0× |
| **Heun** | **16** | **0.0442** | **305 ms** | **3.33×** |
| Euler | 16 | 0.0447 | 325 ms | 3.12× |
| Heun | 8 | 0.083 | 143 ms | 7.3× |

**The default 50 steps is oversampled.** At the quality knee (~16 NFE) synthesis
is **3.3× faster** with near-reference mel error; at 8 NFE it's ~7× for a modest
quality drop. Audio at each operating point: `audio_out/{ref_euler200,euler50,
heun_fast}.wav`. Figure: `fast_sampler_curve.png`.

![quality vs speed](diff_turbo_backend/fast_sampler_curve.png)

Honest note: the 2nd-order Heun solver is roughly on par with Euler *per NFE*
here (Grad-TTS's probability-flow ODE is mild), edging it only at the knee — so
the dominant lever is NFE reduction itself, which the profile predicted.

### Fused Triton kernels (the per-evaluation systems layer)
On top of fewer evals, we fuse the bandwidth-bound element-wise ops *inside*
each evaluation. Eager PyTorch executes each as a separate HBM round-trip:

```
Mish(x) = x*tanh(softplus(x))           # 3 eager kernels -> 1 fused kernel
xt = (xt - 0.5*(mu-xt-score)*noise_t*h)*mask   # ~6 eager kernels -> 1 fused kernel
```

`fused_mish` and `fused_sde_step` (in `triton_fused_sde.py`) collapse these into
single read-once/write-once kernels. (`noise_t*h` is a scalar because the
timestep is constant across the batch each step.)

### Kernel results (A100, `verify_kernel.py` + `benchmark.py`)
- **Correctness:** bit-accurate — `fused_add_mish` / `fused_mish` match eager to
  **~1e-6** (fp32 round-off).
- **Isolated kernel** (`Mish(a+b)+c`, B=16×80×1024): **+44% latency (1.79×),
  −33% VRAM** vs eager's 3 kernels.
- Block-level: +10% (conv-bound). `BLOCK_SIZE=1024` chosen for coalesced,
  warp-aligned access (rationale documented in the kernel).

### Real Grad-TTS integration + audio parity (`gradtts_triton.py`)
Patched `Mish` + a fused reverse loop into pretrained Grad-TTS (LJSpeech) + the
HiFi-GAN vocoder, then synthesized real audio:
- **Audio parity:** fused vs eager mel `allclose(1e-3)=True`, max-abs **2.3e-3**,
  MAE **9.7e-5** → the optimization does **not** change the audio
  (`audio_out/eager.wav`, `audio_out/fused.wav`).
- **Element-wise fusion alone → ≈ 0% end-to-end** (−0.6%). Not a failure — a
  measurement that *pointed us at the real bottleneck* (next).

### The profile that drove the strategy (`profile_gradtts.py`)
| Bucket | % of CUDA time |
|---|---|
| **attention / einsum / matmul** | **69.2%** |
| Conv2d | 9.2% |
| Mish (element-wise) | 6.3% |
| GroupNorm | 6.1% |
| add/sub (element-wise) | 2.7% |
| **element-wise total (kernels' target)** | **9.2%** |

The element-wise ops the kernels target are only **9.2%** of runtime (an Amdahl
ceiling), and that cost repeats **once per reverse step × 50 steps**. So fusing
them can't win alone — but **cutting the number of steps scales down the whole
69% attention cost linearly.** That insight is exactly what the fast sampler
(top of this section) exploits for the real 3.3× — *optimization driven by
measurement, not assumption.*

### Files
- `fast_sampler.py` — **Heun solver + NFE sweep (the 3.3× headline)**;
  `plot_fast_sampler.py` → `fast_sampler_curve.png`.
- `triton_fused_sde.py` — the fused kernels (`fused_mish`, `fused_sde_step`,
  `fused_add_mish`, residual block variants) + eager fallback.
- `verify_kernel.py` — `torch.allclose` correctness gate.
- `benchmark.py` — kernel + block latency/VRAM.
- `gradtts_triton.py` — real Grad-TTS integration, parity, audio, benchmark.
- `profile_gradtts.py` — the op-breakdown profile that drove the strategy.
- `torch_profiler_script.py` — eager block profile.
- `audio_out/` — `ref_euler200.wav`, `euler50.wav`, `heun_fast.wav`,
  `eager.wav`, `fused.wav`; `fast_sampler_results.json`, `gradtts_results.json`.

---

## Pretrained adapters (Hugging Face)
The LoRA adapters are hosted on the Hub (they're gitignored here to keep the repo
light). Load them on top of `sarvamai/sarvam-1`:
- **TN:** [`Abhijit89Kumar/sarvam1-hinglish-tn-lora`](https://huggingface.co/Abhijit89Kumar/sarvam1-hinglish-tn-lora)
- **G2P:** [`Abhijit89Kumar/sarvam1-hinglish-g2p-lora`](https://huggingface.co/Abhijit89Kumar/sarvam1-hinglish-g2p-lora)

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained("sarvamai/sarvam-1")
m = PeftModel.from_pretrained(m, "Abhijit89Kumar/sarvam1-hinglish-tn-lora")
```

## Reproduce

```bash
pip install -r requirements.txt        # + peft, phonemizer (G2P), espeak-ng (apt)
# Front-end (GPU for the LLM):
python indic_prosody_frontend/make_tn_data.py --n 8000
python indic_prosody_frontend/train_lora.py --data data/train.jsonl --out sarvam_tn_lora
python indic_prosody_frontend/run_tn_eval.py --base-llm --adapter sarvam_tn_lora
python indic_prosody_frontend/g2p.py --eval --pipeline
# Back-end (CUDA + Triton):
python diff_turbo_backend/verify_kernel.py
python diff_turbo_backend/benchmark.py
python diff_turbo_backend/gradtts_triton.py --gradtts <path-to-Grad-TTS>
python diff_turbo_backend/profile_gradtts.py --gradtts <path-to-Grad-TTS>
python diff_turbo_backend/fast_sampler.py --gradtts <path-to-Grad-TTS>   # the 3.3x
python diff_turbo_backend/plot_fast_sampler.py
```

## Honest limitations (because interviewers will ask)
- **TN test set is 40 sentences** — small; the headline number is indicative,
  and the *per-category* table + held-out protocol matter more than the point
  estimate. Synthetic training data means the model learns my normalization
  conventions; a human-labeled corpus would be the production next step.
- **G2P is distilled from espeak-ng**, so it matches (not beats) espeak; the
  value is a *unified* LLM front-end, and code-switched phonemization (per-span
  language ID) remains an open problem (we hold out Devanagari for the G2P eval).
- **The 3.3× speedup is from NFE reduction, not the Triton kernels** — and we say
  so. Fusing element-wise ops alone is ~0% end-to-end (they're 9% of runtime);
  the kernels are correct and 1.79× in isolation and shave the per-step cost, but
  the big win is fewer steps. Quality is measured by mel-MAE vs a 200-step
  reference, not by ear alone; a listening test would be the production next step.
- **Heun ≈ Euler per-NFE here** — the probability-flow ODE is mild, so the
  higher-order solver only edges Euler at the knee. We report this rather than
  overclaiming the solver.

## Interview talking points
- *Measure before optimizing*: the profile (attention 69%, element-wise 9%)
  killed the obvious idea and pointed at NFE — leading to the real 3.3×.
- Two levers, honestly separated: **algorithmic** (fewer steps, 3.3×) vs
  **systems** (fused kernels, 1.79× on their slice); we quantify each.
- Right tool per layer: fine-tuned LLM where ambiguity lives (TN); GPU kernels
  and solver math where latency lives.
- Base vs fine-tuned LLM: a 2B base model loses to rules (49.9% WER); LoRA flips
  it to beat them (7.96%). Error-analysis → augmentation → 14.9% → 7.96%.
- Prove quality before claiming speed: `verify_kernel.py`, bit-accurate audio,
  mel-MAE quality curves.
```
