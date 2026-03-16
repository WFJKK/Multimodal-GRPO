# Can RL Teach Vision-Language Models to See Spatially?

**TL;DR:** We train a VLM (Qwen2.5-VL-7B) to estimate physical distances from images using reinforcement learning. GRPO discovers scale bar usage that nobody explicitly taught. SFT→RL combines the best of both: SFT's output precision + RL's spatial reasoning. Matched pair diagnostics causally verify what the model learned.

## Key Results

| Method | MAE (mm) | Scale Bar Usage | What It Learned |
|---|---|---|---|
| Baseline (no training) | 7.72 | ✗ Ignores | Random guessing |
| CoT GRPO | 5.42 | ✗ Ignores | Better pixel guessing, not spatial reasoning |
| SFT | 3.08 | ✓ Partial (r=0.58) | Precise decimals + some scale bar use |
| GRPO answer-only | 3.61 | ✓ Strong (r=0.75) | Consistent scale bar use, but rounds to 5mm |
| **SFT → RL** | **2.29** | **✓ Strong (r=0.65)** | **Precise output + spatial reasoning** |

**Pixel regression floor: 5.86mm** — anything below this requires using the scale bar.
**Perfect (scale bar formula): 0.007mm** — the theoretical ceiling.

## Three Findings

**1. RL discovers spatial reasoning that nobody programmed.**
GRPO, with nothing but a "your number was wrong" reward signal, taught the model to relate hole size to scale bar size. The matched pair diagnostic proves this causally: same hole pixels + different scale bar → model gives different answers (r=0.75). No explicit supervision on *how* to use the scale bar.

**2. SFT and RL learn different things.**
SFT produces precise decimals (17.64mm) but ignores the scale bar 21% of the time. GRPO always uses the scale bar but outputs round numbers (20mm). They solve different subproblems: SFT learns output calibration, RL learns spatial consistency. This decomposition is invisible to MAE — only the matched pair diagnostic reveals it.

**3. Chain-of-thought doesn't help spatial perception.**
CoT GRPO (5.42mm) performed worse than answer-only GRPO (3.61mm) despite using more compute. The model generated plausible reasoning ("compare the circle to the scale bar") but it didn't translate to better answers. This aligns with MeasureBench's finding: spatial perception requires better visual decoding, not more reasoning tokens.

## Methodology: Shortcut-Proof Evaluation

Most spatial benchmarks have exploitable shortcuts. We designed against them:

**Anti-shortcut dataset.** Continuous uniform diameters (3–30mm), variable zoom independent of diameter, 8 different scale bar values, no text leaks. Every possible shortcut is verified:

```
Correlation with ground truth diameter:
  hole_pixels alone                    r = +0.44  (not enough)
  scale_bar_mm alone                   r = -0.00  (useless)
  CORRECT: hole_px × sb_mm / sb_px    r = +1.00  (only path)
```

**Matched pair diagnostic.** 40 image pairs where the hole is pixel-identical but the scale bar differs. If the model gives the same answer for both → not using the scale bar. If answers diverge correctly → causally proven scale bar usage. This is a general-purpose tool for testing whether a VLM uses any specific visual feature.

## Architecture

```
Task:    Image of hole + scale bar  →  diameter in mm
Model:   Qwen2.5-VL-7B-Instruct + LoRA (rank 64)
Reward:  -|predicted - ground_truth| / ground_truth  (continuous)
Method:  Custom GRPO loop (4 generations, per-completion backward)
Compute: A100 80GB, ~$5 per experiment
```

The correct strategy the model must discover:
```
diameter_mm = hole_pixels × (scale_bar_mm / scale_bar_pixels)
```

No part of this formula is provided during training. The model receives only images and scalar reward.

## Experimental Design

Five conditions, all on the same 1000-image shortcut-proof dataset, same model, same LoRA config:

```
Baseline ──────────────────────────── Qwen2.5-VL-7B (no training)
SFT ───────────────────────────────── Supervised on ground truth answers
GRPO answer-only ──────────────────── RL with spatial reward, 32 tokens
GRPO CoT ──────────────────────────── RL with spatial reward, 128 tokens (reasoning)
SFT → RL ──────────────────────────── SFT first, then GRPO on top
```

All evaluated on 200 held-out test images + 40 matched pairs.

## What This Connects To

**MeasureBench (Oct 2025)** applied GRPO to Qwen2.5-VL-7B for gauge reading. They found RL helps on synthetic but not real images, and asked whether architectural changes are needed. We extend their work with shortcut-proof evaluation and show RL learns genuine spatial reasoning — the architecture is sufficient, the training signal matters.

**SpatialVLM (CVPR 2024)** found VLMs' spatial limitations come from datasets, not architecture — only 37.2% of distance estimates were within acceptable range. Our SFT→RL pipeline cuts MAE from 7.72mm to 2.29mm without any architectural changes, supporting their hypothesis.

**VLAA-Thinker (2025)** found SFT before GRPO degrades reasoning performance. Our results are more nuanced: for spatial tasks, SFT→RL outperforms either alone because they solve complementary subproblems.

## Reproducing

```bash
# Full experiment: ~$5 on A100 80GB
git clone https://github.com/WFJKK/spatial-perception-rl.git
cd spatial-perception-rl
bash setup.sh                    # Generate shortcut-proof dataset
python3 train_sft.py             # SFT baseline (~30 min)
python3 train_grpo_custom.py     # GRPO answer-only (~3 hrs)
python3 train_grpo_from_sft.py   # SFT→RL pipeline (~2.5 hrs)
python3 evaluate.py --compare    # Full comparison table
```

All results are deterministic (seed 42) and the dataset generator verifies no shortcuts exist before producing images.

## Limitations

- **No KL penalty** in GRPO — leads to mode collapse (round numbers in standalone GRPO, repeated values in SFT→RL). Standard fix, not implemented.
- **Synthetic only** — no real-world transfer test. The matched pair methodology transfers but the trained models may not.
- **Single task** — hole diameter measurement. The methodology generalizes but we haven't demonstrated it on other spatial tasks.
- **Compute not matched** — GRPO uses ~10x more compute than SFT per step. Multi-epoch SFT might close the gap without RL.
- **CoT ran at 500/1000 steps** — partial training, results may improve with full run.

## Repository Structure

```
generate_dataset.py        # Shortcut-proof dataset generator with verification
train_sft.py               # Supervised fine-tuning baseline
train_grpo_custom.py       # Custom GRPO loop (answer-only)
train_grpo_cot.py          # GRPO with chain-of-thought
train_grpo_from_sft.py     # SFT→RL pipeline
evaluate.py                # Evaluation + matched pair diagnostic
results/                   # All experimental results (JSON)
```

## Citation

```
@misc{kames2026spatial,
  title={Can RL Teach Vision-Language Models to See Spatially?},
  author={Kames, Joshua},
  year={2026},
  url={https://github.com/WFJKK/spatial-perception-rl}
}
```
