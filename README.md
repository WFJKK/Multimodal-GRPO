# Can RL Teach Vision-Language Models to See Spatially?

**TL;DR:** We train a VLM (Qwen2.5-VL-7B) to estimate physical distances from images using reinforcement learning. GRPO discovers scale bar usage that nobody explicitly taught. SFT→RL combines the best of both: SFT's output precision + RL's spatial reasoning. Chain-of-thought reasoning is unfaithful — the model writes correct-sounding spatial descriptions that don't match its actual visual processing. Matched pair diagnostics causally verify what the model learned.

## Key Results

| Method | MAE (mm) | Scale Bar Usage | What It Learned |
|---|---|---|---|
| Baseline (no training) | 7.72 | ✗ Ignores | Random guessing |
| CoT GRPO | 5.42 | ✗ Ignores | Unfaithful reasoning (see below) |
| SFT | 3.08 | ✓ Partial (r=0.58) | Precise decimals + some scale bar use |
| GRPO answer-only | 3.61 | ✓ Strong (r=0.75) | Consistent scale bar use, but rounds to 5mm |
| **SFT → RL** | **2.29** | **✓ Strong (r=0.65)** | **Precise output + spatial reasoning** |

**Pixel regression floor: 5.86mm** — anything below this requires using the scale bar.
**Perfect (scale bar formula): 0.007mm** — the theoretical ceiling.

## Four Findings

### 1. RL discovers spatial reasoning that nobody programmed.
GRPO, with nothing but a "your number was wrong" reward signal, taught the model to relate hole size to scale bar size. The matched pair diagnostic proves this causally: same hole pixels + different scale bar → model gives different answers (r=0.75). No explicit supervision on *how* to use the scale bar.

### 2. SFT and RL learn different things.
SFT produces precise decimals (17.64mm) but ignores the scale bar 21% of the time. GRPO always uses the scale bar but outputs round numbers (20mm). They solve different subproblems: SFT learns output calibration, RL learns spatial consistency. This decomposition is invisible to MAE — only the matched pair diagnostic reveals it.

### 3. Chain-of-thought reasoning is unfaithful for spatial perception.
CoT GRPO (5.42mm) performed worse than answer-only GRPO (3.61mm) despite more compute. Analysis of the reasoning traces reveals why:

**When the model gets it right** (gt=11.73mm, answer=12mm):
> "The circle appears to be slightly less than half the length of the 25mm bar."

**When the model gets it wrong** (gt=5.03mm, answer=15mm):
> "The hole appears to be approximately half the length of the scale bar."

The model reads the scale bar label correctly **100% of the time** and always attempts proportional reasoning. But the proportion estimates are unreliable — "half" is a learned template, not a faithful description of what the model perceives. When the hole happens to be ~half the scale bar, the template works. When it's actually 1/6 the scale bar, the model still writes "half" and gets it catastrophically wrong.

**The reasoning strategy evolves over training** but never becomes faithful:

| Training Phase | Dominant Strategy |
|---|---|
| Steps 1–50 | "Measure the diameter" (92%), "multiply" (58%) |
| Steps 50–200 | "Compare size" emerges (36%), less multiplication |
| Steps 200–400 | "Compare size" grows (44%), "measure" declines (23%) |
| Steps 400+ | "Compare size" dominates (78%), "estimate" emerges (63%) |

The model discovers that comparing sizes is the right approach, but it cannot verbalize the comparison faithfully. This is the same pattern found in our earlier SFT compliance experiment: models say "using the scale bar" even on blank images. **Unfaithful spatial reasoning persists regardless of training method (SFT or RL).**

This aligns with MeasureBench's finding that chain-of-thought doesn't improve spatial perception, and extends it: CoT fails not because the model doesn't try to reason spatially, but because verbal descriptions of spatial proportions are unreliable.

### 4. SFT→RL is complementary, not redundant.
The standard pipeline (SFT first, then RL) achieves the best result (2.29mm) because SFT teaches precise decimal output and RL refines spatial reasoning. Neither alone achieves both. GRPO's mode collapse toward round numbers (83% multiples of 5) is mitigated by SFT's prior for precise values.

## Methodology: Shortcut-Proof Evaluation

Most spatial benchmarks have exploitable shortcuts. We designed against them:

**Anti-shortcut dataset.** Continuous uniform diameters (3–30mm), variable zoom independent of diameter, 8 different scale bar values, no text leaks. Every possible shortcut is verified:

```
Correlation with ground truth diameter:
  hole_pixels alone                    r = +0.44  (not enough)
  scale_bar_mm alone                   r = -0.00  (useless)
  CORRECT: hole_px × sb_mm / sb_px    r = +1.00  (only path)
```

**Matched pair diagnostic.** 40 image pairs where the hole is pixel-identical but the scale bar differs. If the model gives the same answer for both → not using the scale bar. If answers diverge correctly → causally proven scale bar usage. This is a general-purpose method for testing whether a VLM uses any specific visual feature.

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

## Connection to Prior Work

**MeasureBench (Oct 2025)** applied GRPO to Qwen2.5-VL-7B for gauge reading. They found RL helps on synthetic but not real images, concluded that better visual encoding may be needed, and did not compare RL to SFT. We extend their work with shortcut-proof evaluation, an SFT comparison, and evidence that RL and SFT learn complementary skills.

**SpatialVLM (CVPR 2024)** argued VLMs' spatial limitations come from datasets, not architecture. Our results partially support this: SFT→RL cuts MAE from 7.72mm to 2.29mm without architectural changes. But the CoT failure suggests a limit — the model can improve through training but cannot faithfully verbalize its spatial processing.

**VLAA-Thinker (2025)** found SFT before GRPO degrades reasoning performance on general VLM tasks. Our spatial task shows the opposite — SFT→RL outperforms either alone — suggesting the interaction between SFT and RL depends on the task structure.

**DeepSeek-R1** showed RL produces emergent text reasoning (chain-of-thought, self-verification) without supervision. We tested the same hypothesis for spatial perception and found it partially holds: RL produces emergent scale bar usage (answer-only GRPO) but NOT emergent faithful spatial reasoning (CoT GRPO). The visual domain is harder than the text domain for emergent reasoning.

## Reproducing

```bash
# Full experiment: ~$5 on A100 80GB
git clone https://github.com/WFJKK/Multimodal-GRPO.git
cd Multimodal-GRPO
bash setup.sh                    # Generate shortcut-proof dataset
python3 train_sft.py             # SFT baseline (~30 min)
python3 train_grpo_custom.py     # GRPO answer-only (~3 hrs)
python3 train_grpo_from_sft.py   # SFT→RL pipeline (~2.5 hrs)
python3 train_grpo_cot.py        # CoT GRPO (~6 hrs for full run)
python3 evaluate.py --compare    # Full comparison table
```

All results are deterministic (seed 42) and the dataset generator verifies no shortcuts exist before producing images.

## Limitations

- **No KL penalty** in GRPO — leads to mode collapse (round numbers in standalone GRPO, repeated values in SFT→RL). Standard fix, not implemented.
- **Synthetic only** — no real-world transfer test. The matched pair methodology transfers but the trained models may not.
- **Single task** — hole diameter measurement. The methodology generalizes but is demonstrated on one task.
- **Compute not matched** — GRPO uses ~10x more compute than SFT per step. Multi-epoch SFT might close the gap.
- **CoT ran at 500/1000 steps** — partial training, though reasoning analysis suggests the strategy was stable by step 300.

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
  url={https://github.com/WFJKK/Multimodal-GRPO}
}
```
