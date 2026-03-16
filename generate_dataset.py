"""Shortcut-proof single-hole dataset generator.

Design principles:
  - The ONLY way to get the right answer is to visually relate the hole's 
    pixel size to the scale bar's pixel size, then use the scale bar's mm label.
  - diameter_mm = hole_pixels * (scale_bar_mm / scale_bar_pixels)

Anti-shortcut measures:
  1. Continuous uniform diameters (3-30mm) — no nominal clusters
  2. Variable zoom (pixels_per_mm) independent of diameter
  3. Many scale bar values, independent of diameter and zoom
  4. No text leaks (no plate dimensions, no dimension labels)
  5. Single hole per image
  6. Variable canvas sizes
  7. Variable hole position (not always centered)
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


SCALE_BAR_VALUES = [5, 10, 15, 20, 25, 30, 40, 50]
DIAMETER_RANGE = (3.0, 30.0)
CANVAS_W_RANGE = (500, 1000)
CANVAS_H_RANGE = (400, 800)
DPI = 150

# Plate appearance: a rectangle around the hole (cosmetic, not informative)
PLATE_COLOR = "#F0F0EA"
PLATE_EDGE_COLOR = "black"
HOLE_EDGE_COLOR = "black"
BG_COLOR = "white"


def compute_ppm_range(diameter_mm, scale_bar_mm, canvas_w, canvas_h):
    """Compute feasible pixels-per-mm range so hole and scale bar fit."""
    min_hole_px = 15  # hole must be at least this many pixels across
    min_sb_px = 30    # scale bar must be at least this many pixels
    max_feature_frac = 0.55  # largest feature can't exceed this fraction of canvas

    ppm_min = max(min_hole_px / diameter_mm, min_sb_px / scale_bar_mm)
    
    # Both hole and scale bar must fit horizontally
    max_feature_mm = max(diameter_mm, scale_bar_mm)
    ppm_max = max_feature_frac * canvas_w / max_feature_mm
    
    # Hole must fit vertically too (with room for scale bar below)
    ppm_max = min(ppm_max, 0.5 * canvas_h / diameter_mm)
    
    return ppm_min, ppm_max


def generate_sample(rng, idx):
    """Generate one sample: image metadata + ground truth."""
    
    # Sample physical parameters (all independent)
    diameter_mm = rng.uniform(*DIAMETER_RANGE)
    scale_bar_mm = rng.choice(SCALE_BAR_VALUES)
    canvas_w = rng.integers(*CANVAS_W_RANGE)
    canvas_h = rng.integers(*CANVAS_H_RANGE)
    
    # Compute feasible zoom range and sample
    ppm_min, ppm_max = compute_ppm_range(diameter_mm, scale_bar_mm, canvas_w, canvas_h)
    
    if ppm_min >= ppm_max:
        # Rare edge case: widen canvas
        canvas_w = int(canvas_w * 1.5)
        canvas_h = int(canvas_h * 1.5)
        ppm_min, ppm_max = compute_ppm_range(diameter_mm, scale_bar_mm, canvas_w, canvas_h)
    
    ppm = rng.uniform(ppm_min, ppm_max)
    
    # Derived pixel sizes
    hole_px = diameter_mm * ppm
    sb_px = scale_bar_mm * ppm
    
    # Random plate size (cosmetic — just a rectangle surrounding the hole)
    # Must be bigger than the hole, but otherwise uninformative
    plate_w_min = max(hole_px * 1.8, 100)
    plate_w_max = max(plate_w_min + 50, canvas_w * 0.75)
    plate_w_px = rng.uniform(plate_w_min, plate_w_max)
    
    plate_h_min = max(hole_px * 1.8, 80)
    plate_h_max = max(plate_h_min + 50, canvas_h * 0.65)
    plate_h_px = rng.uniform(plate_h_min, plate_h_max)
    
    # Plate position: leave room for scale bar below
    plate_x_max = max(21, canvas_w - plate_w_px - 20)
    plate_x = rng.uniform(20, plate_x_max)
    plate_y_max = max(61, canvas_h - plate_h_px - 20)
    plate_y = rng.uniform(60, plate_y_max)
    
    # Hole position: random within plate, but must fit
    margin = hole_px / 2 + 10
    hx_min = plate_x + margin
    hx_max = max(hx_min + 1, plate_x + plate_w_px - margin)
    hole_cx = rng.uniform(hx_min, hx_max)
    
    hy_min = plate_y + margin
    hy_max = max(hy_min + 1, plate_y + plate_h_px - margin)
    hole_cy = rng.uniform(hy_min, hy_max)
    
    # Scale bar position: below the plate
    sb_x = plate_x
    sb_y = plate_y - 35  # below plate with gap
    
    return {
        "idx": idx,
        "diameter_mm": round(float(diameter_mm), 2),
        "scale_bar_mm": int(scale_bar_mm),
        "canvas_w": int(canvas_w),
        "canvas_h": int(canvas_h),
        "ppm": round(float(ppm), 4),
        "hole_px": round(float(hole_px), 1),
        "sb_px": round(float(sb_px), 1),
        "plate_x": float(plate_x),
        "plate_y": float(plate_y),
        "plate_w_px": float(plate_w_px),
        "plate_h_px": float(plate_h_px),
        "hole_cx": float(hole_cx),
        "hole_cy": float(hole_cy),
        "sb_x": float(sb_x),
        "sb_y": float(sb_y),
    }


def render_image(sample, output_path):
    """Render a single-hole technical drawing."""
    cw, ch = sample["canvas_w"], sample["canvas_h"]
    fig_w = cw / DPI
    fig_h = ch / DPI
    
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=DPI)
    ax.set_xlim(0, cw)
    ax.set_ylim(0, ch)
    ax.set_aspect("equal")
    ax.axis("off")
    
    # Draw plate rectangle
    rect = patches.Rectangle(
        (sample["plate_x"], sample["plate_y"]),
        sample["plate_w_px"], sample["plate_h_px"],
        linewidth=1.5, edgecolor=PLATE_EDGE_COLOR,
        facecolor=PLATE_COLOR, zorder=1
    )
    ax.add_patch(rect)
    
    # Draw hole
    hole_r = sample["hole_px"] / 2
    circle = patches.Circle(
        (sample["hole_cx"], sample["hole_cy"]),
        hole_r, linewidth=1.2,
        edgecolor=HOLE_EDGE_COLOR, facecolor="white", zorder=2
    )
    ax.add_patch(circle)
    
    # Crosshairs
    cx, cy = sample["hole_cx"], sample["hole_cy"]
    cross_len = hole_r * 1.3
    ax.plot([cx - cross_len, cx + cross_len], [cy, cy],
            color="#999999", linewidth=0.7, zorder=3)
    ax.plot([cx, cx], [cy - cross_len, cy + cross_len],
            color="#999999", linewidth=0.7, zorder=3)
    
    # Hole label
    ax.text(cx, cy + hole_r + 8, "H1",
            ha="center", va="bottom", fontsize=7,
            fontweight="bold", zorder=4)
    
    # Scale bar
    sb_x = sample["sb_x"]
    sb_y = sample["sb_y"]
    sb_len = sample["sb_px"]
    
    # Main bar
    ax.plot([sb_x, sb_x + sb_len], [sb_y, sb_y],
            color="black", linewidth=2, zorder=5)
    # End ticks
    tick_h = 5
    ax.plot([sb_x, sb_x], [sb_y - tick_h, sb_y + tick_h],
            color="black", linewidth=1.5, zorder=5)
    ax.plot([sb_x + sb_len, sb_x + sb_len], [sb_y - tick_h, sb_y + tick_h],
            color="black", linewidth=1.5, zorder=5)
    # Label
    ax.text(sb_x + sb_len / 2, sb_y - 10,
            f"{sample['scale_bar_mm']} mm",
            ha="center", va="top", fontsize=7, zorder=5)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)


def verify_no_shortcuts(samples):
    """Verify that no single feature predicts diameter_mm."""
    diams = np.array([s["diameter_mm"] for s in samples])
    hole_pxs = np.array([s["hole_px"] for s in samples])
    sb_mms = np.array([s["scale_bar_mm"] for s in samples])
    sb_pxs = np.array([s["sb_px"] for s in samples])
    canvas_ws = np.array([s["canvas_w"] for s in samples])
    ppms = np.array([s["ppm"] for s in samples])
    
    # The correct computation
    correct = hole_pxs * sb_mms / sb_pxs
    
    print("=== SHORTCUT VERIFICATION ===")
    print(f"N = {len(samples)}")
    print(f"Diameter range: {diams.min():.1f} - {diams.max():.1f} mm")
    print(f"Diameter mean: {diams.mean():.1f} mm")
    print()
    
    # Correlations (should all be ~0 except the correct formula)
    features = {
        "hole_pixels": hole_pxs,
        "scale_bar_mm": sb_mms,
        "scale_bar_pixels": sb_pxs,
        "canvas_width": canvas_ws,
        "pixels_per_mm": ppms,
        "hole_px / canvas_w": hole_pxs / canvas_ws,
        "CORRECT (hole_px * sb_mm / sb_px)": correct,
    }
    
    print("Correlation with diameter_mm:")
    for name, feat in features.items():
        r = np.corrcoef(feat, diams)[0, 1]
        print(f"  {name:45s} r = {r:+.3f}")
    
    print()
    
    # Baseline MAEs
    print("Baseline MAEs (no vision needed):")
    mean_mae = np.mean(np.abs(diams - diams.mean()))
    print(f"  Always guess mean ({diams.mean():.1f}mm):     {mean_mae:.2f} mm")
    
    median_mae = np.mean(np.abs(diams - np.median(diams)))
    print(f"  Always guess median ({np.median(diams):.1f}mm):   {median_mae:.2f} mm")
    
    # Best linear predictor from pixel diameter alone
    from numpy.polynomial import polynomial as P
    c = P.polyfit(hole_pxs, diams, 1)
    pred_lr = P.polyval(hole_pxs, c)
    lr_mae = np.mean(np.abs(diams - pred_lr))
    print(f"  Linear regress on pixel diameter:  {lr_mae:.2f} mm")
    
    # Correct formula
    correct_mae = np.mean(np.abs(diams - correct))
    print(f"  Correct formula (scale bar):       {correct_mae:.4f} mm")
    
    print()
    print("If GRPO gets MAE < linear regression baseline, it learned something.")
    print("If GRPO gets MAE << linear regression baseline, it likely uses the scale bar.")


def generate_matched_pairs(rng, n_pairs=50):
    """Generate matched pairs: identical pixel hole, different scale bars.
    
    Each pair has:
      - Same hole pixel diameter, same canvas size, same hole position
      - Different scale bar (different mm value AND different pixel length)
      - Therefore different ground truth diameter_mm
    
    This is the definitive test: if the model gives the same answer
    for both images in a pair, it's ignoring the scale bar.
    """
    pairs = []
    for i in range(n_pairs):
        # Pick two different scale bar values
        sb_pair = rng.choice(SCALE_BAR_VALUES, size=2, replace=False)
        sb_a, sb_b = int(sb_pair[0]), int(sb_pair[1])
        
        # Fixed canvas
        canvas_w = int(rng.integers(600, 900))
        canvas_h = int(rng.integers(500, 700))
        
        # Fixed pixel diameter for the hole (what the model sees)
        hole_px = float(rng.uniform(30, 200))
        
        # Two different zoom levels such that the hole pixel size is the same
        # but the scale bar pixel sizes differ, reflecting different physical scales
        # For variant A: pick a ppm, derive diameter_a
        # For variant B: pick a different ppm, derive diameter_b
        # Both must produce the same hole_px
        
        # ppm_a and ppm_b chosen so that scale bars are visually distinct
        # and both fit on canvas
        ppm_a = float(rng.uniform(3.0, 15.0))
        ppm_b = float(rng.uniform(3.0, 15.0))
        
        # Ensure they're meaningfully different
        while abs(ppm_a - ppm_b) < 1.5:
            ppm_b = float(rng.uniform(3.0, 15.0))
        
        # Derived physical diameters (what the correct answer should be)
        diam_a = hole_px / ppm_a
        diam_b = hole_px / ppm_b
        
        # Scale bar pixel lengths
        sb_px_a = sb_a * ppm_a
        sb_px_b = sb_b * ppm_b
        
        # Check everything fits
        max_feature = max(hole_px, sb_px_a, sb_px_b)
        if max_feature > canvas_w * 0.6:
            continue
        if hole_px > canvas_h * 0.4:
            continue
        
        # Build the two sample dicts with identical hole appearance
        plate_w = max(hole_px * 2, 150)
        plate_h = max(hole_px * 2, 120)
        plate_x = (canvas_w - plate_w) / 2
        plate_y = 70.0
        hole_cx = canvas_w / 2
        hole_cy = plate_y + plate_h / 2
        
        base = {
            "canvas_w": canvas_w,
            "canvas_h": canvas_h,
            "hole_px": round(hole_px, 1),
            "plate_x": plate_x,
            "plate_y": plate_y,
            "plate_w_px": plate_w,
            "plate_h_px": plate_h,
            "hole_cx": hole_cx,
            "hole_cy": hole_cy,
            "sb_x": plate_x,
            "sb_y": plate_y - 35,
        }
        
        var_a = {**base, "idx": i,
                 "diameter_mm": round(diam_a, 2),
                 "scale_bar_mm": sb_a,
                 "ppm": round(ppm_a, 4),
                 "sb_px": round(sb_px_a, 1)}
        
        var_b = {**base, "idx": i,
                 "diameter_mm": round(diam_b, 2),
                 "scale_bar_mm": sb_b,
                 "ppm": round(ppm_b, 4),
                 "sb_px": round(sb_px_b, 1)}
        
        pairs.append({
            "pair_id": i,
            "hole_px": round(hole_px, 1),
            "a": var_a,
            "b": var_b,
            "diam_a": round(diam_a, 2),
            "diam_b": round(diam_b, 2),
            "sb_a": sb_a,
            "sb_b": sb_b,
        })
    
    return pairs


def verify_matched_pairs(pairs):
    """Print matched pair diagnostics."""
    print("\n=== MATCHED PAIR DIAGNOSTIC ===")
    print(f"N pairs: {len(pairs)}")
    diffs = [abs(p["diam_a"] - p["diam_b"]) for p in pairs]
    print(f"Ground truth difference per pair: {np.mean(diffs):.1f} mm avg "
          f"(range {np.min(diffs):.1f} - {np.max(diffs):.1f})")
    print(f"Same hole pixels, different scale bars -> different correct answers")
    print(f"If model gives same answer for both: NOT using scale bar")
    print(f"If model answers diverge correctly: USING scale bar")
    print()
    for p in pairs[:5]:
        print(f"  Pair {p['pair_id']}: hole={p['hole_px']}px, "
              f"SB {p['sb_a']}mm->{p['diam_a']}mm vs "
              f"SB {p['sb_b']}mm->{p['diam_b']}mm "
              f"(diff={abs(p['diam_a']-p['diam_b']):.1f}mm)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="dataset")
    parser.add_argument("--verify-only", action="store_true",
                        help="Generate metadata and verify, don't render images")
    args = parser.parse_args()
    
    rng = np.random.default_rng(args.seed)
    
    # Generate samples
    train_samples = [generate_sample(rng, i) for i in range(args.n_train)]
    test_samples = [generate_sample(rng, i) for i in range(args.n_test)]
    
    # Verify
    print("=== TRAIN SET ===")
    verify_no_shortcuts(train_samples)
    print()
    print("=== TEST SET ===")
    verify_no_shortcuts(test_samples)
    
    if args.verify_only:
        return
    
    # Save metadata
    out = Path(args.output_dir)
    (out / "train").mkdir(parents=True, exist_ok=True)
    (out / "test").mkdir(parents=True, exist_ok=True)
    (out / "test_matched").mkdir(parents=True, exist_ok=True)
    
    with open(out / "train" / "metadata.jsonl", "w") as f:
        for s in train_samples:
            f.write(json.dumps(s) + "\n")
    
    with open(out / "test" / "metadata.jsonl", "w") as f:
        for s in test_samples:
            f.write(json.dumps(s) + "\n")
    
    # Render images
    for split, samples in [("train", train_samples), ("test", test_samples)]:
        for i, s in enumerate(samples):
            path = out / split / f"image_{s['idx']:04d}.png"
            render_image(s, str(path))
            if (i + 1) % 100 == 0:
                print(f"  {split}: {i+1}/{len(samples)}")
    
    # === MATCHED PAIR DIAGNOSTIC ===
    # Generate pairs where the hole looks IDENTICAL in pixels but the
    # scale bar differs, so ground truth diameter differs.
    # If model gives same answer for both -> not using scale bar.
    # If model gives different (correct) answers -> using scale bar.
    print("\nGenerating matched pair diagnostic set...")
    matched_pairs = generate_matched_pairs(rng, n_pairs=50)
    
    with open(out / "test_matched" / "metadata.jsonl", "w") as f:
        for pair in matched_pairs:
            f.write(json.dumps(pair) + "\n")
    
    for pair in matched_pairs:
        for variant in ["a", "b"]:
            s = pair[variant]
            path = out / "test_matched" / f"pair_{pair['pair_id']:03d}_{variant}.png"
            render_image(s, str(path))
    
    print(f"  Generated {len(matched_pairs)} matched pairs")
    verify_matched_pairs(matched_pairs)
    
    print(f"\nDone. Output in {out}/")
    print(f"  train/: {len(train_samples)} images")
    print(f"  test/: {len(test_samples)} images")
    print(f"  test_matched/: {len(matched_pairs)} pairs ({len(matched_pairs)*2} images)")


if __name__ == "__main__":
    main()
