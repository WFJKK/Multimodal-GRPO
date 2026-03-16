#!/bin/bash
set -e

# Session 3: CoT GRPO training + evaluation
# SFT and answer-only GRPO already completed in previous sessions.
# Budget: 6 hours on A100 80GB
#
# Run inside tmux:
#   bash run_experiments.sh

echo "=== Experiment Session 3: CoT GRPO ==="
echo "Start time: $(date)"
echo ""

# ---- Deps that may be missing on fresh instance ----
pip install --break-system-packages -q qwen-vl-utils 2>/dev/null || true

# ---- Generate dataset if needed ----
if [ ! -d "dataset/train" ]; then
    echo "Generating dataset..."
    python3 generate_dataset.py --n-train 1000 --n-test 200 --seed 42 --output-dir dataset
fi

# ---- CoT GRPO Training ----
echo "=========================================="
echo "  CoT GRPO Training"
echo "=========================================="
python3 train_grpo_cot.py --resume
echo ""

# ---- Evaluate whatever checkpoint exists ----
echo "=========================================="
echo "  CoT GRPO Evaluation"
echo "=========================================="
if [ -d "checkpoints_cot/final" ]; then
    python3 evaluate.py --checkpoint-dir checkpoints_cot/final --tag grpo_cot
else
    # Find latest checkpoint
    LATEST=$(ls -d checkpoints_cot/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
    if [ -n "$LATEST" ]; then
        echo "No final checkpoint, using $LATEST"
        python3 evaluate.py --checkpoint-dir "$LATEST" --tag grpo_cot_partial
    else
        echo "No checkpoints found!"
    fi
fi
echo ""

# ---- Comparison ----
echo "=========================================="
echo "  Final Comparison"
echo "=========================================="
python3 evaluate.py --compare
echo ""

# ---- Push results ----
git add results/
git add -f checkpoints_cot/training_log.jsonl 2>/dev/null || true
git commit -m "CoT GRPO results" 2>/dev/null || true
git push 2>/dev/null || true

echo ""
echo "=== Complete ==="
echo "End time: $(date)"
