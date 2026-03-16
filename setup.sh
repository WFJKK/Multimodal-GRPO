#!/bin/bash
set -e

echo "=== Multimodal GRPO Setup ==="
echo "Target: A100 80GB (compute capability 8.0)"
echo ""

# ---- Check GPU ----
if command -v nvidia-smi &>/dev/null; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
    
    COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
    if [[ "$COMPUTE_CAP" == "8.0" ]]; then
        echo "✓ Compute capability 8.0 (A100) — compatible with FineGrainedFP8Config"
    elif [[ "$COMPUTE_CAP" == "8.9" || "$COMPUTE_CAP" == "10.0" || "$COMPUTE_CAP" == "12.0" ]]; then
        echo "✗ Compute capability $COMPUTE_CAP — FineGrainedFP8Config may not auto-dequantize correctly"
        echo "  Use A100 80GB instead."
        exit 1
    else
        echo "⚠ Compute capability $COMPUTE_CAP — untested, proceed with caution"
    fi
else
    echo "⚠ No GPU detected. Dataset generation will work, training will not."
fi
echo ""

# ---- Check disk space ----
AVAIL_GB=$(df -BG /home | tail -1 | awk '{print $4}' | tr -d 'G')
echo "Available disk: ${AVAIL_GB}GB"
if [[ "$AVAIL_GB" -lt 30 ]]; then
    echo "⚠ Less than 30GB free. Model weights (~16GB) + dataset + checkpoints need ~40GB."
    echo "  Consider freeing space or mounting additional storage."
fi
echo ""

# ---- Install dependencies ----
echo "Installing Python dependencies..."
pip install --break-system-packages -q \
    torch torchvision \
    transformers>=4.45.0 \
    trl>=0.12.0 \
    peft>=0.13.0 \
    vllm>=0.6.0 \
    datasets \
    accelerate \
    bitsandbytes \
    matplotlib \
    numpy \
    Pillow

echo ""

# ---- Generate dataset ----
echo "Generating dataset (seed=42, deterministic)..."
python3 generate_dataset.py \
    --n-train 1000 \
    --n-test 200 \
    --seed 42 \
    --output-dir dataset

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. python3 evaluate.py --baseline   # Baseline (no training)"
echo "  2. python3 train_grpo.py            # GRPO training"
echo "  3. python3 evaluate.py              # Post-training evaluation"
