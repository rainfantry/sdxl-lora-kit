#!/bin/bash
# One-shot SDXL LoRA training setup for RunPod (bare-pod bootstrap).
#
# Assumes a FRESH pod with nothing installed beyond the base image. Bootstraps
# tmux, system deps, Python ML stack, Kohya_ss, SDXL base model, and launches
# training under accelerate (so bf16 actually works).
#
# RECOMMENDED USAGE:
#   1. SSH into pod
#   2. Install tmux + start a session (so training survives SSH drops):
#        apt-get update -qq && apt-get install -y -qq tmux
#        tmux new -s train
#   3. INSIDE tmux:
#        curl -fsSL https://raw.githubusercontent.com/rainfantry/sdxl-lora-kit/main/runpod_scripts/runpod_sdxl_lora_setup.sh -o setup.sh
#        chmod +x setup.sh
#        nano setup.sh        # edit CONFIG section
#        ./setup.sh
#   4. Detach from tmux with Ctrl+B then D (training keeps running)
#   5. Reattach later with: tmux attach -t train
#
# Total time: ~10 min setup + ~50 min training = ~60 min end-to-end
# Cost on 3090 community ($0.27/hr): ~$0.25
#
# FIX HISTORY (2026-05-17):
#   - Added tmux install + recommendation
#   - Added accelerate install + `accelerate config default` (prevents interactive hang)
#   - Switched bare `python` -> `accelerate launch` (required for bf16)
#   - Added full kohya runtime deps (imagesize, voluptuous, lycoris-lora, etc.)
#   - Added xformers from PyTorch index (matches torch ABI, avoids NMS error)
#   - Added --min_snr_gamma=5 (proven SDXL convergence improvement)
#   - Photo count now includes .jpeg/.png/.webp + early-exit on empty dataset
#   - Added import verification before launching training
#   - Added GPU + driver verification at startup
#   - REMOVED `pip install -r requirements.txt` from kohya — its pinned torch
#     breaks torchvision ABI on RunPod. Install only python-level deps.
#
# RECOVERY (if torchvision::nms error appears on legacy pods):
#   pip install --force-reinstall torch==2.4.1 torchvision==0.19.1 \
#     --index-url https://download.pytorch.org/whl/cu121
#   pip install --force-reinstall xformers --index-url https://download.pytorch.org/whl/cu121

set -e

# ============================================================
# CONFIG — edit these per training run
# ============================================================
TOKEN="g30rg3wu"                           # your trigger token (rare nonsense word)
NUM_REPEATS=20                             # photos repeated per epoch
EPOCHS=10                                  # total training epochs
RANK=32                                    # LoRA rank (16-64 for faces, 32 is sweet spot)
ALPHA=16                                   # LoRA alpha (usually rank/2)
LEARNING_RATE="1e-4"                       # SDXL LoRA standard
DATASET_GDRIVE_ID=""                       # google drive file id of dataset.zip
OUTPUT_NAME="${TOKEN}_lora_v1"             # final lora filename

# ============================================================
# Phase 0: Sanity — verify GPU + driver present
# ============================================================
echo "[0/7] Verifying GPU + driver..."
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found. Is this a GPU pod? Aborting."
    exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || {
    echo "ERROR: nvidia-smi failed. Driver issue. Aborting."
    exit 1
}

# ============================================================
# Phase 1: System packages (tmux, git, aria2, unzip)
# ============================================================
echo ""
echo "[1/7] Installing system packages (tmux, git, wget, aria2, unzip)..."
apt-get update -qq
apt-get install -y -qq tmux unzip git wget aria2

# Warn if not inside tmux — heavily recommended for crash recovery
if [ -z "$TMUX" ]; then
    echo ""
    echo "  ⚠️  WARNING: You are NOT inside a tmux session."
    echo "      If your SSH drops, training will die mid-run."
    echo "      Strongly recommended: Ctrl+C now, run 'tmux new -s train', then re-run this script."
    echo "      Continuing in 10 seconds — Ctrl+C to abort."
    sleep 10
fi

# ============================================================
# Phase 2: Python download tooling (gdown + bitsandbytes for AdamW8bit)
# ============================================================
echo ""
echo "[2/7] Installing Python download tooling..."
pip install -q gdown bitsandbytes

# ============================================================
# Phase 3: Clone Kohya_ss + submodules
# ============================================================
if [ ! -d "/workspace/kohya_ss" ]; then
    echo ""
    echo "[3/7] Cloning Kohya_ss..."
    cd /workspace
    git clone https://github.com/bmaltais/kohya_ss.git
    cd kohya_ss
    git submodule update --init --recursive
else
    echo ""
    echo "[3/7] Kohya_ss already present, skipping clone."
fi

# ============================================================
# Phase 3.5: Install kohya runtime deps + xformers
# ============================================================
# CRITICAL: install training deps WITHOUT touching torch/torchvision.
# DO NOT run `pip install -r kohya_ss/requirements.txt` — kohya pins older
# torch versions that clash with RunPod's pre-installed torch, breaking
# torchvision's C++ ops (NMS error mid-training).
#
# Install only python-level deps; xformers comes from PyTorch index matching
# the cu121 wheel ABI. List was empirically verified — missing any of these
# causes ModuleNotFoundError 5 min into training.
echo ""
echo "[3.5/7] Installing kohya runtime deps (safe set, no torch touch)..."
pip install -q \
    accelerate transformers diffusers safetensors einops ftfy toml \
    imagesize voluptuous lycoris-lora prodigyopt dadaptation \
    pytorch-lightning tensorboard altair easygui omegaconf rich \
    opencv-python-headless
pip install -q -U xformers --index-url https://download.pytorch.org/whl/cu121

# Configure accelerate with default single-GPU config (non-interactive).
# Without this, accelerate's first invocation triggers an interactive prompt
# that hangs forever in non-TTY contexts (and is annoying in interactive ones).
accelerate config default

# Verify imports BEFORE launching training — fail fast at setup time, not
# 5 minutes into training when avr_loss isn't even printing yet.
echo ""
echo "[3.6/7] Verifying kohya imports..."
cd /workspace/kohya_ss/sd-scripts
python -c "from library import sdxl_train_util; print('  [OK] kohya imports verified')" || {
    echo "ERROR: kohya import failed. Missing deps. Recovery hint:"
    echo "  pip install -r /workspace/kohya_ss/requirements.txt"
    echo "  (note: may break torchvision — see RECOVERY block at top of script)"
    exit 1
}

# ============================================================
# Phase 4: Download SDXL Base 1.0 (only if missing)
# ============================================================
mkdir -p /workspace/models
if [ ! -f "/workspace/models/sd_xl_base_1.0.safetensors" ]; then
    echo ""
    echo "[4/7] Downloading SDXL Base 1.0 (6.9 GB, ~30 sec on RunPod)..."
    aria2c -x 16 -s 16 --console-log-level=warn \
        -d /workspace/models -o sd_xl_base_1.0.safetensors \
        "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors"
else
    echo ""
    echo "[4/7] SDXL Base already present, skipping download."
fi

# ============================================================
# Phase 5: Pull dataset from Google Drive + restructure
# ============================================================
echo ""
echo "[5/7] Pulling dataset from Google Drive..."
cd /workspace
if [ -z "$DATASET_GDRIVE_ID" ]; then
    echo "ERROR: DATASET_GDRIVE_ID is empty. Edit CONFIG section + retry."
    exit 1
fi
if [ ! -f "dataset.zip" ]; then
    gdown "$DATASET_GDRIVE_ID" -O dataset.zip
fi

rm -rf /workspace/training_data
mkdir -p "/workspace/training_data/${NUM_REPEATS}_${TOKEN}"
unzip -q -j dataset.zip -d "/workspace/training_data/${NUM_REPEATS}_${TOKEN}/"
mkdir -p /workspace/output

shopt -s nullglob
PHOTO_COUNT=$(ls /workspace/training_data/${NUM_REPEATS}_${TOKEN}/*.{jpg,jpeg,png,webp,JPG,JPEG,PNG,WEBP} 2>/dev/null | wc -l)
CAPTION_COUNT=$(ls /workspace/training_data/${NUM_REPEATS}_${TOKEN}/*.txt 2>/dev/null | wc -l)
echo "    Dataset ready: ${PHOTO_COUNT} photos, ${CAPTION_COUNT} captions."
if [ "$PHOTO_COUNT" -eq 0 ]; then
    echo "ERROR: no photos found in dataset.zip — check archive structure."
    exit 1
fi
if [ "$PHOTO_COUNT" -ne "$CAPTION_COUNT" ]; then
    echo "WARN: photo/caption count mismatch (uncaptioned photos will use folder name token)."
fi

# ============================================================
# Phase 6: Pre-flight summary
# ============================================================
echo ""
echo "[6/7] Pre-flight summary:"
echo "  Token:           $TOKEN"
echo "  Repeats × Epochs: $NUM_REPEATS × $EPOCHS"
echo "  Photos:          $PHOTO_COUNT"
echo "  Rank/Alpha:      $RANK / $ALPHA"
echo "  Learning rate:   $LEARNING_RATE"
echo "  Output:          /workspace/output/${OUTPUT_NAME}.safetensors"
echo ""
echo "      Training begins in 5 seconds. Ctrl+C to abort."
sleep 5

# ============================================================
# Phase 7: Train
# ============================================================
echo ""
echo "[7/7] Starting training (~50 min on RTX 3090)..."
echo "      Watch avr_loss — should be ~0.05-0.15, NOT 'nan'."
echo "      If you see nan: Ctrl+C and check config."
echo ""

cd /workspace/kohya_ss/sd-scripts
# accelerate launch (NOT bare python) — required for bf16 mixed precision to work.
# Bare python silently ignores --mixed_precision="bf16" and trains in fp32 instead.
accelerate launch --num_cpu_threads_per_process=2 sdxl_train_network.py \
    --pretrained_model_name_or_path="/workspace/models/sd_xl_base_1.0.safetensors" \
    --train_data_dir="/workspace/training_data" \
    --output_dir="/workspace/output" \
    --output_name="${OUTPUT_NAME}" \
    --save_model_as="safetensors" \
    --caption_extension=".txt" \
    --resolution=1024 \
    --train_batch_size=1 \
    --max_train_epochs=${EPOCHS} \
    --learning_rate=${LEARNING_RATE} \
    --unet_lr=${LEARNING_RATE} \
    --lr_scheduler="cosine" \
    --optimizer_type="AdamW8bit" \
    --network_module=networks.lora \
    --network_dim=${RANK} \
    --network_alpha=${ALPHA} \
    --network_train_unet_only \
    --mixed_precision="bf16" \
    --no_half_vae \
    --max_grad_norm=1.0 \
    --cache_latents \
    --cache_text_encoder_outputs \
    --gradient_checkpointing \
    --sdpa \
    --save_every_n_epochs=2 \
    --min_snr_gamma=5 \
    --seed=42

echo ""
echo "============================================"
echo "DONE. LoRA saved to:"
echo "  /workspace/output/${OUTPUT_NAME}.safetensors"
echo ""
echo "Intermediate epoch checkpoints (test each):"
ls -lh /workspace/output/*.safetensors
echo ""
echo "DOWNLOAD NOW before killing the pod."
echo "  (e.g. scp -P <port> -i ~/.ssh/id_ed25519 root@<ip>:/workspace/output/*.safetensors ./)"
echo "============================================"
