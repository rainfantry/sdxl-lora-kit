#!/bin/bash
# One-shot SDXL LoRA training setup for RunPod.
#
# Usage on the pod:
#   1. wget https://your-cdn/runpod_sdxl_lora_setup.sh
#      (or scp it from laptop, or paste contents into nano)
#   2. chmod +x runpod_sdxl_lora_setup.sh
#   3. Edit the CONFIG section below (trigger token + gdrive id)
#   4. ./runpod_sdxl_lora_setup.sh
#
# Total time: ~5 min setup + ~50 min training = ~55 min end-to-end
# Cost on 3090 community ($0.27/hr): ~$0.25

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
# Phase 1: System deps
# ============================================================
echo "[1/6] Installing system deps..."
apt-get update -qq
apt-get install -y -qq unzip git wget aria2

# ============================================================
# Phase 2: Python deps
# ============================================================
echo "[2/6] Installing Python deps..."
pip install -q gdown bitsandbytes

# ============================================================
# Phase 3: Clone Kohya_ss + submodules
# ============================================================
if [ ! -d "/workspace/kohya_ss" ]; then
    echo "[3/6] Cloning Kohya_ss..."
    cd /workspace
    git clone https://github.com/bmaltais/kohya_ss.git
    cd kohya_ss
    git submodule update --init --recursive
else
    echo "[3/6] Kohya_ss already present, skipping clone."
fi

# ============================================================
# Phase 4: Download SDXL Base 1.0 (only if missing)
# ============================================================
mkdir -p /workspace/models
if [ ! -f "/workspace/models/sd_xl_base_1.0.safetensors" ]; then
    echo "[4/6] Downloading SDXL Base 1.0 (6.9 GB, ~30 sec on RunPod)..."
    aria2c -x 16 -s 16 --console-log-level=warn \
        -d /workspace/models -o sd_xl_base_1.0.safetensors \
        "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors"
else
    echo "[4/6] SDXL Base already present, skipping download."
fi

# ============================================================
# Phase 5: Pull dataset from Google Drive + restructure
# ============================================================
echo "[5/6] Pulling dataset from Google Drive..."
cd /workspace
if [ ! -f "dataset.zip" ]; then
    gdown "$DATASET_GDRIVE_ID" -O dataset.zip
fi

rm -rf /workspace/training_data
mkdir -p "/workspace/training_data/${NUM_REPEATS}_${TOKEN}"
unzip -q -j dataset.zip -d "/workspace/training_data/${NUM_REPEATS}_${TOKEN}/"
mkdir -p /workspace/output

PHOTO_COUNT=$(ls /workspace/training_data/${NUM_REPEATS}_${TOKEN}/*.jpg 2>/dev/null | wc -l)
CAPTION_COUNT=$(ls /workspace/training_data/${NUM_REPEATS}_${TOKEN}/*.txt 2>/dev/null | wc -l)
echo "    Dataset ready: ${PHOTO_COUNT} photos, ${CAPTION_COUNT} captions."

# ============================================================
# Phase 6: Train
# ============================================================
echo "[6/6] Starting training (~50 min on RTX 3090)..."
echo "      Watch avr_loss — should be ~0.05-0.15, NOT 'nan'."
echo "      If you see nan: Ctrl+C and check config."
echo ""

cd /workspace/kohya_ss/sd-scripts
python sdxl_train_network.py \
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
echo "============================================"
