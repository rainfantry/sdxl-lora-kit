# SDXL Face LoRA — Complete Pipeline

End-to-end documentation for training a face LoRA on RunPod and deploying it locally via ComfyUI. Battle-tested through one full training run with all the gotchas resolved.

**Total time investment:** ~3 hours of attention across ~6 hours wall time.
**Total cost:** ~$0.50-1.00 on RunPod community 3090.
**Output:** A 325 MB `.safetensors` file that injects a specific person's identity into any SDXL-family generation, forever, for free.

---

## Table of Contents

- [TL;DR — what this kit produces](#tldr)
- [Prerequisites](#prerequisites)
- [Phase 1 — Photo prep (local laptop)](#phase-1--photo-prep-local-laptop)
- [Phase 2 — RunPod training](#phase-2--runpod-training)
- [Phase 3 — Validation testing (still on RunPod)](#phase-3--validation-testing-still-on-runpod)
- [Phase 4 — Local ComfyUI deployment](#phase-4--local-comfyui-deployment)
- [Settings cheatsheet (proven values)](#settings-cheatsheet-proven-values)
- [Troubleshooting (errors actually hit + fixes)](#troubleshooting-errors-actually-hit--fixes)
- [Files in this kit](#files-in-this-kit)
- [Reusability for the next LoRA](#reusability-for-the-next-lora)

---

## TL;DR

This pipeline takes 20-30 photos of one person, fine-tunes SDXL into recognising that person via a portable LoRA file, and lets you generate unlimited images of that person in any scene/style/checkpoint forever — for the cost of ~$0.30 of one-time GPU rental.

**The deliverable:** `sks_woman_lora_v3.safetensors` (325 MB) that:
- Works on any SDXL-family checkpoint (Base, Juggernaut, RealVisXL, etc.)
- Is portable — drop into any ComfyUI install, swap in/out via dropdown
- Stays private — gitignored, never committed, never uploaded
- Costs nothing to use after training (local inference is free)

---

## Prerequisites

**Hardware:**
- Windows laptop with any NVIDIA GPU (4 GB minimum for inference; training uses cloud)
- NVIDIA Studio Driver (latest) — required for CUDA 13+ support
- RTX 2050 specifically: works but slow (~60 sec/image at 1024×1024 with `--lowvram`)

**Cloud:**
- RunPod account with $1-5 in credit
- Recommend: RTX 3090 community pod ($0.22-0.34/hr)

**Software:**
- ComfyUI Desktop installed at `C:\Users\<USER>\Desktop\ComfyUi\`
  - User data folder: `C:\Users\<USER>\Desktop\ui\`
- Google Drive account (for dataset transfer to pod)
- `gh` CLI authenticated (optional — for repo commits)

---

## Phase 1 — Photo prep (local laptop)

### 1.1 Gather photos

**Target: 20-30 photos** in `C:\Users\<USER>\Desktop\kohya\lora_dataset_raw\`

**Hard rules:**
- ❌ No sunglasses, hats covering forehead, masks
- ❌ No motion blur, out-of-focus shots
- ❌ No filters / heavy editing (model learns the filter, not the face)
- ❌ No group shots where multiple faces show
- ✓ Min resolution 1024×1024 (more is fine, will downscale)

**Required mix:**
- 8-10 close-ups (head fills frame, varied expressions)
- 8-10 medium shots (head + shoulders, varied angles: front, 3/4, side)
- 4-6 wider shots (half-body, varied outfits)
- Mix lighting (indoor warm, outdoor daylight, side-lit)
- Mix backgrounds (avoid all-same-room → model bakes that into identity)

**Diversity > quantity.** 22 varied photos beat 50 similar ones.

### 1.2 Crop + caption

**Crop to 1024×1024 square** using BIRME (free, web): https://www.birme.net/?target_width=1024&target_height=1024

**Caption files:** one `.txt` per photo with same filename. `photo_01.jpg` → `photo_01.txt`

**Caption structure (in order):**
1. **Trigger token** (rare word, e.g. `sks_woman`)
2. **Class word** (`woman` or `man`) — anchors trigger to a category SDXL already knows
3. **Pose / expression** (looking at camera, smiling)
4. **Clothing** (black t-shirt, white tank top)
5. **Hair STATE** — only if varies between photos
6. **Setting** (indoor, outdoor, white background)
7. **Lighting** (soft window light, warm tungsten)

**Example caption (`photo_01.txt`):**
```
sks_woman, woman, looking at camera, slight smile, white tank top, indoor, soft natural light
```

**DO NOT caption these (permanent features → bind to trigger):**
- Race / ethnicity
- Eye colour, hair colour, age
- Face shape, body type

**The rule:** Only caption what VARIES between photos. Constants get absorbed into the trigger token. That's how the model learns "sks_woman = this specific person."

### 1.3 Final prep

Place ready dataset in:
```
C:\Users\<USER>\Desktop\kohya\lora_dataset_ready\
  ├── photo_01.jpg
  ├── photo_01.txt
  ├── photo_02.jpg
  ├── photo_02.txt
  └── ...
```

**Zip + upload to Google Drive:**
```powershell
Compress-Archive -Path C:\Users\<USER>\Desktop\kohya\lora_dataset_ready\* `
                 -DestinationPath C:\Users\<USER>\Desktop\kohya\dataset.zip
```

Upload `dataset.zip` to Drive → Share → Anyone with link.

Copy the **file ID** from the share URL:
```
https://drive.google.com/file/d/1Vrb7Ua1Ep988IwudtptwC0G2WghokHa4/view
                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                this is your GDRIVE_ID
```

---

## Phase 2 — RunPod training

### 2.1 Spin up pod

Console: https://www.runpod.io/console/pods → Deploy

- **GPU:** RTX 3090 (24 GB) — Community Cloud (~$0.27/hr)
- **Template:** RunPod Pytorch 2.4 (or newer)
- **Container disk:** 10 GB
- **Volume:** 50 GB (smaller = quota issues during checkpoint downloads)
- **Ports:** SSH + HTTP defaults

### 2.2 Launch training (paste-block)

Open the pod's web terminal. Paste the master block from [runpod_scripts/MASTER_WORKFLOW.txt](runpod_scripts/MASTER_WORKFLOW.txt) — it's a single executable script that:

1. Installs deps (apt + pip + tmux)
2. Clones Kohya_ss with submodules
3. Downloads SDXL Base 1.0 via aria2c (~30 sec at 100 MB/s)
4. Pulls dataset.zip from your Drive via `gdown`
5. Restructures into Kohya's `<repeats>_<token>/` folder convention
6. Launches training inside **tmux** (survives connection drops)

**Edit two lines at the top before pasting:**
```bash
GDRIVE_ID="1Vrb7Ua1Ep988IwudtptwC0G2WghokHa4"
TRIGGER="sks_woman"
```

Then paste, hit enter, walk away.

### 2.3 Watch / detach

```bash
tmux ls                      # see sessions
tmux attach -t train         # watch live (loss should be 0.05-0.15, NOT 'nan')
# Ctrl+B then D              # detach without killing
```

**Connection drops are safe** — tmux keeps training alive. Re-attach any time.

**Training takes ~50-60 min** on a 3090. Cost ~$0.25.

### 2.4 Training success indicators

When watching with `tmux attach -t train`, look for:
- `avr_loss=0.0X to 0.15` consistently (NOT `nan`)
- Speed `~1.2 it/s` on a 3090
- Checkpoints saved every 2 epochs: `sks_woman_lora_v3-000002.safetensors`, then `-000004`, etc.
- Final: `sks_woman_lora_v3.safetensors` (no number suffix) at epoch 10

---

## Phase 3 — Validation testing (still on RunPod)

Don't kill the pod yet. Test the LoRA to pick the best epoch + tune inference parameters.

### 3.1 A/B test epoch checkpoints

Run [runpod_scripts/INFERENCE_PASTE.txt](runpod_scripts/INFERENCE_PASTE.txt) to generate test images across multiple epochs.

**Why test multiple epochs:** Face LoRAs hit a sweet spot. Too few epochs = weak likeness. Too many = overfit (rigid, won't follow prompts). Save checkpoints every 2 epochs and test each.

Expected pattern (from one real run):
- **Epoch 02-04:** Wrong hair colour, sepia tones — under-trained
- **Epoch 06:** Asian features locking in
- **Epoch 08:** Strong likeness, full colour, natural
- **Epoch 10:** Nearly identical to 08 = healthy plateau

**Winner = whichever epoch has best likeness AND follows prompts.** Often epoch 8-10 for a 10-epoch run.

### 3.2 A/B test checkpoint compatibility

Same LoRA, different SDXL base checkpoints. Same prompts, same seed.

**Results from real testing:**

| Checkpoint | Best for |
|-----------|----------|
| **SDXL Base 1.0** | Neutral baseline, training reference |
| **Juggernaut XL v9** | **Winner — cinematic warm tones, dramatic lighting** |
| **RealVisXL V5** | Sharpest photoreal, professional headshots |

### 3.3 Tuned inference settings (PRODUCTION — locked after 4 rounds of A/B)

After v1 baseline → v2 sampler sweep → v3 dial-in → v4 final showdown:

| Setting | Value | Why |
|---------|-------|-----|
| LoRA scale | **0.9** | Below 0.85 lets blonde-hair drift through |
| CFG | **5.0** | 5.5 was OK but 5.0 paired with SDE samplers held identity better |
| Steps | **35** | 30 worked, 35 adds polish, 50 was no better |
| Sampler | **dpmpp_3m_sde** | Newer SDE variant — sharper than 2m, ID-stable at cfg 5.0 |
| Scheduler | **karras** | Classic photoreal pairing with 3m_sde |
| Seed mode | **randomize** | Production phase — variety. Save "money seeds" of keepers. |
| Aspect ratio | **832×1216** | SDXL-trained portrait aspect; full body fits |
| Avoid | "close-up" in prompt | SDXL hallucinates fuzz at extreme zoom |

**Runner-up settings (save as alt workflow):** dpmpp_2m_sde + exponential / cfg 5.0 / 35 steps — slightly different aesthetic, same identity quality.

### 3.4 Download winners + kill pod

Once you've picked the best epoch + checkpoint, tar everything for download:

```bash
cd /workspace
tar -cvf laptop_bundle.tar \
  output/sks_woman_lora_v3.safetensors \
  output/sks_woman_lora_v3-000008.safetensors \
  models/sd_xl_base_1.0.safetensors \
  models/Juggernaut-XL_v9.safetensors \
  models/RealVisXL_V5.0_fp16.safetensors
```

Download via JupyterLab right-click. ~22 GB tarball.

**THEN: kill the pod** (RunPod console → Pods → Stop → **Terminate**, NOT just Stop). Otherwise storage charges accumulate.

---

## Phase 4 — Local ComfyUI deployment

### 4.1 Pre-flight checks

**NVIDIA driver:**
```powershell
nvidia-smi
# Should show CUDA Version: 13.0+ (if older, install Studio Driver from nvidia.com)
```

**PyTorch CUDA:**
```powershell
& "C:\Users\<USER>\Desktop\ui\.venv\Scripts\python.exe" -c "import torch; print(torch.cuda.is_available())"
# Should print True
```

If False → driver / PyTorch mismatch. Update driver to latest Studio version, restart, retry.

### 4.2 Place files

Unpack `laptop_bundle.tar` and place files in the correct ComfyUI folders:

| File | Goes into |
|------|-----------|
| `sks_woman_lora_v3.safetensors` | `C:\Users\<USER>\Desktop\ui\models\loras\` |
| `sks_woman_lora_v3-000008.safetensors` | same |
| `sd_xl_base_1.0.safetensors` | `C:\Users\<USER>\Desktop\ui\models\checkpoints\` |
| `Juggernaut-XL_v9.safetensors` | same |
| `RealVisXL_V5.0_fp16.safetensors` | same |

PowerShell unpack + move:
```powershell
cd C:\Users\<USER>\Downloads
tar -xvf laptop_bundle.tar
Move-Item output\sks_woman_lora_v3*.safetensors C:\Users\<USER>\Desktop\ui\models\loras\
Move-Item models\*.safetensors C:\Users\<USER>\Desktop\ui\models\checkpoints\
```

### 4.3 Free local VRAM before launching

On a 4 GB card, every MB counts. Close before launching ComfyUI:
- Discord (or disable hw acceleration in settings)
- Chrome / Edge (or close GPU-heavy tabs)
- Any leftover ComfyUI processes from crashes

Verify:
```powershell
nvidia-smi
# Memory-Usage should be <500 MiB before launching ComfyUI
```

### 4.4 Load the prebuilt workflow

1. Launch **ComfyUI Desktop** (the .exe)
2. Wait for canvas
3. **Drag** [runpod_scripts/sks_woman_workflow.json](runpod_scripts/sks_woman_workflow.json) onto the canvas
4. Verify dropdowns show your files (Load Checkpoint = Juggernaut, Load LoRA = sks_woman_lora_v3)
5. If anything's RED, hit the **Refresh** button (toolbar)
6. Click **Queue Prompt** (right panel)

First generation: ~60-90 sec (JIT compile). Subsequent: ~30-60 sec on RTX 2050.

Output appears in the **Save Image** node thumbnail. File at:
```
C:\Users\<USER>\Desktop\ui\output\sks_woman_XXXXX_.png
```

### 4.5 Iterate

To experiment:
- **Click positive CLIP Text Encode node** → edit prompt → Queue Prompt
- **Click KSampler node** → change seed → Queue Prompt (different image)
- **Click Load Checkpoint dropdown** → swap to RealVis or Base → Queue Prompt (different aesthetic)
- **Click Load LoRA strength** → change from 0.9 to 0.95 (stronger identity) → Queue Prompt

### 4.6 Speed boost (optional but recommended for 2050)

Download **SDXL Lightning 8-step LoRA** (394 MB):
```
https://huggingface.co/ByteDance/SDXL-Lightning/resolve/main/sdxl_lightning_8step_lora.safetensors
```

Drop into `C:\Users\<USER>\Desktop\ui\models\loras\`.

Add a second `Load LoRA` node in your workflow, chain it AFTER the face LoRA, strength 1.0.

**Required setting changes when Lightning is active:**
- Steps: 30 → **8**
- CFG: 5.5 → **2.0**
- Sampler: dpmpp_2m → **euler**
- Scheduler: karras → **sgm_uniform**

Result: ~3× faster (15-25 sec per image instead of 60-90).

---

## Settings cheatsheet (proven values)

**Training (Kohya_ss on RunPod 3090):**
```
mixed_precision = bf16              # NOT fp16 — causes NaN
no_half_vae = true                  # VAE in fp32 — also prevents NaN
max_grad_norm = 1.0                 # gradient clipping safety
network_dim = 32                    # LoRA rank
network_alpha = 16                  # LoRA alpha (rank / 2)
learning_rate = 1e-4
unet_lr = 1e-4
network_train_unet_only = true      # for face LoRAs (faster + cleaner)
optimizer_type = AdamW8bit          # saves ~4 GB VRAM
gradient_checkpointing = true       # fits 24 GB cards
cache_latents = true                # speedup
cache_text_encoder_outputs = true   # speedup (requires network_train_unet_only)
sdpa = true                         # fast attention
max_train_epochs = 10
batch_size = 1
save_every_n_epochs = 2
seed = 42
```

**Inference (ComfyUI / diffusers):**

Standard (high quality):
```
LoRA scale = 0.9
CFG = 5.5
Steps = 30
Sampler = dpmpp_2m
Scheduler = karras
Aspect = 832×1216 (portrait) or 1024×1024 (square)
Seed = pick anything; fix it for A/B testing
```

Lightning fast (stacked):
```
Face LoRA scale = 0.9
Lightning LoRA strength = 1.0
CFG = 2.0
Steps = 8
Sampler = euler
Scheduler = sgm_uniform
```

**Negative prompt (universal anti-AI-artifact):**
```
blurry, deformed, ugly, bad anatomy, low quality, plastic skin,
stray hairs, peach fuzz, facial fuzz, beauty marks, oversharpened,
skin noise, weird textures, double face, watermark, text,
cartoon, anime, 3d render, illustration, painting, blonde hair,
green eyes, blue eyes, wrong ethnicity
```

---

## Troubleshooting (errors actually hit + fixes)

| Symptom | Cause | Fix |
|---------|-------|-----|
| `avr_loss=nan` mid-training | fp16 overflow | Switch `--mixed_precision="bf16"` + add `--no_half_vae` + `--max_grad_norm=1.0` |
| `CUDA out of memory` during training | Activations + gradients too big | Add `--gradient_checkpointing` |
| `unrecognized arguments: --num_repeats=20` | Kohya idiom | Use folder name pattern: `20_token/` instead of CLI flag |
| `network for Text Encoder cannot be trained with caching` | Conflict | Add `--network_train_unet_only` |
| `No such file: sdxl_train_network.py` | Submodule path | `cd /workspace/kohya_ss/sd-scripts/` first |
| `ModuleNotFoundError: bitsandbytes` | Missing dep | `pip install bitsandbytes` |
| `unzip: command not found` | Missing tool | `apt-get install -y unzip` |
| Connection drop kills training | No tmux wrapper | Run inside `tmux new-session -d -s train '...'` |
| `PEFT backend is required for this method` | Multi-adapter API needs PEFT | `pip install peft` |
| `Quota exceeded` during checkpoint download | RunPod volume too small | Delete partial files + their `.aria2` metadata, skip non-essential checkpoints |
| `CUDA out of memory` during inference | Zombie process holding VRAM | `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`, then `kill -9 <pid>` or `pkill -9 python` |
| First inference takes 60+ sec with no output | JIT compile, not hang | Wait — check `nvidia-smi` shows GPU at 95-100% util |
| Output has peach fuzz / stray hairs | SDXL extreme-close-up problem | Drop "close-up" from prompt; use "portrait" instead |
| Output blonde hair when training was dark | Checkpoint prior winning | Raise LoRA scale 0.8 → 0.9-0.95 + add "blonde hair" to negative prompt |
| Output oversaturated "AI face" | CFG too high | Lower CFG 7.5 → 5.5 |
| Local PyTorch CUDA available: False | Driver / PyTorch version mismatch | Update NVIDIA Studio Driver to latest |

---

## Files in this kit

```
C:\Users\<USER>\Desktop\PROG\lora_training\
├── README.md                            ← this file
└── runpod_scripts\
    ├── MASTER_WORKFLOW.txt              ← full RunPod training workflow (Parts A-E)
    ├── INFERENCE_PASTE.txt              ← reusable inference paste-block + troubleshooting
    ├── sks_woman_workflow.json          ← ComfyUI workflow (drag onto canvas)
    └── runpod_sdxl_lora_setup.sh        ← .sh version of the training setup
```

**Memory store (for cross-session continuity):**
```
C:\Users\<USER>\.claude\projects\C--Users-<USER>\memory\
├── sdxl_lora_training.md                ← LoRA training learnings (settings + pitfalls)
└── model_sources_au.md                  ← HuggingFace URLs (Civitai geoblocked in AU)
```

---

## Reusability for the next LoRA

To train a NEW person's LoRA (say `xyz_man`):

1. **Photos:** new dataset in `C:\Users\<USER>\Desktop\kohya\lora_dataset_raw\` (20-30 photos)
2. **Captions:** use new trigger token (`xyz_man` instead of `sks_woman`)
3. **Zip + Drive upload:** new file → new GDRIVE_ID
4. **RunPod:** spin up pod, paste MASTER_WORKFLOW.txt with new `GDRIVE_ID` + `TRIGGER`
5. **Train + test:** same workflow, ~$0.30, ~1 hour
6. **Local:** drop new `.safetensors` into ComfyUI loras folder, swap in workflow's Load LoRA dropdown

**Same pipeline, different inputs.** Everything you built is reusable.

---

## Lessons from one full run (the non-obvious stuff)

1. **bf16 is mandatory for SDXL training. fp16 produces NaN loss within 1 epoch.** This is the single biggest training-killer.

2. **Tmux on RunPod is mandatory for any task >5 min.** Web terminals drop. SSH drops. Tmux is the only way to survive connection loss.

3. **Kohya's folder name encodes num_repeats.** `20_token/` not `--num_repeats=20`. This is a Kohya idiom that breaks first-time users.

4. **A/B test multiple epochs + checkpoints.** Don't assume final epoch is best. Don't assume one checkpoint is best. Test 5 epochs × 3 checkpoints = 15 generations = 3 min of pod time = informed decision.

5. **The artifacts you see in output are usually inference settings, not training data.** CFG, scale, prompt structure — fix those before retraining.

6. **Same seed for A/B testing.** Different seed = different noise = can't compare. Fix seed=42 for comparison shots, only vary AFTER you've isolated the right LoRA/checkpoint.

7. **Extreme close-ups are SDXL's weak spot.** Generate "portrait" framing instead, crop in post if you need tight shots.

8. **Civitai is geoblocked in Australia.** Use HuggingFace mirrors for everything.

9. **Download the LoRA + 5 checkpoint backups to your laptop before killing the pod.** The pod dies cheap; re-downloading costs you 30 min.

10. **The trigger token must be rare.** Real names ("george", "sarah") have semantic baggage. `sks_woman`, `g30rg3wu`, `ohwx_person` start from zero and bind strongly.
