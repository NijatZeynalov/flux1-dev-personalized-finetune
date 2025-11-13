# FLUX.1-dev Fine-Tuning for Personalized Image Generation

This repository provides a modular implementation of FLUX.1-dev fine-tuning for personalized image generation using Tıktık xanım imagery. Tıktık xanım is a beloved Azerbaijani fairy-tale heroine, and this repo keeps the whole fine-tuning workflow reusable: prompt embedding generation, DreamBooth-style LoRA training, inference, merging, and Hugging Face Hub upload, all under `src/flux_ft`.

Training was performed on an NVIDIA H100 GPU using a small custom dataset consisting of 13 annotated samples.
Each sample includes a single frame of Tık-tık khanum, a descriptive text prompt and consistent character portrayal (pose, expression, style)

The dataset was intentionally kept compact to evaluate how well FLUX.1-dev adapts to a character with minimal training examples.

<img width="1365" height="726" alt="Image" src="https://github.com/user-attachments/assets/7323f328-7e47-424f-81d2-61d14d6af5bc" />

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate         # PowerShell on Windows
pip install -r requirements.txt
```

## Usage

All scripts can be invoked via `python -m flux_ft.<module>`. Below are the most common flows; tweak arguments as needed.

### Pretrained Adapter

You can download or reference the already-trained LoRA adapter on Hugging Face Hub at https://huggingface.co/nijatzeynalov/tik_tik_xanim_qlora_flux.

### 1. Generate Prompt Embeddings

```bash
python -m flux_ft.embeddings \
  --dataset-name nijatzeynalov/tiktik_xanim_v4 \
  --output-path embeddings_tiktik.parquet
```

### 2. Train LoRA Adapters

```bash
python -m flux_ft.training \
  --data-df-path embeddings_tiktik.parquet \
  --output-dir tiktik_flux_lora \
  --max-train-steps 700
```

### 3. Run Inference (load adapters on the fly)

```bash
python -m flux_ft.inference adapter \
  --prompt "Tıktık xanım stares at the Baku Flame Towers with amazement" \
  --lora-path tiktik_flux_lora \
  --output-path tiktik.png
```

### 4. Merge the Adapter Into the Base Model

```bash
python -m flux_ft.inference merge \
  --lora-path tiktik_flux_lora \
  --output-dir fused_flux_model
```

### 5. Upload to Hugging Face Hub

```bash
python -m flux_ft.hf_upload \
  --repo-id nijatzeynalov/tik_tik_xanim_qlora_flux \
  --folder-path tiktik_flux_lora
```
---
Below are the exact prompts used and sample images generated from the fine-tuned FLUX.1-dev LoRA.

*Tık-tık khanum standing on a busy city street, looking around with a relaxed smile as people walk and cars pass by behind her.*

<img width="512" height="768" alt="Image" src="https://github.com/user-attachments/assets/49cbc62e-6e8a-404d-81e4-8e3acb526126" />

<br><br>

*Tık-tık khanum standing among plants, holding a small glowing lantern in her hand, smiling confidently as she lights her surroundings.*

<img width="512" height="768" alt="Image" src="https://github.com/user-attachments/assets/03c3d767-430f-4d07-8811-554e90661d79" />

<br><br>

*Tık-tık khanum kneeling near a small campfire, warming her hands while watching the flames with a calm and thoughtful expression.*

<img width="512" height="768" alt="Image" src="https://github.com/user-attachments/assets/fcecf106-24e1-4c87-a8f4-0f843508f5a9" />

---

Each CLI exposes `--help` with every tunable flag so you can reproduce or customize the workflow.
