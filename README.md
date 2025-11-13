# Flux Tıktık Xanım LoRA Project

This repository packages the Colab notebook you used for fine-tuning **FLUX** on Tıktık xanım imagery into reusable Python modules. Tıktık xanım is a beloved Azerbaijani nağıl qəhramanı (fairy-tale heroine), and this repo keeps the whole fine-tuning workflow reusable: prompt embedding generation, DreamBooth-style LoRA training, inference, merging, and Hugging Face Hub upload, all under `src/flux_ft`.

> **Heads up:** the scripts are meant to be executed manually when you are ready. This conversion did not run any training or inference code.

## Project Layout

```
src/flux_ft/
|-- embeddings.py   # Generate prompt embeddings and serialize them to Parquet
|-- dataset.py      # DreamBooth dataset + collate_fn utilities
|-- training.py     # LoRA fine-tuning loop
|-- inference.py    # Inference & adapter fusion helpers
`-- hf_upload.py    # Push trained adapters to Hugging Face Hub
```

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

Each CLI exposes `--help` with every tunable flag from the original notebook so you can reproduce or customize the workflow without touching the code again.
