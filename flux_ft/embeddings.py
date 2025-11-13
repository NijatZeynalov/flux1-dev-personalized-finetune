"""Generate FLUX text embeddings that pair with DreamBooth images."""

from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import pandas as pd
import torch
from datasets import load_dataset
from diffusers import FluxPipeline
from huggingface_hub import login
from huggingface_hub.utils import insecure_hashlib
from transformers import T5EncoderModel


DEFAULT_MODEL_ID = "black-forest-labs/FLUX.1-dev"


def generate_image_hash(image) -> str:
    return insecure_hashlib.sha256(image.tobytes()).hexdigest()


def load_flux_dev_pipeline(model_id: str, device_map: str) -> FluxPipeline:
    text_encoder = T5EncoderModel.from_pretrained(
        model_id,
        subfolder="text_encoder_2",
        load_in_8bit=True,
        device_map="auto",
    )
    pipeline = FluxPipeline.from_pretrained(
        model_id,
        text_encoder_2=text_encoder,
        transformer=None,
        vae=None,
        device_map=device_map,
    )
    return pipeline


@torch.no_grad()
def compute_embeddings(
    pipeline: FluxPipeline,
    prompts: Sequence[str],
    max_sequence_length: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    all_prompt_embeds = []
    all_pooled_prompt_embeds = []
    all_text_ids = []
    for prompt in prompts:
        prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
            prompt=prompt, prompt_2=None, max_sequence_length=max_sequence_length
        )
        all_prompt_embeds.append(prompt_embeds)
        all_pooled_prompt_embeds.append(pooled_prompt_embeds)
        all_text_ids.append(text_ids)

    max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024 if torch.cuda.is_available() else 0.0
    print(f"Max memory allocated: {max_memory:.3f} GB")
    return all_prompt_embeds, all_pooled_prompt_embeds, all_text_ids


@dataclass
class EmbeddingArgs:
    dataset_name: str
    output_path: str
    model_id: str = DEFAULT_MODEL_ID
    max_sequence_length: int = 77
    device_map: str = "balanced"
    hf_token: str | None = None


def run(args: EmbeddingArgs) -> None:
    if args.hf_token:
        login(token=args.hf_token)

    dataset = load_dataset(args.dataset_name, split="train")
    image_prompts = {generate_image_hash(sample["image"]): sample["text"] for sample in dataset}
    all_prompts = list(image_prompts.values())
    print(f"Collected {len(all_prompts)} prompts from {args.dataset_name}")

    pipeline = load_flux_dev_pipeline(args.model_id, args.device_map)
    all_prompt_embeds, all_pooled_prompt_embeds, all_text_ids = compute_embeddings(
        pipeline, all_prompts, args.max_sequence_length
    )

    data = []
    for i, (image_hash, _) in enumerate(image_prompts.items()):
        data.append((image_hash, all_prompt_embeds[i], all_pooled_prompt_embeds[i], all_text_ids[i]))

    embedding_cols = ["prompt_embeds", "pooled_prompt_embeds", "text_ids"]
    df = pd.DataFrame(data, columns=["image_hash"] + embedding_cols)

    for col in embedding_cols:
        df[col] = df[col].apply(lambda x: x.cpu().numpy().flatten().tolist())

    df.to_parquet(args.output_path)
    print(f"Embeddings saved to {args.output_path}")

    del pipeline
    del dataset
    torch.cuda.empty_cache()
    gc.collect()


def parse_args() -> EmbeddingArgs:
    parser = argparse.ArgumentParser(description="Generate FLUX embeddings for DreamBooth fine-tuning.")
    parser.add_argument("--dataset-name", default="nijatzeynalov/tiktik_xanim_v4", help="HF dataset repo with 'image' and 'text'.")
    parser.add_argument("--output-path", default="embeddings.parquet", help="Where parquet embeddings will be written.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Base FLUX checkpoint to use.")
    parser.add_argument("--max-seq-length", type=int, default=77, help="Sequence length for the tokenizer.")
    parser.add_argument("--device-map", default="balanced", help="Device map passed to the FLUX pipeline.")
    parser.add_argument("--hf-token", default=None, help="Optional Hugging Face token. You can also login via CLI/env var.")
    ns = parser.parse_args()
    return EmbeddingArgs(
        dataset_name=ns.dataset_name,
        output_path=ns.output_path,
        model_id=ns.model_id,
        max_sequence_length=ns.max_seq_length,
        device_map=ns.device_map,
        hf_token=ns.hf_token,
    )


if __name__ == "__main__":
    run(parse_args())
