"""Inference helpers for FLUX LoRA adapters."""

from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
from typing import Optional

import torch
from diffusers import BitsAndBytesConfig, FluxPipeline, FluxTransformer2DModel
from PIL.Image import Image
from transformers import T5EncoderModel
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig

DEFAULT_MODEL_ID = "black-forest-labs/FLUX.1-dev"


def _prepare_transformer(ckpt_id: str, compute_dtype: torch.dtype) -> FluxTransformer2DModel:
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )
    return FluxTransformer2DModel.from_pretrained(
        ckpt_id,
        subfolder="transformer",
        quantization_config=nf4_config,
        torch_dtype=torch.float16,
    )


def _prepare_text_encoder(ckpt_id: str) -> T5EncoderModel:
    quant_config = TransformersBitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    return T5EncoderModel.from_pretrained(
        ckpt_id,
        subfolder="text_encoder_2",
        quantization_config=quant_config,
        torch_dtype=torch.float16,
    )


def _build_pipeline(ckpt_id: str, compute_dtype: torch.dtype) -> FluxPipeline:
    transformer = _prepare_transformer(ckpt_id, compute_dtype)
    text_encoder = _prepare_text_encoder(ckpt_id)
    pipeline = FluxPipeline.from_pretrained(
        ckpt_id,
        transformer=transformer,
        text_encoder_2=text_encoder,
        torch_dtype=compute_dtype,
    )
    del text_encoder
    del transformer
    gc.collect()
    return pipeline


def _seed_generator(seed: Optional[int]) -> Optional[torch.Generator]:
    if seed is None:
        return None
    generator = torch.manual_seed(seed)
    return generator


def load_lora_pipeline(ckpt_id: str, lora_path: str, compute_dtype: torch.dtype) -> FluxPipeline:
    pipeline = _build_pipeline(ckpt_id, compute_dtype)
    pipeline.load_lora_weights(lora_path)
    return pipeline


def run_lora_inference(
    prompt: str,
    output_path: str,
    ckpt_id: str,
    lora_path: str,
    device: str,
    steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    seed: Optional[int],
) -> Image:
    compute_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipeline = load_lora_pipeline(ckpt_id, lora_path, compute_dtype)
    pipeline.to(device)
    generator = _seed_generator(seed)
    image = pipeline(
        prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        generator=generator,
    ).images[0]
    image.save(output_path)
    print(f"Saved image to {output_path}")
    if torch.cuda.is_available():
        print(f"Pipeline memory usage: {torch.cuda.max_memory_reserved() / 1024**3:.3f} GB")
    pipeline.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return image


def merge_lora_adapter(ckpt_id: str, lora_path: str, output_dir: str) -> None:
    compute_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipeline = load_lora_pipeline(ckpt_id, lora_path, compute_dtype)
    pipeline.fuse_lora()
    pipeline.unload_lora_weights()
    pipeline.save_pretrained(output_dir)
    print(f"Merged pipeline saved to {output_dir}")
    pipeline.to("cpu")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description="Run FLUX inference with LoRA adapters.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    adapter_parser = subparsers.add_parser("adapter", help="Load the adapter on top of the base model and generate an image.")
    adapter_parser.add_argument("--prompt", required=True)
    adapter_parser.add_argument("--output-path", default="output.png")
    adapter_parser.add_argument("--ckpt-id", default=DEFAULT_MODEL_ID)
    adapter_parser.add_argument("--lora-path", default="flux_lora")
    adapter_parser.add_argument("--device", default="cuda")
    adapter_parser.add_argument("--steps", type=int, default=28)
    adapter_parser.add_argument("--guidance-scale", type=float, default=3.5)
    adapter_parser.add_argument("--height", type=int, default=768)
    adapter_parser.add_argument("--width", type=int, default=512)
    adapter_parser.add_argument("--seed", type=int, default=0)

    merge_parser = subparsers.add_parser("merge", help="Fuse the adapter into the base FLUX checkpoint.")
    merge_parser.add_argument("--ckpt-id", default=DEFAULT_MODEL_ID)
    merge_parser.add_argument("--lora-path", default="flux_lora")
    merge_parser.add_argument("--output-dir", default="fused_flux")

    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "adapter":
        run_lora_inference(
            prompt=args.prompt,
            output_path=args.output_path,
            ckpt_id=args.ckpt_id,
            lora_path=args.lora_path,
            device=args.device,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            seed=args.seed,
        )
    elif args.command == "merge":
        merge_lora_adapter(
            ckpt_id=args.ckpt_id,
            lora_path=args.lora_path,
            output_dir=args.output_dir,
        )
    else:
        raise ValueError(f"Unsupported command {args.command}")


if __name__ == "__main__":
    main()
