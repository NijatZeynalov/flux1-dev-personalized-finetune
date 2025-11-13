"""Train FLUX LoRA adapters using cached text embeddings."""

from __future__ import annotations

import argparse
import copy
import gc
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path

import bitsandbytes as bnb
import diffusers
import torch
import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKL,
    BitsAndBytesConfig,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    FluxTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from peft.utils import get_peft_model_state_dict
from tqdm.auto import tqdm

from .dataset import DreamBoothDataset, collate_fn

logger = get_logger(__name__)


@dataclass
class TrainArgs:
    pretrained_model_name_or_path: str
    dataset_name: str
    data_df_path: str
    output_dir: str
    mixed_precision: str
    weighting_scheme: str
    width: int
    height: int
    train_batch_size: int
    learning_rate: float
    guidance_scale: float
    report_to: str | None
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    rank: int
    max_train_steps: int
    num_train_epochs: int
    seed: int | None
    checkpointing_steps: int
    max_sequence_length: int


def save_model_hook_factory(accelerator: Accelerator, transformer):
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                if isinstance(
                    accelerator.unwrap_model(model), type(accelerator.unwrap_model(transformer))
                ):
                    lora_layers = get_peft_model_state_dict(accelerator.unwrap_model(model))
                    FluxPipeline.save_lora_weights(
                        output_dir,
                        transformer_lora_layers=lora_layers,
                        text_encoder_lora_layers=None,
                    )
                    if weights:
                        weights.pop()

    return save_model_hook


def train(args: TrainArgs) -> None:
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=ProjectConfiguration(project_dir=args.output_dir, logging_dir=Path(args.output_dir, "logs")),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")

    nf4_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        quantization_config=nf4_config,
        torch_dtype=torch.float16,
    )
    transformer = prepare_model_for_kbit_training(transformer, use_gradient_checkpointing=False)

    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    vae.to(accelerator.device, dtype=torch.float16)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    transformer.add_adapter(transformer_lora_config)

    logger.info(
        "trainable params: %s || all params: %s",
        transformer.num_parameters(only_trainable=True),
        transformer.num_parameters(),
    )

    trainable_params = filter(lambda p: p.requires_grad, transformer.parameters())
    optimizer = bnb.optim.AdamW8bit(
        [{"params": list(trainable_params), "lr": args.learning_rate}],
        betas=(0.9, 0.999),
        weight_decay=1e-4,
        eps=1e-8,
    )

    train_dataset = DreamBoothDataset(
        data_df_path=args.data_df_path,
        dataset_name=args.dataset_name,
        width=args.width,
        height=args.height,
        max_sequence_length=args.max_sequence_length,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    vae_config = vae.config
    latents_cache = []
    for batch in tqdm(train_dataloader, desc="Caching latents"):
        with torch.no_grad():
            pixel_values = batch["pixel_values"].to(accelerator.device, dtype=torch.float16)
            latents_cache.append(vae.encode(pixel_values).latent_dist)

    del vae
    free_memory()

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.max_train_steps or args.num_train_epochs * num_update_steps_per_epoch
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=max_train_steps,
    )

    (
        transformer,
        optimizer,
        train_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(transformer, optimizer, train_dataloader, lr_scheduler)

    accelerator.register_save_state_pre_hook(save_model_hook_factory(accelerator, transformer))

    if args.mixed_precision == "fp16":
        cast_training_params([transformer], dtype=torch.float32)

    base_transformer = accelerator.unwrap_model(transformer)
    supports_guidance = getattr(base_transformer.config, "guidance_embeds", False)

    if accelerator.is_main_process:
        accelerator.init_trackers(
            "dreambooth-flux",
            config={
                "pretrained_model_name_or_path": args.pretrained_model_name_or_path,
                "learning_rate": args.learning_rate,
                "train_batch_size": args.train_batch_size,
                "rank": args.rank,
            },
        )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps.to(accelerator.device)]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    global_step = 0
    progress_bar = tqdm(range(max_train_steps), desc="Steps", disable=not accelerator.is_local_main_process)

    for epoch in range(num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                model_input = latents_cache[step].sample()
                model_input = (model_input - vae_config.shift_factor) * vae_config.scaling_factor
                model_input = model_input.to(dtype=torch.float16)

                latent_image_ids = FluxPipeline._prepare_latent_image_ids(
                    model_input.shape[0],
                    model_input.shape[2] // 2,
                    model_input.shape[3] // 2,
                    accelerator.device,
                    torch.float16,
                )

                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]

                u = compute_density_for_timestep_sampling(args.weighting_scheme, bsz, 0.0, 1.0, 1.29)
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                packed_noisy_model_input = FluxPipeline._pack_latents(
                    noisy_model_input,
                    model_input.shape[0],
                    model_input.shape[1],
                    model_input.shape[2],
                    model_input.shape[3],
                )

                guidance = (
                    torch.tensor([args.guidance_scale], device=accelerator.device).expand(bsz)
                    if supports_guidance
                    else None
                )

                model_pred = transformer(
                    hidden_states=packed_noisy_model_input,
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    pooled_projections=batch["pooled_prompt_embeds"].to(accelerator.device, dtype=torch.float16),
                    encoder_hidden_states=batch["prompt_embeds"].to(accelerator.device, dtype=torch.float16),
                    txt_ids=batch["text_ids"].to(accelerator.device, dtype=torch.float16),
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]

                vae_scale_factor = 2 ** (len(vae_config.block_out_channels) - 1)
                model_pred = FluxPipeline._unpack_latents(
                    model_pred,
                    model_input.shape[2] * vae_scale_factor,
                    model_input.shape[3] * vae_scale_factor,
                    vae_scale_factor,
                )

                weighting = compute_loss_weighting_for_sd3(args.weighting_scheme, sigmas)
                target = noise - model_input
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                ).mean()

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), 1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)

                if global_step % args.checkpointing_steps == 0 and (
                    accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED
                ):
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            accelerator.log(logs, step=global_step)

            if global_step >= max_train_steps:
                break

        if global_step >= max_train_steps:
            break

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        transformer_lora_layers = get_peft_model_state_dict(accelerator.unwrap_model(transformer))
        FluxPipeline.save_lora_weights(
            args.output_dir,
            transformer_lora_layers=transformer_lora_layers,
            text_encoder_lora_layers=None,
        )

    if torch.cuda.is_available():
        print(f"Pipeline memory usage: {torch.cuda.max_memory_reserved() / 1024**3:.3f} GB")
    else:
        print("Training completed.")

    accelerator.end_training()
    gc.collect()
    torch.cuda.empty_cache()


def parse_args() -> TrainArgs:
    parser = argparse.ArgumentParser(description="Train a FLUX LoRA adapter.")
    parser.add_argument("--pretrained-model-name-or-path", default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--dataset-name", default="nijatzeynalov/tiktik_xanim_v4")
    parser.add_argument("--data-df-path", default="embeddings.parquet")
    parser.add_argument("--output-dir", default="flux_lora")
    parser.add_argument("--mixed-precision", default="fp16", choices=["fp16", "bf16", "no"])
    parser.add_argument("--weighting-scheme", default="none")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--report-to", default=None, help="Optional tracker integration (e.g., wandb).")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--max-train-steps", type=int, default=700)
    parser.add_argument("--num-train-epochs", type=int, default=1)
    parser.add_argument("--max-seq-length", type=int, default=77)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpointing-steps", type=int, default=700)
    ns = parser.parse_args()
    return TrainArgs(
        pretrained_model_name_or_path=ns.pretrained_model_name_or_path,
        dataset_name=ns.dataset_name,
        data_df_path=ns.data_df_path,
        output_dir=ns.output_dir,
        mixed_precision=ns.mixed_precision,
        weighting_scheme=ns.weighting_scheme,
        width=ns.width,
        height=ns.height,
        train_batch_size=ns.train_batch_size,
        learning_rate=ns.learning_rate,
        guidance_scale=ns.guidance_scale,
        report_to=ns.report_to,
        gradient_accumulation_steps=ns.gradient_accumulation_steps,
        gradient_checkpointing=ns.gradient_checkpointing,
        rank=ns.rank,
        max_train_steps=ns.max_train_steps,
        num_train_epochs=ns.num_train_epochs,
        seed=ns.seed,
        checkpointing_steps=ns.checkpointing_steps,
        max_sequence_length=ns.max_seq_length,
    )


if __name__ == "__main__":
    train(parse_args())
