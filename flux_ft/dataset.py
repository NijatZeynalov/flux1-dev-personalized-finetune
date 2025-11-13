"""Dataset utilities used by the FLUX DreamBooth fine-tuning scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from huggingface_hub.utils import insecure_hashlib
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms


TensorTuple = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class DreamBoothDataset(Dataset):
    """Loads images and precomputed text embeddings for DreamBooth-style training."""

    def __init__(
        self,
        data_df_path: str | Path,
        dataset_name: str,
        width: int,
        height: int,
        max_sequence_length: int = 77,
    ) -> None:
        self.width = width
        self.height = height
        self.max_sequence_length = max_sequence_length
        self.data_df_path = Path(data_df_path)

        if not self.data_df_path.exists():
            raise FileNotFoundError(f"`data_df_path` not found: {self.data_df_path}")

        dataset = load_dataset(dataset_name, split="train")
        self.instance_images = [sample["image"] for sample in dataset]
        self.image_hashes = [insecure_hashlib.sha256(img.tobytes()).hexdigest() for img in self.instance_images]
        self.pixel_values = self._apply_transforms()
        self.data_dict = self._map_embeddings()
        self._length = len(self.instance_images)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        idx = index % len(self.instance_images)
        hash_key = self.image_hashes[idx]
        prompt_embeds, pooled_prompt_embeds, text_ids = self.data_dict[hash_key]
        return {
            "instance_images": self.pixel_values[idx],
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "text_ids": text_ids,
        }

    def _apply_transforms(self) -> torch.Tensor:
        transform = transforms.Compose(
            [
                transforms.Resize((self.height, self.width), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.RandomCrop((self.height, self.width)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        pixel_values = []
        for image in self.instance_images:
            image = exif_transpose(image).convert("RGB") if image.mode != "RGB" else exif_transpose(image)
            pixel_values.append(transform(image))
        return torch.stack(pixel_values)

    def _map_embeddings(self) -> Dict[str, TensorTuple]:
        df = pd.read_parquet(self.data_df_path)
        data_dict: Dict[str, TensorTuple] = {}
        for _, row in df.iterrows():
            prompt_embeds = torch.from_numpy(np.array(row["prompt_embeds"]).reshape(self.max_sequence_length, 4096))
            pooled_prompt_embeds = torch.from_numpy(np.array(row["pooled_prompt_embeds"]).reshape(768))
            text_ids = torch.from_numpy(np.array(row["text_ids"]).reshape(self.max_sequence_length, -1))
            data_dict[row["image_hash"]] = (prompt_embeds, pooled_prompt_embeds, text_ids)
        return data_dict


def collate_fn(examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Batch samples that already contain embeddings."""

    pixel_values = torch.stack([ex["instance_images"] for ex in examples]).float()
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    prompt_embeds = torch.stack([ex["prompt_embeds"] for ex in examples])
    pooled_prompt_embeds = torch.stack([ex["pooled_prompt_embeds"] for ex in examples])
    text_ids = torch.stack([ex["text_ids"] for ex in examples])[0]

    return {
        "pixel_values": pixel_values,
        "prompt_embeds": prompt_embeds,
        "pooled_prompt_embeds": pooled_prompt_embeds,
        "text_ids": text_ids,
    }
