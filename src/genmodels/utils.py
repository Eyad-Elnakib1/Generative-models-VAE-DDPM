# Repository file: src/genmodels/utils.py
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path, overrides=()):
    """Load a YAML config and apply `a.b.c=value` overrides (values parsed as YAML)."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for item in overrides:
        key, value = item.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(value)
    return cfg


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def atomic_save(obj, path):
    """Write to a temp file then rename, so a disconnect mid-write never corrupts the checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def autocast_dtype(precision: str, device: torch.device):
    """'auto' -> bf16 on Ampere or newer (A100, L4), else fp32. fp16 is not used (unstable for diffusion)."""
    if device.type != "cuda" or precision == "fp32":
        return None
    if precision == "bf16" or (precision == "auto" and torch.cuda.get_device_capability(device)[0] >= 8):
        return torch.bfloat16
    return None


def to_uint8(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] float images -> uint8 in [0, 255] (same quantisation as the official DDPM code)."""
    return ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)


def save_grid(x: torch.Tensor, path, nrow=8):
    from torchvision.utils import save_image
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_image(to_uint8(x).float() / 255.0, path, nrow=nrow, padding=2)
