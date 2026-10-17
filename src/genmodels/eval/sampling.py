# Repository file: src/genmodels/eval/sampling.py
"""Loading trained runs and generating sample sets reproducibly.

Batch i of a sample set with seed s always uses noise seeded by (s, i), so every sample set is
reproducible, and DDIM sets with the same seed start from the same x_T (useful for comparing step counts).
"""
import time
from pathlib import Path

import numpy as np
import torch

from ..models import build_model
from ..utils import autocast_dtype, to_uint8


def load_run(run_dir, weights="ema", device="cuda", snapshot=None):
    """Model from `<run_dir>/ema_final.pt` (weights 'ema' or 'raw') or from an EMA snapshot file."""
    final = torch.load(Path(run_dir) / "ema_final.pt", map_location="cpu", weights_only=False)
    model = build_model(final["cfg"]["model"])
    if snapshot is not None:
        model.load_state_dict(torch.load(snapshot, map_location="cpu", weights_only=False)["ema"])
    else:
        model.load_state_dict(final["ema"] if weights == "ema" else final["model"])
    return model.to(device).eval(), final


def autocast(device):
    dtype = autocast_dtype("auto", torch.device(device))
    return torch.autocast("cuda", dtype=dtype or torch.float32, enabled=dtype is not None)


@torch.no_grad()
def generate(model, n, sampler="ddpm", steps=None, seed=0, batch_size=500, device="cuda"):
    """Returns (uint8 images [n, 3, 32, 32] on CPU, GPU seconds spent)."""
    out, seconds = [], 0.0
    for b, start in enumerate(range(0, n, batch_size)):
        g = torch.Generator(device).manual_seed(seed * 1_000_003 + b)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with autocast(device):
            x = model.sample(min(batch_size, n - start), sampler=sampler, steps=steps, generator=g, device=device)
        torch.cuda.synchronize()
        seconds += time.perf_counter() - t0
        out.append(to_uint8(x.float()).cpu())
    return torch.cat(out), seconds


@torch.no_grad()
def reconstruct(model, x_uint8, batch_size=500, device="cuda"):
    out = []
    for i in range(0, len(x_uint8), batch_size):
        with autocast(device):
            out.append(to_uint8(model.reconstruct(x_uint8[i:i + batch_size].to(device)).float()).cpu())
    return torch.cat(out)


def save_samples(path, images, **meta):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, images=images.numpy(), **meta)


def load_samples(path):
    return torch.from_numpy(np.load(path)["images"])
