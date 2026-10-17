# Repository file: src/genmodels/data.py
"""CIFAR-10 kept entirely on the GPU as uint8 (150 MB), with deterministic, resumable shuffling.

Batch `step` is a pure function of (seed, step): epoch e uses a permutation seeded by (seed, e)
and the flip mask is seeded by (seed, step). A run resumed from a checkpoint therefore sees exactly
the batches it would have seen without the interruption, and no DataLoader workers are needed.
"""
import numpy as np
import torch
from torchvision.datasets import CIFAR10


def load_cifar10(root: str = "data", train: bool = True):
    """Returns (images uint8 [N, 3, 32, 32], labels int64 [N])."""
    ds = CIFAR10(root=root, train=train, download=True)
    images = torch.from_numpy(np.ascontiguousarray(ds.data)).permute(0, 3, 1, 2).contiguous()
    return images, torch.tensor(ds.targets, dtype=torch.long)


class GPUBatcher:
    def __init__(self, images: torch.Tensor, batch_size: int, seed: int = 0, hflip: bool = True, device="cuda"):
        self.images = images.to(device)
        self.batch_size, self.seed, self.hflip, self.device = batch_size, seed, hflip, device
        self.steps_per_epoch = len(images) // batch_size  # drop the last partial batch
        self._epoch, self._perm = None, None

    def _permutation(self, epoch):
        if epoch != self._epoch:
            g = torch.Generator().manual_seed(self.seed * 1_000_003 + epoch)
            self._perm = torch.randperm(len(self.images), generator=g).to(self.device)
            self._epoch = epoch
        return self._perm

    def batch(self, step: int) -> torch.Tensor:
        epoch, i = divmod(step, self.steps_per_epoch)
        idx = self._permutation(epoch)[i * self.batch_size:(i + 1) * self.batch_size]
        x = self.images[idx]
        if self.hflip:
            g = torch.Generator().manual_seed(self.seed * 7_919 + step)
            flip = (torch.rand(len(idx), generator=g) < 0.5).to(self.device)
            x = torch.where(flip[:, None, None, None], x.flip(3), x)
        return x
