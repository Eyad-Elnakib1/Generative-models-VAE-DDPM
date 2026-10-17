# Repository file: src/genmodels/ema.py
"""Exponential moving average of model weights (DDPM uses decay 0.9999, no warm-up)."""
import copy

import torch
import torch.nn as nn


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.model = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        ema_p = list(self.model.parameters())
        cur_p = [p.detach() for p in model.parameters()]
        torch._foreach_lerp_(ema_p, cur_p, 1.0 - self.decay)
        for eb, b in zip(self.model.buffers(), model.buffers()):
            eb.copy_(b)

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state):
        self.model.load_state_dict(state)
