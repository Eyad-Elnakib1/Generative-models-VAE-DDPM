# Repository file: src/genmodels/models.py
"""Model registry: builds a model from the `model` section of a config."""
from .diffusion import DDPM
from .vae import ConvVAE, MLPVAE

REGISTRY = {"ddpm": DDPM, "conv_vae": ConvVAE, "mlp_vae": MLPVAE}


def build_model(model_cfg: dict):
    cfg = dict(model_cfg)
    return REGISTRY[cfg.pop("type")](**cfg)


def network(model):
    """The neural network inside a model (what gets torch.compile'd)."""
    return model.net if hasattr(model, "net") else model
