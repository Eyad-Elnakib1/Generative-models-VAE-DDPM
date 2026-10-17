# Repository file: src/genmodels/diffusion.py
"""Gaussian diffusion process of Ho et al. (2020).

Forward process q(x_t | x_0) with a fixed linear beta schedule, training with
L_simple (epsilon prediction), ancestral sampling (Algorithm 2), deterministic DDIM
sampling (Song et al., 2021, applied post hoc to the same network) and the full
variational bound in bits/dim (Section 3.3 + Appendix, discretised Gaussian decoder).
All schedule constants are computed in float64 and stored as float32 buffers.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import UNet


def get_beta_schedule(name: str, T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> np.ndarray:
    if name == "linear":
        return np.linspace(beta_start, beta_end, T, dtype=np.float64)
    raise ValueError(f"unknown beta schedule {name!r}")


def _extract(a: torch.Tensor, t: torch.Tensor, ndim: int) -> torch.Tensor:
    """a[t] reshaped to broadcast against a tensor with `ndim` dims."""
    return a.gather(0, t).reshape(-1, *([1] * (ndim - 1)))


def normal_kl(mean1, logvar1, mean2, logvar2):
    """KL(N(mean1, exp(logvar1)) || N(mean2, exp(logvar2))), elementwise, in nats."""
    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2)
                  + (mean1 - mean2) ** 2 * torch.exp(-logvar2))


def _approx_std_normal_cdf(x):
    return 0.5 * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def discretized_gaussian_log_likelihood(x, means, log_scales):
    """log p(x) for x on the 256-level grid in [-1, 1] (bin half-width 1/255), as in the official code."""
    centered = x - means
    inv_stdv = torch.exp(-log_scales)
    cdf_plus = _approx_std_normal_cdf(inv_stdv * (centered + 1.0 / 255.0))
    cdf_min = _approx_std_normal_cdf(inv_stdv * (centered - 1.0 / 255.0))
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    log_cdf_delta = torch.log((cdf_plus - cdf_min).clamp(min=1e-12))
    return torch.where(x < -0.999, log_cdf_plus,
                       torch.where(x > 0.999, log_one_minus_cdf_min, log_cdf_delta))


def _mean_flat(x):
    return x.mean(dim=tuple(range(1, x.ndim)))


class GaussianDiffusion(nn.Module):
    """Holds the fixed forward-process constants (buffers only, no parameters)."""

    def __init__(self, betas: np.ndarray, model_var_type: str = "fixedlarge", clip_denoised: bool = True):
        super().__init__()
        self.T = len(betas)
        self.model_var_type = model_var_type
        self.clip_denoised = clip_denoised

        alphas = 1.0 - betas
        ac = np.cumprod(alphas)
        ac_prev = np.append(1.0, ac[:-1])
        post_var = betas * (1.0 - ac_prev) / (1.0 - ac)
        consts = {
            "betas": betas,
            "alphas_cumprod": ac,
            "alphas_cumprod_prev": ac_prev,
            "sqrt_alphas_cumprod": np.sqrt(ac),
            "sqrt_one_minus_alphas_cumprod": np.sqrt(1.0 - ac),
            "sqrt_recip_alphas_cumprod": np.sqrt(1.0 / ac),
            "sqrt_recipm1_alphas_cumprod": np.sqrt(1.0 / ac - 1.0),
            "posterior_variance": post_var,
            # posterior variance is 0 at t=0, so the log is clipped to its t=1 value
            "posterior_log_variance_clipped": np.log(np.append(post_var[1], post_var[1:])),
            "posterior_mean_coef1": betas * np.sqrt(ac_prev) / (1.0 - ac),
            "posterior_mean_coef2": (1.0 - ac_prev) * np.sqrt(alphas) / (1.0 - ac),
            # sigma_t^2 = beta_t ("fixedlarge", used for CIFAR-10) or beta~_t ("fixedsmall")
            "logvar_fixedlarge": np.log(np.append(post_var[1], betas[1:])),
            "logvar_fixedsmall": np.log(np.append(post_var[1], post_var[1:])),
        }
        for name, value in consts.items():
            self.register_buffer(name, torch.tensor(value, dtype=torch.float32), persistent=False)

    # ---- forward process -------------------------------------------------------------
    def q_sample(self, x0, t, noise):
        return (_extract(self.sqrt_alphas_cumprod, t, x0.ndim) * x0
                + _extract(self.sqrt_one_minus_alphas_cumprod, t, x0.ndim) * noise)

    def q_posterior(self, x0, xt, t):
        mean = (_extract(self.posterior_mean_coef1, t, xt.ndim) * x0
                + _extract(self.posterior_mean_coef2, t, xt.ndim) * xt)
        return mean, _extract(self.posterior_log_variance_clipped, t, xt.ndim)

    def predict_x0_from_eps(self, xt, t, eps):
        return (_extract(self.sqrt_recip_alphas_cumprod, t, xt.ndim) * xt
                - _extract(self.sqrt_recipm1_alphas_cumprod, t, xt.ndim) * eps)

    # ---- reverse process -------------------------------------------------------------
    def p_mean_variance(self, model, xt, t, var_type=None):
        eps = model(xt, t).float()
        x0 = self.predict_x0_from_eps(xt, t, eps)
        if self.clip_denoised:
            x0 = x0.clamp(-1.0, 1.0)
        mean, _ = self.q_posterior(x0, xt, t)
        logvar = _extract(getattr(self, f"logvar_{var_type or self.model_var_type}"), t, xt.ndim)
        return mean, logvar, x0

    def training_loss(self, model, x0, generator=None):
        """L_simple: E_{t, eps} || eps - eps_theta(sqrt(a_bar) x0 + sqrt(1 - a_bar) eps, t) ||^2."""
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x0.device, generator=generator)
        noise = torch.randn(x0.shape, device=x0.device, generator=generator)
        eps = model(self.q_sample(x0, t, noise), t)
        return F.mse_loss(eps.float(), noise)

    @torch.no_grad()
    def p_sample_loop(self, model, shape, generator=None, device="cuda", progress=None, record_steps=()):
        """Algorithm 2 of the paper: T ancestral steps (no noise at the final step).

        With `record_steps`, also returns {t: (x_t, predicted x_0)} for those timesteps.
        """
        x = torch.randn(shape, device=device, generator=generator)
        trajectory = {}
        steps = range(self.T - 1, -1, -1)
        for i in (progress(steps) if progress else steps):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            mean, logvar, pred_x0 = self.p_mean_variance(model, x, t)
            if i in record_steps:
                trajectory[i] = (x.float().cpu(), pred_x0.float().cpu())
            if i > 0:
                x = mean + torch.exp(0.5 * logvar) * torch.randn(shape, device=device, generator=generator)
            else:
                x = mean
        return (x, trajectory) if record_steps else x

    def ddim_timesteps(self, num_steps: int):
        """Uniform sub-sequence of timesteps, as in the DDIM paper's CIFAR-10 setting."""
        skip = self.T // num_steps
        return list(range(0, self.T, skip))[:num_steps]

    @torch.no_grad()
    def ddim_sample_loop(self, model, shape, num_steps=100, eta=0.0, generator=None, device="cuda", progress=None,
                         x_T=None):
        """Deterministic for eta = 0: the sample is a fixed function of the starting noise x_T."""
        x = torch.randn(shape, device=device, generator=generator) if x_T is None else x_T
        seq = self.ddim_timesteps(num_steps)
        seq_prev = [-1] + seq[:-1]
        pairs = list(zip(reversed(seq), reversed(seq_prev)))
        for i, j in (progress(pairs) if progress else pairs):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            ab = self.alphas_cumprod[i]
            ab_prev = self.alphas_cumprod[j] if j >= 0 else torch.ones((), device=device)
            eps = model(x, t).float()
            x0 = (x - (1 - ab).sqrt() * eps) / ab.sqrt()
            if self.clip_denoised:
                x0 = x0.clamp(-1.0, 1.0)
                eps = (x - ab.sqrt() * x0) / (1 - ab).sqrt()  # keep eps consistent with the clipped x0
            sigma = eta * ((1 - ab_prev) / (1 - ab) * (1 - ab / ab_prev)).sqrt()
            x = ab_prev.sqrt() * x0 + (1 - ab_prev - sigma ** 2).clamp(min=0).sqrt() * eps
            if eta > 0 and j >= 0:
                x = x + sigma * torch.randn(shape, device=device, generator=generator)
        return x

    # ---- likelihood ------------------------------------------------------------------
    @torch.no_grad()
    def calc_bpd(self, model, x0, var_types=("fixedlarge",), generator=None, progress=None):
        """Variational bound -log p(x0) <= L_T + sum_{t>1} L_{t-1} + L_0, in bits/dim, per image.

        One forward pass per timestep serves every requested reverse-process variance, since
        they share the predicted mean. Returns {var_type: {"total", "prior", "terms"}}.
        """
        B, T = x0.shape[0], self.T
        terms = {v: torch.zeros(B, T, device=x0.device) for v in var_types}
        steps = range(T - 1, -1, -1)
        for i in (progress(steps) if progress else steps):
            t = torch.full((B,), i, device=x0.device, dtype=torch.long)
            noise = torch.randn(x0.shape, device=x0.device, generator=generator)
            xt = self.q_sample(x0, t, noise)
            true_mean, true_logvar = self.q_posterior(x0, xt, t)
            eps = model(xt, t).float()
            pred_x0 = self.predict_x0_from_eps(xt, t, eps)
            if self.clip_denoised:
                pred_x0 = pred_x0.clamp(-1.0, 1.0)
            model_mean, _ = self.q_posterior(pred_x0, xt, t)
            for v in var_types:
                model_logvar = _extract(getattr(self, f"logvar_{v}"), t, xt.ndim)
                if i == 0:
                    nll = -discretized_gaussian_log_likelihood(x0, model_mean, 0.5 * model_logvar)
                    terms[v][:, i] = _mean_flat(nll) / math.log(2.0)
                else:
                    kl = normal_kl(true_mean, true_logvar, model_mean, model_logvar)
                    terms[v][:, i] = _mean_flat(kl) / math.log(2.0)
        tT = torch.full((B,), T - 1, device=x0.device, dtype=torch.long)
        qT_mean = _extract(self.sqrt_alphas_cumprod, tT, x0.ndim) * x0
        qT_logvar = torch.log(_extract(1.0 - self.alphas_cumprod, tT, x0.ndim))
        prior = _mean_flat(normal_kl(qT_mean, qT_logvar, torch.zeros_like(x0), torch.zeros_like(x0))) / math.log(2.0)
        return {v: {"total": terms[v].sum(1) + prior, "prior": prior, "terms": terms[v]} for v in var_types}


class DDPM(nn.Module):
    """U-Net + diffusion process, operating on uint8 images (scaled to [-1, 1])."""

    def __init__(self, unet: dict, diffusion: dict):
        super().__init__()
        self.net = UNet(**unet)
        self.in_ch, self.resolution = self.net.conv_in.in_channels, unet.get("resolution", 32)
        betas = get_beta_schedule(diffusion.get("beta_schedule", "linear"), diffusion.get("T", 1000),
                                  diffusion.get("beta_start", 1e-4), diffusion.get("beta_end", 0.02))
        self.diffusion = GaussianDiffusion(betas, diffusion.get("model_var_type", "fixedlarge"),
                                           diffusion.get("clip_denoised", True))

    @staticmethod
    def preprocess(x_uint8):
        return x_uint8.float() / 127.5 - 1.0

    def forward(self, x_uint8):
        """Training loss for a batch of uint8 images; returns a dict with key 'loss'."""
        return {"loss": self.diffusion.training_loss(self.net, self.preprocess(x_uint8))}

    @torch.no_grad()
    def eval_metrics(self, x_uint8, generator):
        """L_simple on held-out images with fixed (t, noise), for overfitting checks."""
        return {"test_loss": self.diffusion.training_loss(self.net, self.preprocess(x_uint8), generator).item()}

    @torch.no_grad()
    def sample(self, n, sampler="ddpm", steps=None, generator=None, device="cuda", progress=None, eta=0.0):
        shape = (n, self.in_ch, self.resolution, self.resolution)
        if sampler == "ddpm":
            return self.diffusion.p_sample_loop(self.net, shape, generator, device, progress)
        if sampler == "ddim":
            return self.diffusion.ddim_sample_loop(self.net, shape, steps or 100, eta, generator, device, progress)
        raise ValueError(sampler)
