# Repository file: src/genmodels/vae.py
"""Variational Autoencoder (Kingma & Welling, 2013 -- Auto-Encoding Variational Bayes).

Method exactly as in the paper (Sections 2.3-2.4, Appendix B/C.2):
  * Gaussian encoder q(z|x) = N(mu(x), diag sigma^2(x)), reparameterised z = mu + sigma * eps,
  * prior p(z) = N(0, I) with the KL term computed analytically (paper eq. 10),
  * Gaussian decoder p(x|z) = N(mu(z), diag sigma^2(z)) for real-valued data,
  * one Monte-Carlo sample per datapoint (L = 1), ELBO summed over all dimensions (beta = 1).

Two networks implement the same method:
  * `MLPVAE`  -- the paper's network: one tanh hidden layer per side (500 units), N(0, 0.01) init.
  * `ConvVAE` -- the same method with a convolutional encoder/decoder built from the DDPM
                 U-Net's blocks, so the VAE-vs-DDPM comparison is not an MLP-vs-U-Net one.

Pixels are uniformly dequantised before training/evaluation, which makes the Gaussian density on
continuous values a valid bound on the discrete likelihood (reported in bits/dim, comparable to
the DDPM's variational bound).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn_blocks import (AttnBlock, Downsample, ResnetBlock, Upsample, conv3x3,
                        linear, normalize)

LOG_2PI = math.log(2.0 * math.pi)


class GaussianVAE(nn.Module):
    latent_dim: int
    dec_logvar_min: float

    # --- implemented by subclasses ---------------------------------------------------------
    def encode(self, x):  # -> (mu, logvar) of q(z|x), each (B, latent_dim)
        raise NotImplementedError

    def decoder_raw(self, z):  # -> (mean, raw_logvar) of p(x|z), each (B, C, H, W)
        raise NotImplementedError

    # --- shared method ---------------------------------------------------------------------
    def decode(self, z):
        mean, raw = self.decoder_raw(z)
        # Soft lower bound on the decoder log-variance for numerical safety. The bound sits at the
        # variance of the dequantisation noise itself, below which no real gain in likelihood exists.
        lo = self.dec_logvar_min
        return mean.float(), lo + F.softplus(raw.float() - lo)

    @staticmethod
    def dequantize(x_uint8, generator=None):
        """uint8 pixel k -> (k + u) / 128 - 1, u ~ U[0, 1): a continuous value inside k's bin in [-1, 1)."""
        u = torch.rand(x_uint8.shape, device=x_uint8.device, generator=generator)
        return (x_uint8.float() + u) / 128.0 - 1.0

    @staticmethod
    def bin_centres(x_uint8):
        return (x_uint8.float() + 0.5) / 128.0 - 1.0

    def neg_elbo(self, x, generator=None):
        """Per-image -ELBO = E_q[-log p(x|z)] + KL(q(z|x) || N(0, I)), in nats, summed over dims."""
        mu, logvar = self.encode(x)
        mu, logvar = mu.float(), logvar.float().clamp(-30.0, 20.0)
        eps = torch.randn(mu.shape, device=mu.device, generator=generator)
        z = mu + torch.exp(0.5 * logvar) * eps
        x_mean, x_logvar = self.decode(z)
        recon = 0.5 * (x_logvar + LOG_2PI + (x - x_mean) ** 2 * torch.exp(-x_logvar)).flatten(1).sum(1)
        kl = 0.5 * (mu ** 2 + torch.exp(logvar) - logvar - 1.0).sum(1)
        return {"neg_elbo": recon + kl, "recon": recon, "kl": kl, "mu": mu}

    @staticmethod
    def nats_to_bpd(neg_elbo_nats, num_dims):
        """Continuous density on [-1, 1] with 256 bins of width 2/256 -> discrete NLL bound in bits/dim."""
        return (neg_elbo_nats + num_dims * math.log(128.0)) / (num_dims * math.log(2.0))

    def forward(self, x_uint8):
        x = self.dequantize(x_uint8)
        out = self.neg_elbo(x)
        D = x[0].numel()
        # Both terms are per-image sums divided by the same D: the per-dim scaling does not change
        # the relative weight of reconstruction vs. KL (beta stays exactly 1).
        return {"loss": out["neg_elbo"].mean() / D,
                "recon_per_dim": (out["recon"].mean() / D).detach(),
                "kl_per_dim": (out["kl"].mean() / D).detach()}

    @torch.no_grad()
    def eval_metrics(self, x_uint8, generator):
        x = self.dequantize(x_uint8, generator)
        out = self.neg_elbo(x, generator)
        D = x[0].numel()
        # Active units (Burda et al., 2016): latent dims whose posterior mean varies across data.
        active = (out["mu"].var(0) > 0.01).sum().item()
        return {"test_bpd": self.nats_to_bpd(out["neg_elbo"], D).mean().item(),
                "test_kl_nats": out["kl"].mean().item(),
                "active_units": active}

    @torch.no_grad()
    def sample(self, n, generator=None, device="cuda", **_):
        """Ancestral sample z ~ N(0, I) -> decoder mean (pixel noise is not added, as is standard)."""
        z = torch.randn((n, self.latent_dim), device=device, generator=generator)
        return self.decode(z)[0].clamp(-1.0, 1.0)

    @torch.no_grad()
    def reconstruct(self, x_uint8):
        mu, _ = self.encode(self.bin_centres(x_uint8))
        return self.decode(mu.float())[0].clamp(-1.0, 1.0)


class MLPVAE(GaussianVAE):
    """The paper's network (Appendix C): x -> tanh(500) -> (mu, logvar); z -> tanh(500) -> (mean, logvar)."""

    def __init__(self, img_shape=(3, 32, 32), hidden=500, latent_dim=200, init_std=0.01, dec_logvar_min=-12.0):
        super().__init__()
        self.img_shape, self.latent_dim, self.dec_logvar_min = tuple(img_shape), latent_dim, dec_logvar_min
        D = math.prod(self.img_shape)
        self.enc_h, self.enc_out = nn.Linear(D, hidden), nn.Linear(hidden, 2 * latent_dim)
        self.dec_h, self.dec_out = nn.Linear(latent_dim, hidden), nn.Linear(hidden, 2 * D)
        for p in self.parameters():  # "initialized by random sampling from N(0, 0.01)"
            nn.init.normal_(p, 0.0, init_std)

    def encode(self, x):
        return self.enc_out(torch.tanh(self.enc_h(x.flatten(1)))).chunk(2, dim=1)

    def decoder_raw(self, z):
        mean, logvar = self.dec_out(torch.tanh(self.dec_h(z))).chunk(2, dim=1)
        # Paper: means constrained with a sigmoid; 2*sigmoid - 1 is the same constraint on [-1, 1].
        mean = 2.0 * torch.sigmoid(mean) - 1.0
        return mean.view(-1, *self.img_shape), logvar.view(-1, *self.img_shape)


class ConvVAE(GaussianVAE):
    """Same VAE method, with an encoder mirroring the U-Net's downsampling path and a decoder
    mirroring its upsampling path (same ResNet/attention blocks, no skip connections)."""

    def __init__(self, in_ch=3, ch=128, ch_mult=(1, 2, 2, 2), num_res_blocks=2, attn_resolutions=(16,),
                 latent_dim=128, dropout=0.0, resamp_with_conv=True, resolution=32, dec_logvar_min=-12.0):
        super().__init__()
        self.latent_dim, self.dec_logvar_min = latent_dim, dec_logvar_min
        self.in_ch = in_ch

        def attn(c, r):
            return AttnBlock(c) if r in attn_resolutions else nn.Identity()

        # Encoder: 32x32 -> 4x4
        enc, cur, res = [conv3x3(in_ch, ch)], ch, resolution
        for level, mult in enumerate(ch_mult):
            for _ in range(num_res_blocks):
                enc += [ResnetBlock(cur, ch * mult, dropout=dropout), attn(ch * mult, res)]
                cur = ch * mult
            if level != len(ch_mult) - 1:
                enc.append(Downsample(cur, resamp_with_conv))
                res //= 2
        enc += [ResnetBlock(cur, cur, dropout=dropout), AttnBlock(cur), ResnetBlock(cur, cur, dropout=dropout),
                normalize(cur), nn.SiLU(), nn.Flatten()]
        self.encoder = nn.Sequential(*enc)
        self.enc_out = linear(cur * res * res, 2 * latent_dim)

        # Decoder: 4x4 -> 32x32
        self.bottleneck = (cur, res)
        self.dec_in = linear(latent_dim, cur * res * res)
        dec = [ResnetBlock(cur, cur, dropout=dropout), AttnBlock(cur), ResnetBlock(cur, cur, dropout=dropout)]
        for level, mult in reversed(list(enumerate(ch_mult))):
            for _ in range(num_res_blocks + 1):
                dec += [ResnetBlock(cur, ch * mult, dropout=dropout), attn(ch * mult, res)]
                cur = ch * mult
            if level != 0:
                dec.append(Upsample(cur, resamp_with_conv))
                res *= 2
        dec += [normalize(cur), nn.SiLU(), conv3x3(cur, 2 * in_ch, init_scale=0.0)]
        self.decoder = nn.Sequential(*dec)

    def encode(self, x):
        return self.enc_out(self.encoder(x)).chunk(2, dim=1)

    def decoder_raw(self, z):
        c, r = self.bottleneck
        out = self.decoder(self.dec_in(z).view(-1, c, r, r))
        return out[:, :self.in_ch], out[:, self.in_ch:]
