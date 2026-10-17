import math

import numpy as np
import torch

from genmodels.diffusion import (DDPM, GaussianDiffusion, discretized_gaussian_log_likelihood,
                                 get_beta_schedule, normal_kl)
from genmodels.unet import UNet


def make_diffusion(T=1000):
    return GaussianDiffusion(get_beta_schedule("linear", T))


def test_unet_matches_paper_parameter_count():
    # Ho et al. (2020), Appendix B: "Our CIFAR10 model has 35.7 million parameters".
    n = sum(p.numel() for p in UNet().parameters())
    assert 35.6e6 < n < 35.8e6


def test_unet_shapes_and_zero_initialised_output():
    net = UNet(ch=32, ch_mult=(1, 2), num_res_blocks=1, attn_resolutions=(16,))
    out = net(torch.randn(2, 3, 32, 32), torch.tensor([0, 999]))
    assert out.shape == (2, 3, 32, 32)
    assert out.abs().max() < 1e-3  # last conv is zero-initialised as in the official code


def test_schedule_endpoints_make_L_T_negligible():
    d = make_diffusion()
    assert math.isclose(d.betas[0].item(), 1e-4, rel_tol=1e-5) and math.isclose(d.betas[-1].item(), 0.02, rel_tol=1e-5)
    assert d.alphas_cumprod[-1] < 1e-4  # x_T is almost pure noise, so L_T ~ 0


def test_q_sample_marginal_statistics():
    d, t = make_diffusion(), torch.full((200_000,), 300)
    x0 = torch.full((200_000, 1), 0.7)
    xt = d.q_sample(x0, t, torch.randn_like(x0))
    ab = d.alphas_cumprod[300].item()
    assert abs(xt.mean().item() - math.sqrt(ab) * 0.7) < 5e-3
    assert abs(xt.var().item() - (1 - ab)) < 5e-3


def test_posterior_matches_bayes_rule():
    """q(x_{t-1} | x_t, x_0) from the closed form vs. a direct Gaussian product.

    The reference is computed in float64 from the exact schedule. It must not be built from the float32 buffers:
    at t = 1, 1 - alpha_bar_0 = 1e-4 suffers catastrophic cancellation in float32 (relative error ~1e-3).
    """
    betas = get_beta_schedule("linear", 1000)
    d = GaussianDiffusion(betas)
    ab = np.cumprod(1.0 - betas)
    g = torch.Generator().manual_seed(0)
    x0, xt = torch.randn(5, 1, generator=g), torch.randn(5, 1, generator=g)
    for step in [1, 10, 500, 999]:
        t = torch.full((5,), step)
        mean, logvar = d.q_posterior(x0, xt, t)
        ab_prev, b = float(ab[step - 1]), float(betas[step])
        a = 1.0 - b
        prec = 1 / (1 - ab_prev) + a / b
        bayes_mean = (math.sqrt(ab_prev) * x0.double() / (1 - ab_prev) + math.sqrt(a) * xt.double() / b) / prec
        assert torch.allclose(mean.double(), bayes_mean, atol=1e-5)
        assert math.isclose(math.exp(logvar[0].item()), 1 / prec, rel_tol=1e-5)


class OracleEps(torch.nn.Module):
    """The exact noise predictor for data concentrated on a single image x*."""

    def __init__(self, d, x_star):
        super().__init__()
        self.d, self.x_star = d, x_star

    def forward(self, xt, t):
        ab = self.d.alphas_cumprod[t].view(-1, 1, 1, 1)
        return (xt - ab.sqrt() * self.x_star) / (1 - ab).sqrt()


def test_samplers_recover_point_mass_with_oracle_model():
    torch.manual_seed(0)
    d = make_diffusion()
    x_star = torch.rand(1, 3, 8, 8) * 2 - 1
    oracle = OracleEps(d, x_star)
    out = d.p_sample_loop(oracle, (4, 3, 8, 8), device="cpu")
    assert torch.allclose(out, x_star.expand_as(out), atol=1e-4)
    for steps in (10, 100, 1000):
        out = d.ddim_sample_loop(oracle, (4, 3, 8, 8), num_steps=steps, device="cpu")
        assert torch.allclose(out, x_star.expand_as(out), atol=1e-4)
        assert len(d.ddim_timesteps(steps)) == steps


def test_discretized_likelihood_is_normalised():
    grid = torch.arange(256).float() / 127.5 - 1.0
    for mean, log_scale in [(0.0, -1.0), (0.3, -4.0), (-0.99, -2.0)]:
        lp = discretized_gaussian_log_likelihood(grid, torch.full_like(grid, mean), torch.full_like(grid, log_scale))
        assert abs(lp.exp().sum().item() - 1.0) < 1e-4


def test_normal_kl_against_monte_carlo():
    torch.manual_seed(0)
    m1, lv1, m2, lv2 = 0.3, -0.5, -0.2, 0.4
    s = m1 + math.exp(0.5 * lv1) * torch.randn(2_000_000)
    log_q = -0.5 * (lv1 + math.log(2 * math.pi) + (s - m1) ** 2 / math.exp(lv1))
    log_p = -0.5 * (lv2 + math.log(2 * math.pi) + (s - m2) ** 2 / math.exp(lv2))
    analytic = normal_kl(torch.tensor(m1), torch.tensor(lv1), torch.tensor(m2), torch.tensor(lv2)).item()
    assert abs((log_q - log_p).mean().item() - analytic) < 3e-3


def test_bpd_with_oracle_has_zero_kl_terms():
    """With the exact model and the true posterior variance, every L_{t-1} (t > 1) is zero."""
    d = make_diffusion()
    x0 = (torch.randint(0, 256, (2, 3, 4, 4)).float() / 127.5) - 1.0
    res = d.calc_bpd(OracleEps(d, x0[:1]).eval(), x0[:1], var_types=("fixedsmall", "fixedlarge"))
    small, large = res["fixedsmall"], res["fixedlarge"]
    assert small["terms"][:, 1:].abs().max() < 1e-4
    assert large["terms"][:, 1:].min() >= -1e-6 and large["terms"][:, 1:].sum() > 0  # mismatched variance costs bits
    assert small["prior"].item() < 1e-3
    assert torch.allclose(small["total"], small["terms"].sum(1) + small["prior"])


def test_ddpm_loss_and_sampling_shapes():
    m = DDPM({"ch": 32, "ch_mult": [1, 2], "num_res_blocks": 1, "attn_resolutions": [16]}, {"T": 50})
    x = torch.randint(0, 256, (4, 3, 32, 32), dtype=torch.uint8)
    out = m(x)
    assert out["loss"].ndim == 0 and out["loss"].item() > 0.5  # untrained model predicts ~0, loss ~ E[eps^2] = 1
    assert m.sample(2, sampler="ddpm", device="cpu").shape == (2, 3, 32, 32)
    assert m.sample(2, sampler="ddim", steps=5, device="cpu").shape == (2, 3, 32, 32)
