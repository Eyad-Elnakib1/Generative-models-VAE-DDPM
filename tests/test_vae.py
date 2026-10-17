import math

import torch

from genmodels.vae import ConvVAE, GaussianVAE, MLPVAE


def small_conv_vae(**kw):
    return ConvVAE(ch=32, ch_mult=(1, 2), num_res_blocks=1, attn_resolutions=(16,), latent_dim=16, **kw)


def test_conv_vae_has_capacity_comparable_to_unet():
    n = sum(p.numel() for p in ConvVAE().parameters())
    assert 25e6 < n < 40e6  # U-Net: 35.7M


def test_mlp_vae_matches_paper_network():
    m = MLPVAE()
    assert m.enc_h.out_features == 500 and m.dec_h.out_features == 500 and m.latent_dim == 200
    stds = [p.std().item() for p in m.parameters() if p.numel() > 1000]
    assert all(abs(s - 0.01) < 1e-3 for s in stds)  # N(0, 0.01) init


def test_neg_elbo_is_sum_of_per_image_terms_beta_one():
    """Guards the classic bug: mean-reduced reconstruction + summed KL (an implicit beta ~ 3000)."""
    torch.manual_seed(0)
    m = small_conv_vae()
    x_uint8 = torch.randint(0, 256, (4, 3, 32, 32), dtype=torch.uint8)
    torch.manual_seed(1)
    out = m(x_uint8)
    torch.manual_seed(1)
    parts = m.neg_elbo(m.dequantize(x_uint8))
    D = 3 * 32 * 32
    assert torch.allclose(out["loss"], parts["neg_elbo"].mean() / D)
    assert torch.allclose(out["recon_per_dim"] + out["kl_per_dim"], out["loss"], atol=1e-6)
    assert torch.allclose(parts["neg_elbo"], parts["recon"] + parts["kl"])


def test_analytic_kl_matches_monte_carlo():
    torch.manual_seed(0)
    mu, logvar = torch.tensor([0.5, -1.0]), torch.tensor([-0.3, 0.8])
    analytic = 0.5 * (mu ** 2 + logvar.exp() - logvar - 1).sum()
    z = mu + (0.5 * logvar).exp() * torch.randn(1_000_000, 2)
    log_q = (-0.5 * (logvar + math.log(2 * math.pi) + (z - mu) ** 2 / logvar.exp())).sum(1)
    log_p = (-0.5 * (math.log(2 * math.pi) + z ** 2)).sum(1)
    assert abs((log_q - log_p).mean().item() - analytic.item()) < 5e-3


def test_bpd_conversion_uniform_density_is_8_bits():
    # A uniform density on [-1, 1] has log p = -ln 2 per dim; the discrete data then costs 8 bits/dim.
    D = 3072
    bpd = GaussianVAE.nats_to_bpd(torch.tensor(D * math.log(2.0)), D)
    assert abs(bpd.item() - 8.0) < 1e-6


def test_dequantisation_stays_inside_pixel_bins():
    x = torch.arange(256, dtype=torch.uint8).repeat(100)
    y = GaussianVAE.dequantize(x)
    k = torch.floor((y + 1) * 128).long()
    assert torch.equal(k, x.long()) and y.min() >= -1 and y.max() < 1


def test_reparameterisation_gradients_reach_encoder():
    m = small_conv_vae()
    m(torch.randint(0, 256, (2, 3, 32, 32), dtype=torch.uint8))["loss"].backward()
    assert m.enc_out.weight.grad is not None and m.enc_out.weight.grad.abs().sum() > 0


def test_sample_and_reconstruct_shapes_and_range():
    for m in (small_conv_vae(), MLPVAE()):
        s = m.sample(3, device="cpu")
        r = m.reconstruct(torch.randint(0, 256, (2, 3, 32, 32), dtype=torch.uint8))
        assert s.shape == (3, 3, 32, 32) and r.shape == (2, 3, 32, 32)
        assert s.min() >= -1 and s.max() <= 1


def test_decoder_logvar_lower_bound():
    m = small_conv_vae(dec_logvar_min=-5.0)
    _, logvar = m.decode(torch.randn(2, 16) * 100)
    assert logvar.min() >= -5.0
