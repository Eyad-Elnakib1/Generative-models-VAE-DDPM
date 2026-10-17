import json

import pytest
import torch

from genmodels.data import GPUBatcher
from genmodels.trainer import train

TINY_DDPM = {
    "name": "tiny_ddpm", "seed": 0, "data": {"hflip": True},
    "model": {"type": "ddpm", "unet": {"ch": 32, "ch_mult": [1, 2], "num_res_blocks": 1, "attn_resolutions": [16],
                                       "dropout": 0.1},
              "diffusion": {"T": 100}},
    "train": {"batch_size": 8, "total_steps": 40, "optim": {"name": "adam", "lr": 2e-4}, "warmup_steps": 5,
              "grad_clip": 1.0, "ema_decay": 0.99, "precision": "fp32", "log_every": 10, "eval_every": 20,
              "sample_every": 1000, "snapshot_every": 20, "ckpt_every_minutes": 0, "eval_images": 8},
}


def fake_data(n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (n, 3, 32, 32), dtype=torch.uint8, generator=g)


def test_batcher_is_deterministic_and_covers_epoch():
    data = torch.arange(64, dtype=torch.uint8).view(64, 1, 1, 1).expand(64, 3, 32, 32).contiguous()
    a, b = GPUBatcher(data, 16, seed=3, device="cpu"), GPUBatcher(data, 16, seed=3, device="cpu")
    assert all(torch.equal(a.batch(s), b.batch(s)) for s in range(10))
    seen = torch.cat([a.batch(s)[:, 0, 0, 0] for s in range(4)])
    assert sorted(seen.tolist()) == list(range(64))  # one epoch = a permutation of the data


def test_resume_reproduces_uninterrupted_run(tmp_path):
    torch.use_deterministic_algorithms(True)
    try:
        data = fake_data()
        _, ema_a = train(TINY_DDPM, tmp_path / "a", data, data, device="cpu", log=lambda *_: None)
        train(TINY_DDPM, tmp_path / "b", data, data, device="cpu", max_steps=20, log=lambda *_: None)
        _, ema_b = train(TINY_DDPM, tmp_path / "b", data, data, device="cpu", log=lambda *_: None)
    finally:
        torch.use_deterministic_algorithms(False)
    for pa, pb in zip(ema_a.model.parameters(), ema_b.model.parameters()):
        assert torch.allclose(pa, pb, atol=1e-6)
    rows = [json.loads(l) for l in open(tmp_path / "a" / "metrics.jsonl")]
    assert any("ema_test_loss" in r for r in rows) and any("loss" in r for r in rows)
    assert (tmp_path / "a" / "ema_final.pt").exists() and (tmp_path / "a" / "snapshots" / "ema_0000020.pt").exists()


def test_tiny_models_overfit_a_batch():
    """Sanity check of each model + optimiser: loss on one fixed batch must drop substantially."""
    from genmodels.models import build_model
    torch.manual_seed(0)
    x = fake_data(8)
    for model_cfg, lr in [
        ({"type": "conv_vae", "ch": 32, "ch_mult": [1, 2], "num_res_blocks": 1, "attn_resolutions": [], "latent_dim": 16}, 1e-3),
        ({"type": "mlp_vae", "hidden": 64, "latent_dim": 8}, 1e-3),
    ]:
        m = build_model(model_cfg)
        opt = torch.optim.Adam(m.parameters(), lr=lr)
        first = None
        for _ in range(150):
            loss = m(x)["loss"]
            opt.zero_grad()
            loss.backward()
            opt.step()
            first = first if first is not None else loss.item()
        assert loss.item() < first - 0.3, (model_cfg["type"], first, loss.item())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_resume_on_gpu(tmp_path):
    """Regression: resuming on a GPU used to fail because RNG states were loaded onto the GPU."""
    data = torch.randint(0, 256, (64, 3, 32, 32), dtype=torch.uint8)
    train(TINY_DDPM, tmp_path / "r", data, data, device="cuda", max_steps=10, log=lambda *_: None)
    _, ema = train(TINY_DDPM, tmp_path / "r", data, data, device="cuda", max_steps=20, log=lambda *_: None)
    assert torch.load(tmp_path / "r" / "ema_final.pt", weights_only=False)["step"] == 20
