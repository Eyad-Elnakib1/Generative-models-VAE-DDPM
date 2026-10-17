import torch

from genmodels.eval import metrics as M
from genmodels.eval.classifier import ResNet18, _augment


def test_precision_recall_identical_and_disjoint_sets():
    torch.manual_seed(0)
    a = torch.randn(500, 16)
    p, r = M.precision_recall(a, a.clone(), device="cpu", chunk=200)
    assert p == 1.0 and r == 1.0
    p, r = M.precision_recall(a, a + 100.0, device="cpu", chunk=200)
    assert p == 0.0 and r == 0.0


def test_precision_recall_detects_mode_collapse():
    """Samples covering only part of the data: high precision (realistic) but low recall (not diverse)."""
    torch.manual_seed(0)
    real = torch.randn(2000, 2)
    part = real[real[:, 0] > 0.5]  # ~30% of the data
    collapsed = part + 0.01 * torch.randn_like(part)
    p, r = M.precision_recall(real, collapsed, device="cpu", chunk=500)
    assert p > 0.9 and r < 0.5


def test_nearest_neighbours_finds_exact_copies():
    ref = torch.randn(300, 32)
    d, i = M.nearest_neighbours(ref[[5, 17, 250]], ref, device="cpu")
    assert i.tolist() == [5, 17, 250] and d.abs().max() < 1e-2  # cdist's fast path has ~1e-3 rounding


def test_memorisation_stats_baseline():
    test = torch.rand(10_000) + 1.0
    s = M.memorisation_stats(test.clone(), test)
    assert abs(s["nn_dist_median_ratio_to_test"] - 1.0) < 1e-6 and abs(s["frac_below_test_p1"] - 0.01) < 2e-3
    s = M.memorisation_stats(torch.zeros(100), test)  # a copying model
    assert s["frac_below_test_p1"] == 1.0


def test_psnr_and_ssim():
    x = torch.randint(0, 256, (4, 3, 32, 32), dtype=torch.uint8)
    assert torch.allclose(M.ssim(x, x), torch.ones(4), atol=1e-5)
    assert (M.psnr(x, x) > 90).all()
    noisy = (x.float() + 20 * torch.randn(x.shape)).clamp(0, 255).to(torch.uint8)
    assert (M.ssim(x, noisy) < 1).all() and (M.psnr(x, noisy) < 30).all()


def test_class_statistics():
    balanced = torch.eye(10).repeat(10, 1)  # every class equally often, fully confident
    s = M.class_statistics(balanced)
    assert abs(s["class_entropy"] - 1.0) < 1e-6 and abs(s["min_class_frac"] - 0.1) < 1e-6 and s["confidence"] == 1.0
    one_class = torch.zeros(50, 10)
    one_class[:, 3] = 1
    s = M.class_statistics(one_class)
    assert s["class_entropy"] == 0.0 and s["min_class_frac"] == 0.0


def test_classifier_and_augmentation_shapes():
    x = torch.randint(0, 256, (6, 3, 32, 32), dtype=torch.uint8)
    aug = _augment(x, torch.Generator().manual_seed(0))
    assert aug.shape == x.shape and aug.dtype == torch.uint8
    assert ResNet18()(x).shape == (6, 10)
