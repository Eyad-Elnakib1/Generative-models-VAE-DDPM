# Repository file: src/genmodels/eval/metrics.py
"""Sample-quality metrics computed from Inception-v3 features.

FID, IS and KID use torch-fidelity's reference implementations (validated against the original
TensorFlow code used by the DDPM paper). Inception features are extracted once per sample set and
reused for every metric, including our own implementations of:
  * precision / recall (Kynkaenniemi et al., 2019): realism vs. coverage of the data,
  * nearest-neighbour memorisation check against the training set,
  * PSNR / SSIM for VAE reconstructions.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F


class InceptionFeatures:
    """Pool (2048-d) features and unbiased logits of the TF-compatible Inception-v3 used for FID/IS."""

    def __init__(self, device):
        from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
        self.net = FeatureExtractorInceptionV3("inception-v3-compat", ["2048", "logits_unbiased"]).to(device).eval()
        self.device = device
        # images are resized to 299x299 inside the network: ~250 per batch needs several GB
        total_gb = torch.cuda.get_device_properties(device).total_memory / 1e9 if str(device) != "cpu" else 0
        self.batch_size = 250 if total_gb >= 12 else 32

    @torch.no_grad()
    def __call__(self, images_uint8: torch.Tensor, batch_size: int = None):
        batch_size = batch_size or self.batch_size
        pools, logits = [], []
        for i in range(0, len(images_uint8), batch_size):
            p, l = self.net(images_uint8[i:i + batch_size].to(self.device))
            pools.append(p.float().cpu())
            logits.append(l.float().cpu())
        return torch.cat(pools), torch.cat(logits)


def fid_statistics(pool: torch.Tensor):
    from torch_fidelity.metric_fid import fid_features_to_statistics
    return fid_features_to_statistics(pool.double())


def fid_from_statistics(stat_fake, stat_real) -> float:
    from torch_fidelity.metric_fid import fid_statistics_to_metric
    return float(fid_statistics_to_metric(stat_fake, stat_real, verbose=False)["frechet_inception_distance"])


def inception_score(logits: torch.Tensor, splits: int = 10):
    from torch_fidelity.metric_isc import isc_features_to_metric
    out = isc_features_to_metric(logits, splits=splits, shuffle=True, rng_seed=2020)
    return float(out["inception_score_mean"]), float(out["inception_score_std"])


def kid(pool_fake: torch.Tensor, pool_real: torch.Tensor, subset_size: int = 1000, subsets: int = 100):
    from torch_fidelity.metric_kid import kid_features_to_metric
    out = kid_features_to_metric(pool_fake, pool_real, kid_subset_size=min(subset_size, len(pool_fake)),
                                 kid_subsets=subsets, verbose=False, rng_seed=2020)
    return float(out["kernel_inception_distance_mean"]), float(out["kernel_inception_distance_std"])


def _kth_nn_radius(feats: torch.Tensor, k: int, chunk: int) -> torch.Tensor:
    """Distance from each point to its k-th nearest neighbour in the same set (self excluded)."""
    radii = []
    for i in range(0, len(feats), chunk):
        d = torch.cdist(feats[i:i + chunk], feats)
        radii.append(d.kthvalue(k + 1, dim=1).values)  # +1 skips the zero self-distance
    return torch.cat(radii)


def _covered(query: torch.Tensor, ref: torch.Tensor, ref_radii: torch.Tensor, chunk: int) -> torch.Tensor:
    """Fraction of `query` points inside at least one k-NN ball of the `ref` manifold."""
    inside = []
    for i in range(0, len(query), chunk):
        d = torch.cdist(query[i:i + chunk], ref)
        inside.append((d <= ref_radii[None]).any(dim=1))
    return torch.cat(inside).float().mean()


@torch.no_grad()
def precision_recall(pool_real, pool_fake, k: int = 3, device="cuda", chunk: int = 1000):
    """Improved precision (realism) and recall (coverage), Kynkaenniemi et al. (2019), k = 3."""
    real, fake = pool_real.to(device).float(), pool_fake.to(device).float()
    r_real, r_fake = _kth_nn_radius(real, k, chunk), _kth_nn_radius(fake, k, chunk)
    precision = _covered(fake, real, r_real, chunk).item()
    recall = _covered(real, fake, r_fake, chunk).item()
    return precision, recall


@torch.no_grad()
def nearest_neighbours(query, ref, device="cuda", chunk: int = 1000):
    """L2 nearest neighbour of every query feature among the reference features -> (dist, idx)."""
    ref = ref.to(device).float()
    dists, idxs = [], []
    for i in range(0, len(query), chunk):
        d = torch.cdist(query[i:i + chunk].to(device).float(), ref)
        v, j = d.min(dim=1)
        dists.append(v.cpu())
        idxs.append(j.cpu())
    return torch.cat(dists), torch.cat(idxs)


def memorisation_stats(nn_dist_gen: torch.Tensor, nn_dist_test: torch.Tensor):
    """Compare sample->train NN distances with real-test->train distances (the no-copying baseline).

    If a model copied training images, many samples would sit closer to the training set than almost
    any unseen real image does; `frac_below_test_p1` is ~0.01 for a model that does not copy.
    """
    p1 = torch.quantile(nn_dist_test, 0.01)
    return {"nn_dist_median": nn_dist_gen.median().item(),
            "nn_dist_median_ratio_to_test": (nn_dist_gen.median() / nn_dist_test.median()).item(),
            "frac_below_test_p1": (nn_dist_gen < p1).float().mean().item()}


def psnr(x_uint8: torch.Tensor, y_uint8: torch.Tensor) -> torch.Tensor:
    mse = ((x_uint8.float() - y_uint8.float()) ** 2).flatten(1).mean(1)
    return 10 * torch.log10(255.0 ** 2 / mse.clamp(min=1e-10))


def ssim(x_uint8: torch.Tensor, y_uint8: torch.Tensor, window: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Per-image SSIM (Wang et al., 2004) with a Gaussian window, averaged over channels."""
    x, y = x_uint8.float(), y_uint8.float()
    C = x.shape[1]
    g = torch.exp(-((torch.arange(window) - window // 2) ** 2) / (2 * sigma ** 2))
    g = (g / g.sum())
    kernel = (g[:, None] * g[None, :]).expand(C, 1, window, window).contiguous()

    def blur(t):
        return F.conv2d(t, kernel, groups=C)

    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mx, my = blur(x), blur(y)
    vx, vy, cxy = blur(x * x) - mx ** 2, blur(y * y) - my ** 2, blur(x * y) - mx * my
    s = ((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (vx + vy + c2))
    return s.flatten(1).mean(1)


def class_statistics(probs: torch.Tensor, num_classes: int = 10):
    """Thematic consistency from a CIFAR-10 classifier's predictions on generated images.

    confidence      mean max-probability: are samples recognisable as *some* CIFAR-10 class?
    class_entropy   normalised entropy of the predicted-class histogram: 1 = all classes equally covered
    min_class_frac  share of the rarest class (0.1 = perfectly balanced); exposes missing classes
    """
    conf, pred = probs.max(dim=1)
    hist = torch.bincount(pred, minlength=num_classes).float() / len(pred)
    nz = hist[hist > 0]
    return {"confidence": conf.mean().item(),
            "frac_confident_0.9": (conf > 0.9).float().mean().item(),
            "class_entropy": (-(nz * nz.log()).sum() / math.log(num_classes)).item(),
            "min_class_frac": hist.min().item(),
            "class_hist": hist.tolist()}
