# Repository file: scripts/make_figures.py
"""Turn the cached evaluation results into result tables (results.md, results.csv) and figures.

    python scripts/make_figures.py --eval-dir runs/eval --out runs/eval/report
    python scripts/make_figures.py ... --model-figures   # also figures that need the trained models
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from genmodels.data import load_cifar10
from genmodels.eval.classifier import CLASSES
from genmodels.eval.sampling import autocast, load_run, load_samples
from genmodels.utils import to_uint8

MODELS = [  # (display name, sample-set key prefix, colour)
    ("VAE (paper MLP)", "vae_mlp", "#9e9e9e"),
    ("VAE (conv)", "vae_conv", "#1f77b4"),
    ("DDPM", "ddpm_ancestral1000", "#d62728"),
]


def grid(images, nrow, pad=1):
    """uint8 [N, 3, H, W] -> HWC array tiled nrow per row, white padding."""
    images = np.asarray(images)
    n, c, h, w = images.shape
    rows = int(np.ceil(n / nrow))
    out = np.full((rows * (h + pad) + pad, nrow * (w + pad) + pad, c), 255, np.uint8)
    for i, im in enumerate(images):
        r, k = divmod(i, nrow)
        out[pad + r * (h + pad):pad + r * (h + pad) + h, pad + k * (w + pad):pad + k * (w + pad) + w] = im.transpose(1, 2, 0)
    return out


def show(ax, img, title=None):
    ax.imshow(img, interpolation="nearest")
    ax.set_xticks([]), ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10)


class Report:
    def __init__(self, eval_dir, out_dir, data_root):
        self.ed, self.out = Path(eval_dir), Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.items = {p.stem: json.loads(p.read_text()) for p in (self.ed / "items").glob("*.json")}
        self.bpd = json.loads((self.ed / "bpd.json").read_text()) if (self.ed / "bpd.json").exists() else {}
        self.timing = json.loads((self.ed / "timing.json").read_text()) if (self.ed / "timing.json").exists() else {}
        self.runs = json.loads((self.ed / "runs.json").read_text()) if (self.ed / "runs.json").exists() else {}
        self.curves = [json.loads(p.read_text()) for p in sorted((self.ed / "curves").glob("*.json"))]
        self.data_root = data_root
        self._train = None

    @property
    def train_x(self):
        if self._train is None:
            self._train = load_cifar10(self.data_root, train=True)[0]
        return self._train

    def samples(self, key):
        p = self.ed / "samples" / f"{key}.npz"
        return load_samples(p).numpy() if p.exists() else None

    def per_sample(self, key):
        p = self.ed / "items" / f"{key}_per_sample.npz"
        return np.load(p) if p.exists() else None

    def save(self, fig, name):
        fig.savefig(self.out / name, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("wrote", self.out / name)

    # ---- aggregation over seeds -----------------------------------------------------------------
    def agg(self, prefix, metric):
        vals = [v[metric] for k, v in self.items.items() if (k == prefix or k.startswith(prefix + "_s")) and metric in v]
        if not vals:
            return None, None
        return float(np.mean(vals)), (float(np.std(vals, ddof=1)) if len(vals) > 1 else None)

    def fmt(self, prefix, metric, digits=2, scale=1.0):
        m, s = self.agg(prefix, metric)
        if m is None:
            return "–"
        return f"{m * scale:.{digits}f}" + (f" ± {s * scale:.{digits}f}" if s is not None else "")

    # ---- tables -----------------------------------------------------------------------------------
    def tables(self):
        L = ["# Results\n", "All sample-based metrics use 10,000 samples against the 50,000 CIFAR-10 training images "
             "(FID-10k: systematically a little higher than the FID-50k reported in papers). "
             "± is the standard deviation over 3 independent sample sets (evaluation noise, not training-seed variance).\n"]
        fam = {"vae_mlp": "mlp", "vae_conv": "vae", "ddpm_ancestral1000": "ddpm", "ddpm_ddim100": "ddpm"}
        tkey = {"vae_mlp": "mlp_prior", "vae_conv": "vae_prior", "ddpm_ancestral1000": "ddpm_ddpm1000",
                "ddpm_ddim100": "ddpm_ddim100"}
        L += ["## Main comparison\n",
              "| Model | FID ↓ | IS ↑ | KID ×10³ ↓ | Precision ↑ | Recall ↑ | NLL bound (bits/dim) ↓ | NFE / image | Images/s ↑ | Params | Train GPU-h |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        real = self.items.get("real_test")
        if real:
            L.append(f"| *Real test images (reference)* | {real['fid']:.2f} | {real['is_mean']:.2f} | "
                     f"{real['kid_mean'] * 1e3:.2f} | {real['precision']:.3f} | {real['recall']:.3f} | – | – | – | – | – |")
        rows = [("VAE — paper MLP", "vae_mlp"), ("VAE — conv", "vae_conv"),
                ("DDPM — 1000 ancestral steps (paper sampler)", "ddpm_ancestral1000"), ("DDPM — 100 DDIM steps", "ddpm_ddim100")]
        for name, prefix in rows:
            if self.agg(prefix, "fid")[0] is None:
                continue
            f = fam[prefix]
            b = self.bpd.get(f, {})
            b = b.get("fixedlarge", b)
            bpd = f"{b['bpd']:.3f}" if "bpd" in b else "–"
            t = self.timing.get(tkey[prefix], {})
            r = self.runs.get(f, {})
            nfe = {"vae_mlp": 1, "vae_conv": 1, "ddpm_ancestral1000": 1000, "ddpm_ddim100": 100}[prefix]
            L.append(f"| {name} | {self.fmt(prefix, 'fid')} | {self.fmt(prefix, 'is_mean')} | {self.fmt(prefix, 'kid_mean', 2, 1e3)} | "
                     f"{self.fmt(prefix, 'precision', 3)} | {self.fmt(prefix, 'recall', 3)} | {bpd} | {nfe} | "
                     f"{t.get('throughput_img_s', float('nan')):,.1f} | {r.get('n_params', 0) / 1e6:.1f}M | "
                     f"{r.get('train_gpu_hours', float('nan')):.2f} |")
        if self.timing.get("gpu"):
            L.append(f"\nSpeed measured on {self.timing['gpu']} (batch 256, bf16). Paper reference (DDPM, FID-50k): "
                     "FID 3.17, IS 9.46, NLL ≤ 3.75 bits/dim after 800k steps.\n")

        L += ["## DDPM: sampling steps vs. quality (same network, same starting noise)\n",
              "| Sampler | NFE | FID ↓ | IS ↑ | Precision | Recall | Images/s (batch 256) | Latency, 1 image |",
              "|---|---|---|---|---|---|---|---|"]
        for key, tk, name in [("ddpm_ancestral1000_s0", "ddpm_ddpm1000", "Ancestral (paper)")] + \
                [(f"ddpm_ddim{k}_s0", f"ddpm_ddim{k}", "DDIM") for k in (10, 25, 50, 100, 250)]:
            if key in self.items:
                v, t = self.items[key], self.timing.get(tk, {})
                L.append(f"| {name} | {v['nfe']} | {v['fid']:.2f} | {v['is_mean']:.2f} | {v['precision']:.3f} | "
                         f"{v['recall']:.3f} | {t.get('throughput_img_s', float('nan')):,.1f} | {t.get('latency_s', float('nan')):.2f} s |")
        for k, name in [("vae_prior", "VAE (conv)"), ("mlp_prior", "VAE (MLP)")]:
            t = self.timing.get(k)
            if t:
                L.append(f"| {name}, for comparison | 1 | | | | | {t['throughput_img_s']:,.1f} | {t['latency_s'] * 1000:.1f} ms |")

        L += ["\n## Ablations that need no extra training\n", "| Comparison | FID ↓ | IS ↑ | Precision | Recall |", "|---|---|---|---|---|"]
        for key, name in [("ddpm_ddim100_s0", "DDPM, DDIM-100, EMA weights"), ("ddpm_raw_ddim100_s0", "DDPM, DDIM-100, raw weights"),
                          ("vae_conv_s0", "VAE (conv), EMA weights"), ("vae_conv_raw_s0", "VAE (conv), raw weights"),
                          ("vae_conv_recon", "VAE (conv) reconstructions of test images"), ("vae_mlp_recon", "VAE (MLP) reconstructions")]:
            if key in self.items:
                v = self.items[key]
                L.append(f"| {name} | {v['fid']:.2f} | {v['is_mean']:.2f} | {v['precision']:.3f} | {v['recall']:.3f} |")
        for key, name in [("vae_conv_recon", "VAE (conv)"), ("vae_mlp_recon", "VAE (MLP)")]:
            if key in self.items:
                v = self.items[key]
                L.append(f"\n{name} reconstruction quality on the test set: PSNR {v['psnr']:.2f} dB, SSIM {v['ssim']:.3f}.")

        L += ["\n## Thematic consistency (CIFAR-10 classifier on generated images)\n",
              "| Images | Mean confidence ↑ | Share with confidence > 0.9 ↑ | Class-histogram entropy (1 = all classes equal) | Rarest class share (0.10 = balanced) |",
              "|---|---|---|---|---|"]
        for key, name in [("real_test", "Real test images"), ("vae_mlp_s0", "VAE (MLP)"), ("vae_conv_s0", "VAE (conv)"),
                          ("ddpm_ancestral1000_s0", "DDPM (1000 steps)"), ("ddpm_ddim100_s0", "DDPM (DDIM 100)")]:
            v = self.items.get(key, {})
            if "confidence" in v:
                L.append(f"| {name} | {v['confidence']:.3f} | {v['frac_confident_0.9']:.3f} | {v['class_entropy']:.3f} | {v['min_class_frac']:.3f} |")
        if real and "classifier_accuracy" in real:
            L.append(f"\nClassifier accuracy on the real test set: {real['classifier_accuracy']:.4f}.")

        L += ["\n## Memorisation check\n",
              "Distance from each sample to its nearest training image (Inception features), relative to the same distance "
              "for unseen real test images. A copying model would have many samples closer than 99% of real test images "
              "(last column ≫ 0.01).\n",
              "| Model | Median NN distance / real-test median | Share closer than the real-test 1st percentile |", "|---|---|---|"]
        for key, name in [("vae_mlp_s0", "VAE (MLP)"), ("vae_conv_s0", "VAE (conv)"), ("ddpm_ancestral1000_s0", "DDPM")]:
            if key in self.items:
                v = self.items[key]
                L.append(f"| {name} | {v['nn_dist_median_ratio_to_test']:.3f} | {v['frac_below_test_p1']:.4f} |")

        if self.bpd:
            L += ["\n## Likelihood (upper bounds on the test-set NLL, bits/dim)\n", "| Model | bits/dim ↓ |", "|---|---|"]
            for f, name in [("mlp", "VAE (MLP), ELBO"), ("vae", "VAE (conv), ELBO")]:
                if f in self.bpd:
                    L.append(f"| {name} | {self.bpd[f]['bpd']:.3f} ± {self.bpd[f]['bpd_sem']:.3f} |")
            if "ddpm" in self.bpd:
                for v, name in [("fixedlarge", "σ²=β"), ("fixedsmall", "σ²=β̃")]:
                    d = self.bpd["ddpm"][v]
                    L.append(f"| DDPM, variational bound, {name} ({d['n']} test images) | {d['bpd']:.3f} ± {d['bpd_sem']:.3f} |")

        blind = self.ed / "blind" / "score.json"
        if blind.exists():
            s = json.loads(blind.read_text())
            L += [f"\n## Blind rating (rater: {s.get('rater', 'human')}, {s.get('n', '?')} images)\n",
                  "| Source | Mean realism (1–5) | Named a class | Judged 'real' |", "|---|---|---|---|"]
            for src, v in s["by_source"].items():
                L.append(f"| {src} | {v['realism']:.2f} | {v['recognised']:.2f} | {v['judged_real']:.2f} |")
            if "real_vs_generated_accuracy" in s:
                L.append(f"\nReal-vs-generated accuracy: {s['real_vs_generated_accuracy']:.2f}.")

        (self.out / "results.md").write_text("\n".join(L) + "\n", encoding="utf-8")
        with open(self.out / "results.csv", "w", newline="") as f:
            keys = sorted({k for v in self.items.values() for k, x in v.items() if not isinstance(x, (list, dict))})
            w = csv.DictWriter(f, ["key"] + keys)
            w.writeheader()
            for k, v in sorted(self.items.items()):
                w.writerow({"key": k, **{a: b for a, b in v.items() if a in keys}})
        print("wrote", self.out / "results.md")

    # ---- figures from cached samples ---------------------------------------------------------------
    def fig_samples(self):
        panels = [("Real CIFAR-10", None)] + [(n, k) for n, k, _ in MODELS]
        panels = [(n, k) for n, k in panels if k is None or self.samples(f"{k}_s0") is not None]
        fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.6))
        for ax, (name, key) in zip(np.atleast_1d(axes), panels):
            imgs = load_cifar10(self.data_root, train=False)[0][:100].numpy() if key is None else self.samples(f"{key}_s0")[:100]
            show(ax, grid(imgs, 10), name)
        fig.suptitle("Uncurated samples (first 100, fixed seed)")
        self.save(fig, "fig_samples.png")

    def fig_speed_quality(self):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
        pts = [(self.items[k]["nfe"], self.items[k]["fid"], k) for k in self.items if k.startswith("ddpm_ddim") and k.endswith("_s0")]
        pts.sort()
        if pts:
            axes[0].plot([p[0] for p in pts], [p[1] for p in pts], "o-", color="#d62728", label="DDPM, DDIM")
        if "ddpm_ancestral1000_s0" in self.items:
            axes[0].plot(1000, self.items["ddpm_ancestral1000_s0"]["fid"], "*", ms=14, color="#d62728", label="DDPM, ancestral (paper)")
        for name, key, c in MODELS[:2]:
            if f"{key}_s0" in self.items:
                axes[0].plot(1, self.items[f"{key}_s0"]["fid"], "s", ms=9, color=c, label=name)
        if "real_test" in self.items:
            axes[0].axhline(self.items["real_test"]["fid"], ls="--", c="k", lw=1, label="real test images")
        axes[0].set(xscale="log", yscale="log", xlabel="network evaluations per image (NFE)", ylabel="FID-10k ↓",
                    title="Quality vs. sampling cost")
        axes[0].legend(fontsize=8)
        tk = [(f"ddpm_ddim{k}_s0", f"ddpm_ddim{k}") for k in (10, 25, 50, 100, 250)] + [("ddpm_ancestral1000_s0", "ddpm_ddpm1000")]
        xs = [(self.timing[t]["throughput_img_s"], self.items[k]["fid"]) for k, t in tk if k in self.items and t in self.timing]
        if xs:
            axes[1].plot(*zip(*sorted(xs)), "o-", color="#d62728", label="DDPM")
        for (name, key, c), t in zip(MODELS[:2], ("mlp_prior", "vae_prior")):
            if f"{key}_s0" in self.items and t in self.timing:
                axes[1].plot(self.timing[t]["throughput_img_s"], self.items[f"{key}_s0"]["fid"], "s", ms=9, color=c, label=name)
        axes[1].set(xscale="log", yscale="log", xlabel=f"images per second ({self.timing.get('gpu', 'GPU')}, batch 256)",
                    ylabel="FID-10k ↓", title="Quality vs. measured speed")
        axes[1].legend(fontsize=8)
        self.save(fig, "fig_speed_quality.png")

    def fig_precision_recall(self):
        fig, ax = plt.subplots(figsize=(5.5, 5))
        for key, name, c, m in [("real_test", "real test", "k", "*"), ("vae_mlp_s0", "VAE (MLP)", "#9e9e9e", "s"),
                                ("vae_conv_s0", "VAE (conv)", "#1f77b4", "s"), ("ddpm_ancestral1000_s0", "DDPM (1000)", "#d62728", "o"),
                                ("ddpm_ddim10_s0", "DDPM (DDIM 10)", "#ff9896", "o"), ("vae_conv_recon", "VAE recon.", "#aec7e8", "^")]:
            if key in self.items:
                v = self.items[key]
                ax.scatter(v["recall"], v["precision"], c=c, marker=m, s=90, label=name)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="recall (diversity / coverage) →", ylabel="precision (realism) →",
               title="Quality vs. diversity")
        ax.legend(fontsize=8, loc="lower left")
        self.save(fig, "fig_precision_recall.png")

    def fig_training_curves(self):
        if not self.curves:
            return
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for fam, name, c in [("ddpm", "DDPM (DDIM-50)", "#d62728"), ("vae", "VAE (conv)", "#1f77b4")]:
            pts = sorted((r["step"], r["fid"]) for r in self.curves if r["family"] == fam)
            if pts:
                ax.plot(*zip(*pts), "o-", c=c, label=name)
        ax.set(yscale="log", xlabel="training step", ylabel=f"FID-{self.curves[0]['n'] // 1000}k (EMA weights) ↓",
               title="Sample quality during training")
        ax.legend()
        self.save(fig, "fig_fid_during_training.png")

    def fig_classes(self):
        rows = [(k, n, c) for k, n, c in [("real_test", "real test", "k"), ("vae_mlp_s0", "VAE (MLP)", "#9e9e9e"),
                                          ("vae_conv_s0", "VAE (conv)", "#1f77b4"), ("ddpm_ancestral1000_s0", "DDPM", "#d62728")]
                if "class_hist" in self.items.get(k, {})]
        if not rows:
            return
        fig, ax = plt.subplots(figsize=(11, 3.8))
        w = 0.8 / len(rows)
        for i, (k, n, c) in enumerate(rows):
            ax.bar(np.arange(10) + i * w, self.items[k]["class_hist"], w, color=c, label=n)
        ax.axhline(0.1, ls="--", c="k", lw=1)
        ax.set_xticks(np.arange(10) + 0.4 - w / 2, CLASSES)
        ax.set(ylabel="share of images", title="Predicted-class distribution (thematic coverage)")
        ax.legend(fontsize=8)
        self.save(fig, "fig_class_distribution.png")

    def fig_by_class(self):
        for key, name in [("ddpm_ancestral1000_s0", "DDPM"), ("vae_conv_s0", "VAE (conv)")]:
            ps, imgs = self.per_sample(key), self.samples(key)
            if ps is None or imgs is None or "pred" not in ps:
                continue
            rows = []
            for c in range(10):
                idx = np.where(ps["pred"] == c)[0]
                idx = idx[np.argsort(-ps["confidence"][idx])][:10]
                block = np.full((10, 3, 32, 32), 255, np.uint8)
                block[:len(idx)] = imgs[idx]
                rows.append(block)
            fig, ax = plt.subplots(figsize=(5.5, 6))
            show(ax, grid(np.concatenate(rows), 10), f"{name}: most confident samples per predicted class")
            ax.set_yticks(np.arange(10) * 33 + 17, CLASSES, fontsize=8)
            self.save(fig, f"fig_by_class_{key.split('_')[0]}_{key.split('_')[1]}.png")

    def fig_failures(self):
        panels = []
        for key, name in [("vae_conv_s0", "VAE (conv)"), ("ddpm_ancestral1000_s0", "DDPM")]:
            ps, imgs = self.per_sample(key), self.samples(key)
            if ps is not None and imgs is not None and "confidence" in ps:
                panels.append((name, imgs[np.argsort(ps["confidence"])[:32]]))
        if not panels:
            return
        fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 3.4))
        for ax, (name, imgs) in zip(np.atleast_1d(axes), panels):
            show(ax, grid(imgs, 8), f"{name}: 32 least recognisable samples")
        self.save(fig, "fig_failures.png")

    def fig_nearest_neighbours(self):
        panels = []
        for key, name in [("vae_conv_s0", "VAE (conv)"), ("ddpm_ancestral1000_s0", "DDPM")]:
            ps, imgs = self.per_sample(key), self.samples(key)
            if ps is None or imgs is None:
                continue
            order = np.argsort(ps["nn_dist"])[:8]
            pairs = np.concatenate([imgs[order], self.train_x[torch.from_numpy(ps["nn_idx"][order])].numpy()])
            panels.append((name, pairs))
        if not panels:
            return
        fig, axes = plt.subplots(len(panels), 1, figsize=(7, 2.2 * len(panels)))
        for ax, (name, pairs) in zip(np.atleast_1d(axes), panels):
            show(ax, grid(pairs, 8), f"{name}: the 8 samples closest to the training set (top) and their nearest training image (bottom)")
        self.save(fig, "fig_nearest_neighbours.png")

    def fig_reconstructions(self):
        rows = [load_cifar10(self.data_root, train=False)[0][:16].numpy()]
        names = ["test images"]
        for key, name in [("vae_conv_recon", "VAE (conv)"), ("vae_mlp_recon", "VAE (MLP)")]:
            s = self.samples(key)
            if s is not None:
                rows.append(s[:16])
                names.append(name)
        if len(rows) == 1:
            return
        fig, ax = plt.subplots(figsize=(9, 1.1 * len(rows) + 0.6))
        show(ax, grid(np.concatenate(rows), 16), "VAE reconstructions (rows: " + ", ".join(names) + ")")
        self.save(fig, "fig_vae_reconstructions.png")

    def fig_ddim_steps(self):
        cols = [(f"ddpm_ddim{k}_s0", f"DDIM {k}") for k in (10, 25, 50, 100, 250)] + [("ddpm_ancestral1000_s0", "ancestral 1000")]
        cols = [(k, n) for k, n in cols if self.samples(k) is not None]
        if not cols:
            return
        imgs = np.stack([self.samples(k)[:6] for k, _ in cols], axis=1).reshape(-1, 3, 32, 32)
        fig, ax = plt.subplots(figsize=(1.1 * len(cols) + 1, 7.5))
        show(ax, grid(imgs, len(cols)), "Same starting noise x_T, different numbers of steps")
        ax.set_xticks(np.arange(len(cols)) * 33 + 17, [n for _, n in cols], rotation=45, fontsize=8)
        self.save(fig, "fig_ddim_steps.png")

    def fig_bits_per_timestep(self):
        d = self.bpd.get("ddpm", {}).get("fixedlarge")
        if not d:
            return
        terms = np.array(d["terms_mean"])
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.plot(np.arange(len(terms)), terms, c="#d62728")
        ax.set(yscale="log", xlabel="timestep t", ylabel="bits/dim of the term L_{t-1}",
               title=f"Where the DDPM's {d['bpd']:.2f} bits/dim go")
        self.save(fig, "fig_bits_per_timestep.png")

    # ---- figures that need the trained models ---------------------------------------------------
    @torch.no_grad()
    def fig_denoising(self, ddpm_run, device="cuda"):
        model, _ = load_run(ddpm_run, device=device)
        g = torch.Generator(device).manual_seed(7)
        record = (999, 800, 600, 400, 300, 200, 100, 50, 0)
        with autocast(device):
            _, traj = model.diffusion.p_sample_loop(model.net, (4, 3, 32, 32), g, device, record_steps=record)
        rows = []
        for i in range(4):
            rows.append(np.stack([to_uint8(traj[t][0][i]).numpy() for t in record]))
            rows.append(np.stack([to_uint8(traj[t][1][i]).numpy() for t in record]))
        fig, ax = plt.subplots(figsize=(9, 8.5))
        show(ax, grid(np.concatenate(rows), len(record)), "DDPM reverse process: x_t (odd rows) and predicted x_0 (even rows)")
        ax.set_xticks(np.arange(len(record)) * 33 + 17, [f"t={t}" for t in record], fontsize=8)
        self.save(fig, "fig_ddpm_denoising.png")

    @torch.no_grad()
    def fig_interpolation(self, ddpm_run, vae_run, device="cuda"):
        alphas = torch.linspace(0, 1, 9, device=device)

        def slerp(a, b, t):
            omega = torch.acos(((a / a.norm()) * (b / b.norm())).sum().clamp(-1, 1))
            return (torch.sin((1 - t) * omega) * a + torch.sin(t * omega) * b) / torch.sin(omega)

        rows, labels = [], []
        test_x = load_cifar10(self.data_root, train=False)[0]
        if vae_run:
            vae, _ = load_run(vae_run, device=device)
            for i, j in [(0, 1), (2, 3), (4, 5)]:
                mu, _ = vae.encode(vae.bin_centres(test_x[[i, j]].to(device)))
                z = torch.stack([(1 - a) * mu[0] + a * mu[1] for a in alphas])
                rows.append(to_uint8(vae.decode(z)[0]).cpu().numpy())
                labels.append("VAE: latent interpolation between two test images")
        if ddpm_run:
            ddpm, _ = load_run(ddpm_run, device=device)
            g = torch.Generator(device).manual_seed(3)
            for _ in range(3):
                a, b = torch.randn((2, 3, 32, 32), device=device, generator=g)
                xT = torch.stack([slerp(a, b, t) for t in alphas])
                with autocast(device):
                    out = ddpm.diffusion.ddim_sample_loop(ddpm.net, xT.shape, 50, device=device, x_T=xT)
                rows.append(to_uint8(out.float()).cpu().numpy())
                labels.append("DDPM: spherical interpolation of the starting noise x_T (DDIM-50)")
        if not rows:
            return
        fig, ax = plt.subplots(figsize=(8, 1.0 * len(rows) + 0.8))
        show(ax, grid(np.concatenate(rows), 9), "Interpolations — rows 1–3: VAE latent space; rows 4–6: DDPM noise space")
        self.save(fig, "fig_interpolation.png")


def samples_grid(src, dst, pad=14):
    """Re-lay the 1x4 uncurated-samples figure out as a 2x2 grid (better for portrait pages).

    Panels are found as the runs of non-white columns; the figure's suptitle (first row of ink) is dropped."""
    from PIL import Image
    if not Path(src).exists():
        return
    im = Image.open(src).convert("RGB")
    ink = np.asarray(im).min(axis=2) < 240
    def runs(mask):
        edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
        return list(zip(edges[::2], edges[1::2]))
    cols = [(a, b) for a, b in runs(ink.any(axis=0)) if b - a > 50]
    top = runs(ink.any(axis=1))[1][0] - 6
    panels = [im.crop((max(a - 8, 0), top, min(b + 8, im.width), im.height)) for a, b in cols]
    w, h = panels[0].size
    out = Image.new("RGB", (2 * w + pad, 2 * h + pad), "white")
    for k, p in enumerate(panels[:4]):
        out.paste(p, ((k % 2) * (w + pad), (k // 2) * (h + pad)))
    out.save(dst)
    print("wrote", dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="runs/eval")
    ap.add_argument("--out", default=None, help="default: <eval-dir>/report")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--model-figures", action="store_true", help="also denoising / interpolation figures (needs GPU)")
    ap.add_argument("--ddpm", default="runs/ddpm_cifar10")
    ap.add_argument("--vae", default="runs/vae_conv_cifar10")
    args = ap.parse_args()
    r = Report(args.eval_dir, args.out or Path(args.eval_dir) / "report", args.data_root)
    r.tables()
    for f in (r.fig_samples, lambda: samples_grid(r.out / "fig_samples.png", r.out / "fig_samples_grid.png"), r.fig_speed_quality, r.fig_precision_recall, r.fig_training_curves, r.fig_classes,
              r.fig_by_class, r.fig_failures, r.fig_nearest_neighbours, r.fig_reconstructions, r.fig_ddim_steps,
              r.fig_bits_per_timestep):
        f()
    if args.model_figures:
        ddpm = args.ddpm if Path(args.ddpm, "ema_final.pt").exists() else None
        vae = args.vae if Path(args.vae, "ema_final.pt").exists() else None
        if ddpm:
            r.fig_denoising(ddpm)
        r.fig_interpolation(ddpm, vae)


if __name__ == "__main__":
    main()
