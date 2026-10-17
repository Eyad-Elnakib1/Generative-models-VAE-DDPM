# Repository file: scripts/plot_training_logs.py
"""Training curves from the runs' metrics.jsonl logs (no GPU, no samples needed).

    python scripts/plot_training_logs.py --runs results/training --out results/figures/fig_training_curves.png
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    rows = [json.loads(line) for line in open(path)]
    train = [r for r in rows if "loss" in r]
    evals = [r for r in rows if any(k.startswith("ema_") for k in r)]
    return train, evals


def smooth(y, k=10):
    y = np.asarray(y, float)
    return np.convolve(y, np.ones(k) / k, mode="valid") if len(y) > k else y


def to_bpd(loss_per_dim):
    """VAE training loss (nats per dim of the dequantised density on [-1, 1]) -> bits/dim."""
    return (np.asarray(loss_per_dim) + math.log(128.0)) / math.log(2.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/training")
    ap.add_argument("--out", default="results/figures/fig_training_curves.png")
    args = ap.parse_args()
    runs = Path(args.runs)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))

    train, evals = read(runs / "ddpm_cifar10" / "metrics.jsonl")
    s = [r["step"] for r in train]
    axes[0].plot(s[9:], smooth([r["loss"] for r in train]), c="#d62728", lw=1, label="training loss (smoothed)")
    axes[0].plot([r["step"] for r in evals], [r["raw_test_loss"] for r in evals], "o-", ms=3, c="k", label="test loss, raw weights")
    axes[0].plot([r["step"] for r in evals], [r["ema_test_loss"] for r in evals], "s-", ms=3, c="#ff9896", label="test loss, EMA weights")
    axes[0].set(yscale="log", ylim=(0.025, 0.06), xlabel="training step", ylabel="L_simple (noise-prediction MSE)",
                title="DDPM: train ≈ test (no overfitting)")
    axes[0].legend(fontsize=8)

    for ax, name, title in [(axes[1], "vae_conv_cifar10", "VAE (conv)"), (axes[2], "vae_mlp_cifar10", "VAE (paper MLP)")]:
        train, evals = read(runs / name / "metrics.jsonl")
        s = [r["step"] for r in train]
        ax.plot(s[9:], smooth(to_bpd([r["loss"] for r in train])), c="#1f77b4", lw=1, label="training images (smoothed)")
        ax.plot([r["step"] for r in evals], [r["raw_test_bpd"] for r in evals], "o-", ms=3, c="k", label="test images, raw weights")
        ax.plot([r["step"] for r in evals], [r["ema_test_bpd"] for r in evals], "s-", ms=3, c="#aec7e8", label="test images, EMA weights")
        best = min(evals, key=lambda r: r["raw_test_bpd"])
        ax.axvline(best["step"], ls=":", c="gray", lw=1)
        ax.set(xlabel="training step", ylabel="negative ELBO (bits/dim)",
               title=f"{title}: best raw test bound at step {best['step']:,}")
        lo = min(to_bpd([r["loss"] for r in train[len(train) // 10:]]).min(), min(r["raw_test_bpd"] for r in evals))
        ax.set_ylim(lo - 0.1, lo + 1.2)
        ax.legend(fontsize=8)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
