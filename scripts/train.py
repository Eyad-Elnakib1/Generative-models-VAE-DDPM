# Repository file: scripts/train.py
"""Train a model from a config. Re-running the same command resumes from `<run-dir>/ckpt_last.pt`.

    python scripts/train.py --config configs/ddpm_cifar10.yaml --run-dir runs/ddpm_cifar10
    python scripts/train.py --config configs/vae_conv_cifar10.yaml --set train.total_steps=50000
"""
import argparse
from pathlib import Path

from genmodels.data import load_cifar10
from genmodels.trainer import train
from genmodels.utils import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", default=None, help="default: runs/<config name>")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. train.total_steps=1000")
    ap.add_argument("--max-steps", type=int, default=None, help="stop early (smoke tests); total_steps is unchanged")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    run_dir = Path(args.run_dir or Path("runs") / cfg["name"])
    train_x, _ = load_cifar10(args.data_root, train=True)
    test_x, _ = load_cifar10(args.data_root, train=False)
    train(cfg, run_dir, train_x, test_x, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
