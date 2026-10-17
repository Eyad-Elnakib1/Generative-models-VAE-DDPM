# Repository file: scripts/train_classifier.py
"""Train the CIFAR-10 classifier used to measure thematic consistency (~10 min on an L4).

    python scripts/train_classifier.py --out runs/classifier
"""
import argparse
from pathlib import Path

import torch

from genmodels.data import load_cifar10
from genmodels.eval.classifier import train_classifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/classifier")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    path = Path(args.out) / "classifier.pt"
    if path.exists():
        print(f"{path} exists (test accuracy {torch.load(path, map_location='cpu')['test_accuracy']:.4f}); skipping")
        return
    train_x, train_y = load_cifar10(args.data_root, train=True)
    test_x, test_y = load_cifar10(args.data_root, train=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, acc = train_classifier(train_x, train_y, test_x, test_y, epochs=args.epochs, device=device)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "test_accuracy": acc, "epochs": args.epochs}, path)
    print(f"saved {path}: test accuracy {acc:.4f}")


if __name__ == "__main__":
    main()
