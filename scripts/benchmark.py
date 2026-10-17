# Repository file: scripts/benchmark.py
"""Phase-0 calibration: measure training speed and sampling throughput on this GPU and project
the GPU-hours / Colab compute units of the full plan. Uses random data (no dataset download).

    python scripts/benchmark.py --units-per-hour 1.7
"""
import argparse
import time

import torch

from genmodels.data import GPUBatcher
from genmodels.models import build_model, network
from genmodels.trainer import make_optimizer
from genmodels.utils import autocast_dtype, load_config

DEFAULT_CONFIGS = ["configs/ddpm_cifar10.yaml", "configs/vae_conv_cifar10.yaml"]
EVAL_NFE = 22e6  # network evaluations needed by the DDPM evaluation protocol (README, "Evaluation")


def bench_train(cfg, device, steps, warmup, compile_):
    tcfg = cfg["train"]
    model = build_model(cfg["model"]).to(device)
    if compile_:
        network(model).compile()
    opt = make_optimizer(model, tcfg["optim"], 50000, 3072)
    data = torch.randint(0, 256, (4096, 3, 32, 32), dtype=torch.uint8)
    batcher = GPUBatcher(data, tcfg["batch_size"], device=device)
    dtype = autocast_dtype(tcfg.get("precision", "auto"), device)
    torch.cuda.reset_peak_memory_stats(device)
    for i in range(warmup + steps):
        if i == warmup:
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=dtype or torch.float32, enabled=dtype is not None):
            loss = model(batcher.batch(i))["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    torch.cuda.synchronize(device)
    its = steps / (time.perf_counter() - t0)
    return model, dtype, its, torch.cuda.max_memory_allocated(device) / 1e9


@torch.no_grad()
def bench_nfe(model, dtype, device, batch, reps=10):
    """Network evaluations (one image through the U-Net once) per second."""
    net = network(model).eval()
    x = torch.randn(batch, 3, 32, 32, device=device)
    t = torch.randint(0, 1000, (batch,), device=device)
    with torch.autocast("cuda", dtype=dtype or torch.float32, enabled=dtype is not None):
        for _ in range(3):
            net(x, t)
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for _ in range(reps):
            net(x, t)
        torch.cuda.synchronize(device)
    return batch * reps / (time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--sample-batch", type=int, default=500)
    ap.add_argument("--units-per-hour", type=float, default=None,
                    help="the rate Colab shows for this GPU (Runtime > View resources)")
    ap.add_argument("--batch-sizes", type=int, nargs="*", default=None,
                    help="only measure training throughput (images/s) of the first config at these batch sizes")
    args = ap.parse_args()

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    print(f"GPU: {torch.cuda.get_device_name(device)} | compile={args.compile}\n")
    if args.batch_sizes:
        # Training speed is images/s, not steps/s: a bigger batch only helps if images/s rises.
        base = None
        for bs in args.batch_sizes:
            cfg = load_config(args.configs[0], [f"train.batch_size={bs}"])
            model, _, its, mem = bench_train(cfg, device, args.steps, args.warmup, args.compile)
            base = base or its * bs
            print(f"{cfg['name']:<20} batch {bs:4d}: {its:6.2f} it/s = {its * bs:7.0f} images/s "
                  f"({its * bs / base:4.2f}x the first batch size) | peak {mem:5.1f} GB")
            del model
            torch.cuda.empty_cache()
        return
    total_h = 0.0
    for path in args.configs:
        cfg = load_config(path)
        model, dtype, its, mem = bench_train(cfg, device, args.steps, args.warmup, args.compile)
        hours = cfg["train"]["total_steps"] / its / 3600
        total_h += hours
        print(f"{cfg['name']:<20} {its:6.2f} it/s | peak {mem:5.1f} GB | autocast {dtype} | "
              f"{cfg['train']['total_steps']} steps -> {hours:5.2f} h")
        if cfg["model"]["type"] == "ddpm":
            nfe = bench_nfe(model, dtype, device, args.sample_batch)
            eval_h = EVAL_NFE / nfe / 3600
            total_h += eval_h
            print(f"{'':<20} sampling {nfe:,.0f} NFE/s | 1000-step sampling of 10k images: "
                  f"{1e7 / nfe / 3600:.2f} h | full eval protocol ({EVAL_NFE / 1e6:.0f}M NFE): {eval_h:.2f} h")
        del model
        torch.cuda.empty_cache()
    print(f"\nprojected total (training + DDPM evaluation): {total_h:.1f} GPU-h", end="")
    if args.units_per_hour:
        print(f" ~= {total_h * args.units_per_hour:.0f} compute units at {args.units_per_hour}/h")
    else:
        print("  (pass --units-per-hour to convert to compute units)")


if __name__ == "__main__":
    main()
