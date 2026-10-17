# Repository file: src/genmodels/trainer.py
"""One training loop for every model: mixed precision, EMA, warm-up, gradient clipping,
periodic evaluation / sample grids / EMA snapshots, and crash-safe resumable checkpoints.

Everything a run produces lives in `run_dir`:
  config.yaml        resolved config
  metrics.jsonl      one JSON object per log / eval event
  samples/           EMA sample grids during training (fixed noise, so progress is comparable)
  snapshots/         EMA weights every `snapshot_every` steps (for FID-vs-training-step curves)
  ckpt_last.pt       full resumable state (model, EMA, optimiser, RNG, step, GPU-seconds)
  ema_final.pt       final EMA weights + bookkeeping, used for evaluation
"""
import json
import math
import time
from pathlib import Path

import torch
import yaml

from .data import GPUBatcher
from .ema import EMA
from .models import build_model, network
from .utils import atomic_save, autocast_dtype, save_grid, seed_everything


def make_optimizer(model, ocfg, num_train, num_dims):
    params = model.parameters()
    name = ocfg["name"]
    if name == "adam":
        return torch.optim.Adam(params, lr=ocfg["lr"], betas=tuple(ocfg.get("betas", (0.9, 0.999))),
                                eps=ocfg.get("eps", 1e-8), weight_decay=ocfg.get("weight_decay", 0.0))
    if name == "adagrad":
        wd = ocfg.get("weight_decay", 0.0)
        if wd == "prior":
            # Kingma & Welling: MAP with p(theta) = N(0, I). Our loss is the per-dim mean of the
            # per-image -ELBO, so the prior's gradient theta / N becomes theta / (N * D).
            wd = 1.0 / (num_train * num_dims)
        return torch.optim.Adagrad(params, lr=ocfg["lr"], weight_decay=wd)
    raise ValueError(f"unknown optimiser {name!r}")


def train(cfg, run_dir, train_images, test_images, device=None, max_steps=None, log=print):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    run_dir = Path(run_dir)
    (run_dir / "samples").mkdir(parents=True, exist_ok=True)
    (run_dir / "snapshots").mkdir(exist_ok=True)
    tcfg = cfg["train"]
    seed_everything(cfg.get("seed", 0))

    model = build_model(cfg["model"]).to(device)
    ema = EMA(model, tcfg["ema_decay"])
    opt = make_optimizer(model, tcfg["optim"], len(train_images), train_images[0].numel())
    batcher = GPUBatcher(train_images, tcfg["batch_size"], cfg.get("seed", 0), cfg["data"].get("hflip", True), device)
    test_x = test_images[:tcfg.get("eval_images", 2000)].to(device)

    step, train_seconds, gpus = 0, 0.0, []
    ckpt_path = run_dir / "ckpt_last.pt"
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        step, train_seconds, gpus = ck["step"], ck["train_seconds"], ck["gpus"]
        # map_location moved every tensor to `device`; RNG states must be CPU ByteTensors
        torch.set_rng_state(ck["rng_cpu"].cpu())
        if device.type == "cuda" and ck.get("rng_cuda") is not None:
            torch.cuda.set_rng_state(ck["rng_cuda"].cpu(), device)
        if ck["cfg"]["model"] != cfg["model"]:
            raise RuntimeError("model config differs from the checkpoint's; refusing to resume")
        log(f"resumed from step {step} ({train_seconds / 3600:.2f} GPU-h so far)")
    else:
        with open(run_dir / "config.yaml", "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    if gpu_name not in gpus:
        gpus.append(gpu_name)

    if tcfg.get("compile", False):
        network(model).compile()
    dtype = autocast_dtype(tcfg.get("precision", "auto"), device)
    total = max_steps or tcfg["total_steps"]
    n_params = sum(p.numel() for p in model.parameters())
    log(f"{cfg['name']}: {n_params / 1e6:.2f}M params | {gpu_name} | autocast={dtype} | steps {step} -> {total}")

    metrics_f = open(run_dir / "metrics.jsonl", "a")

    def emit(row):
        metrics_f.write(json.dumps(row) + "\n")
        metrics_f.flush()

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def save_ckpt():
        atomic_save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                     "step": step, "train_seconds": train_seconds, "gpus": gpus, "cfg": cfg,
                     "rng_cpu": torch.get_rng_state(),
                     "rng_cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None},
                    ckpt_path)

    def autocast():
        return torch.autocast(device.type, dtype=dtype or torch.float32, enabled=dtype is not None)

    base_lr, warmup = tcfg["optim"]["lr"], tcfg.get("warmup_steps", 0)
    sample_gen_seed = 12345  # fixed noise for the progress grids
    win_t0, win_steps, loss_acc, extra_acc, acc_n = time.perf_counter(), 0, None, {}, 0
    last_ckpt = time.time()

    def close_window():
        """Account wall time spent in training steps since the window opened."""
        nonlocal train_seconds, win_t0, win_steps
        sync()
        dt = time.perf_counter() - win_t0
        train_seconds += dt
        its = win_steps / dt if dt > 0 else 0.0
        win_t0, win_steps = time.perf_counter(), 0
        return its

    model.train()
    while step < total:
        x = batcher.batch(step)
        lr = base_lr * min(1.0, (step + 1) / warmup) if warmup else base_lr
        for g in opt.param_groups:
            g["lr"] = lr
        with autocast():
            out = model(x)
        loss = out["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = (torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
                 if tcfg.get("grad_clip") else None)
        opt.step()
        ema.update(model)
        step += 1
        win_steps += 1
        loss_acc = loss.detach() if loss_acc is None else loss_acc + loss.detach()
        acc_n += 1
        for k, v in out.items():
            if k != "loss":
                extra_acc[k] = extra_acc.get(k, 0.0) + v

        if step % tcfg.get("log_every", 100) == 0 or step == total:
            n = acc_n
            mean_loss = (loss_acc / n).item()
            if not math.isfinite(mean_loss):
                raise FloatingPointError(f"non-finite loss at step {step}; last checkpoint is kept")
            its = close_window()
            row = {"step": step, "loss": mean_loss, "lr": lr, "it_s": its, "gpu_h": train_seconds / 3600,
                   **{k: (v / n).item() if torch.is_tensor(v) else v / n for k, v in extra_acc.items()}}
            if gnorm is not None:
                row["grad_norm"] = gnorm.item()
            if device.type == "cuda":
                row["mem_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
            emit(row)
            log(" | ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()))
            loss_acc, extra_acc, acc_n = None, {}, 0

        eval_now = step % tcfg.get("eval_every", 5000) == 0 or step == total
        sample_now = step % tcfg.get("sample_every", 5000) == 0 or step == total
        snap_now = step % tcfg.get("snapshot_every", 10000) == 0
        ckpt_now = time.time() - last_ckpt > 60 * tcfg.get("ckpt_every_minutes", 15) or step == total
        if eval_now or sample_now or snap_now or ckpt_now:
            close_window()
            if eval_now:
                g = torch.Generator(device).manual_seed(0)
                model.eval()
                with autocast():
                    row = {"step": step, **{f"raw_{k}": v for k, v in model.eval_metrics(test_x, g).items()}}
                    g.manual_seed(0)
                    row.update({f"ema_{k}": v for k, v in ema.model.eval_metrics(test_x, g).items()})
                model.train()
                emit(row)
                log(f"eval @ {step}: " + ", ".join(f"{k}={v:.4g}" for k, v in row.items() if k != "step"))
            if sample_now:
                g = torch.Generator(device).manual_seed(sample_gen_seed)
                with autocast():
                    imgs = ema.model.sample(64, sampler="ddim", steps=100, generator=g, device=device)
                save_grid(imgs.float(), run_dir / "samples" / f"step_{step:07d}.png")
            if snap_now:
                atomic_save({"ema": ema.state_dict(), "step": step}, run_dir / "snapshots" / f"ema_{step:07d}.pt")
            if ckpt_now:
                save_ckpt()
                last_ckpt = time.time()
            win_t0 = time.perf_counter()  # time spent above is not training time

    atomic_save({"ema": ema.state_dict(), "model": model.state_dict(), "step": step, "cfg": cfg,
                 "train_seconds": train_seconds, "gpus": gpus, "n_params": n_params},
                run_dir / "ema_final.pt")
    metrics_f.close()
    log(f"done: {step} steps, {train_seconds / 3600:.2f} GPU-h on {gpus}")
    return model, ema
