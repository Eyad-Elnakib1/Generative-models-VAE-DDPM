# Repository file: scripts/evaluate.py
"""Full evaluation protocol. Resumable: every finished item is cached under --eval-dir and skipped
when the script is re-run, so a Colab disconnect only loses the item in progress.

Stages
  reference  Inception features / FID statistics of CIFAR-10 train (50k) and test (10k); real-data
             reference row (the ceiling every metric can reach).
  samples    sample sets of every model: FID, IS, KID, precision/recall, classifier-based thematic
             consistency, memorisation check; reconstructions (VAE) with PSNR / SSIM / FID.
  curves     FID during training from the EMA snapshots (5k samples; DDPM with 50 DDIM steps).
  bpd        likelihood bound in bits/dim on the test set (DDPM: full variational bound, VAE: ELBO).
  timing     sampling throughput (batch 256) and single-image latency on this GPU.

    python scripts/evaluate.py                      # everything, 10k samples per set
    python scripts/evaluate.py --stages samples     # one stage
    python scripts/evaluate.py --quick              # small smoke-test version
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from genmodels.data import load_cifar10
from genmodels.eval import metrics as M
from genmodels.eval.classifier import load_classifier, predict_proba
from genmodels.eval.sampling import autocast, generate, load_run, load_samples, reconstruct, save_samples

DDIM_STEPS = [10, 25, 50, 100, 250]


def sample_set_specs(args):
    """Every sample set of the protocol. `family` is the run; `kind` is 'sample' or 'recon'."""
    n = args.n
    specs = []
    if args.mlp:
        specs += [dict(key=f"vae_mlp_s{s}", family="mlp", label="VAE (paper MLP)", seed=s, n=n) for s in range(3)]
        specs += [dict(key="vae_mlp_recon", family="mlp", label="VAE (paper MLP) reconstructions", kind="recon")]
    if args.vae:
        specs += [dict(key=f"vae_conv_s{s}", family="vae", label="VAE (conv)", seed=s, n=n) for s in range(3)]
        specs += [dict(key="vae_conv_raw_s0", family="vae", label="VAE (conv), no EMA", weights="raw", seed=0, n=n),
                  dict(key="vae_conv_recon", family="vae", label="VAE (conv) reconstructions", kind="recon")]
    if args.ddpm:
        specs += [dict(key="ddpm_ancestral1000_s0", family="ddpm", label="DDPM (1000 ancestral steps)",
                       sampler="ddpm", steps=1000, seed=0, n=n)]
        specs += [dict(key=f"ddpm_ddim{k}_s0", family="ddpm", label=f"DDPM (DDIM {k} steps)", sampler="ddim",
                       steps=k, seed=0, n=n) for k in DDIM_STEPS]
        specs += [dict(key=f"ddpm_ddim100_s{s}", family="ddpm", label="DDPM (DDIM 100 steps)", sampler="ddim",
                       steps=100, seed=s, n=n) for s in (1, 2)]
        specs += [dict(key="ddpm_raw_ddim100_s0", family="ddpm", label="DDPM (DDIM 100 steps), no EMA",
                       weights="raw", sampler="ddim", steps=100, seed=0, n=n)]
    for s in specs:
        s.setdefault("kind", "sample")
        s.setdefault("weights", "ema")
        s.setdefault("sampler", None)
        s.setdefault("steps", None)
        s["nfe"] = 1 if s["family"] in ("vae", "mlp") else s["steps"]
    if args.quick:
        for s in specs:
            if s["kind"] == "sample":
                s["n"] = min(s["n"], 200 if s["family"] == "ddpm" else 1000)
    return specs


class Evaluator:
    def __init__(self, args):
        self.args = args
        self.dev = "cuda"
        self.out = Path(args.eval_dir)
        for sub in ("samples", "items", "curves"):
            (self.out / sub).mkdir(parents=True, exist_ok=True)
        self.train_x, self.train_y = load_cifar10(args.data_root, train=True)
        self.test_x, self.test_y = load_cifar10(args.data_root, train=False)
        if args.quick:
            self.test_x, self.test_y = self.test_x[:1000], self.test_y[:1000]
        self._inception, self._classifier, self._models, self._ref = None, None, {}, None
        self.runs = {"ddpm": args.ddpm, "vae": args.vae, "mlp": args.mlp}

    # ---- lazily created tools ---------------------------------------------------------------
    @property
    def inception(self):
        if self._inception is None:
            self._inception = M.InceptionFeatures(self.dev)
        return self._inception

    @property
    def classifier(self):
        if self._classifier is None and Path(self.args.classifier).exists():
            self._classifier = load_classifier(self.args.classifier, self.dev)
        return self._classifier

    def model(self, family, weights="ema"):
        key = (family, weights)
        if key not in self._models:
            self._models = {}  # keep one model in memory at a time
            torch.cuda.empty_cache()
            self._models[key] = load_run(self.runs[family], weights, self.dev)
        return self._models[key][0]

    # ---- reference statistics -----------------------------------------------------------------
    def reference(self):
        path = self.out / "reference.pt"
        if self._ref is None and path.exists():
            self._ref = torch.load(path, weights_only=False)
        if self._ref is not None:
            return self._ref
        print("computing Inception features of CIFAR-10 train (50k) and test ...")
        train_pool, _ = self.inception(self.train_x)
        test_pool, test_logits = self.inception(self.test_x)
        g = torch.Generator().manual_seed(0)
        pr_idx = torch.randperm(len(train_pool), generator=g)[:len(self.test_x)]
        test_nn, _ = M.nearest_neighbours(test_pool, train_pool, self.dev)
        self._ref = {"train_stats": M.fid_statistics(train_pool), "train_pool": train_pool.half(),
                     "pr_idx": pr_idx, "test_pool": test_pool.half(), "test_logits": test_logits,
                     "test_nn_dist": test_nn}
        torch.save(self._ref, path)
        return self._ref

    # ---- one sample set -------------------------------------------------------------------------
    def evaluate_images(self, images, spec, compare_to=None):
        """All metrics of one set of uint8 images; `compare_to` = paired originals for reconstructions."""
        ref = self.reference()
        train_pool = ref["train_pool"].float()
        pool, logits = self.inception(images)
        res = {"n": len(images),
               "fid": M.fid_from_statistics(M.fid_statistics(pool), ref["train_stats"])}
        res["is_mean"], res["is_std"] = M.inception_score(logits)
        res["kid_mean"], res["kid_std"] = M.kid(pool, train_pool)
        res["precision"], res["recall"] = M.precision_recall(train_pool[ref["pr_idx"][:len(pool)]], pool, device=self.dev)
        nn_dist, nn_idx = M.nearest_neighbours(pool, train_pool, self.dev)
        res.update(M.memorisation_stats(nn_dist, ref["test_nn_dist"]))
        per_sample = {"nn_dist": nn_dist.numpy(), "nn_idx": nn_idx.numpy()}
        if self.classifier is not None:
            probs = predict_proba(self.classifier, images, self.dev)
            res.update(M.class_statistics(probs))
            per_sample["confidence"], per_sample["pred"] = (t.numpy() for t in probs.max(1))
        if compare_to is not None:
            res["psnr"] = M.psnr(images, compare_to).mean().item()
            res["ssim"] = M.ssim(images, compare_to).mean().item()
        return res, per_sample

    def run_spec(self, spec):
        item_path = self.out / "items" / f"{spec['key']}.json"
        if item_path.exists():
            return
        print(f"[{spec['key']}] {spec['label']}")
        sample_path = self.out / "samples" / f"{spec['key']}.npz"
        compare_to, seconds = None, None
        if sample_path.exists():
            images = load_samples(sample_path)
            seconds = float(np.load(sample_path).get("seconds", np.nan))
        elif spec["kind"] == "recon":
            images = reconstruct(self.model(spec["family"]), self.test_x, device=self.dev)
            save_samples(sample_path, images)
        else:
            images, seconds = generate(self.model(spec["family"], spec["weights"]), spec["n"], spec["sampler"],
                                       spec["steps"], spec["seed"], self.args.batch_size, self.dev)
            save_samples(sample_path, images, seconds=seconds)
        if spec["kind"] == "recon":
            compare_to = self.test_x
        res, per_sample = self.evaluate_images(images, spec, compare_to)
        res.update({k: v for k, v in spec.items() if k != "n"}, sample_gpu_seconds=seconds)
        np.savez(self.out / "items" / f"{spec['key']}_per_sample.npz", **per_sample)
        item_path.write_text(json.dumps(res, indent=1))
        print("   " + ", ".join(f"{k}={res[k]:.4g}" for k in ("fid", "is_mean", "precision", "recall") if k in res))

    def real_reference(self):
        spec = dict(key="real_test", family="real", label="Real CIFAR-10 test images", kind="real", nfe=0)
        item_path = self.out / "items" / "real_test.json"
        if item_path.exists():
            return
        res, per_sample = self.evaluate_images(self.test_x, spec)
        if self.classifier is not None:
            res["classifier_accuracy"] = float((per_sample["pred"] == self.test_y.numpy()).mean())
        res.update(spec)
        np.savez(self.out / "items" / "real_test_per_sample.npz", **per_sample)
        item_path.write_text(json.dumps(res, indent=1))
        print(f"[real_test] FID {res['fid']:.2f}, IS {res['is_mean']:.2f}")

    # ---- FID during training --------------------------------------------------------------------
    def curves(self):
        n = 1000 if self.args.quick else self.args.curve_n
        ref = self.reference()
        for family in ("ddpm", "vae"):
            run = self.runs[family]
            if not run:
                continue
            for snap in sorted(Path(run, "snapshots").glob("ema_*.pt")):
                step = int(snap.stem.split("_")[1])
                path = self.out / "curves" / f"{family}_{step:07d}.json"
                if path.exists():
                    continue
                model, _ = load_run(run, device=self.dev, snapshot=snap)
                images, _ = generate(model, n, "ddim", 50, seed=0, batch_size=self.args.batch_size, device=self.dev)
                pool, logits = self.inception(images)
                res = {"family": family, "step": step, "n": n, "sampler": "ddim50" if family == "ddpm" else "prior",
                       "fid": M.fid_from_statistics(M.fid_statistics(pool), ref["train_stats"]),
                       "is_mean": M.inception_score(logits)[0]}
                path.write_text(json.dumps(res))
                print(f"[curve] {family} step {step}: FID-{n // 1000}k {res['fid']:.2f}")
                del model
                torch.cuda.empty_cache()

    # ---- likelihood ---------------------------------------------------------------------------------
    @torch.no_grad()
    def bpd(self):
        path = self.out / "bpd.json"
        out = json.loads(path.read_text()) if path.exists() else {}
        g = torch.Generator(self.dev)
        for family in ("vae", "mlp"):
            if self.runs[family] and family not in out:
                model = self.model(family)
                vals, recon, kl = [], [], []
                g.manual_seed(0)
                for i in range(0, len(self.test_x), 500):
                    x = model.dequantize(self.test_x[i:i + 500].to(self.dev), g)
                    r = model.neg_elbo(x, g)
                    vals.append(model.nats_to_bpd(r["neg_elbo"], x[0].numel()).cpu())
                    recon.append(r["recon"].cpu())
                    kl.append(r["kl"].cpu())
                vals = torch.cat(vals)
                out[family] = {"bpd": vals.mean().item(), "bpd_sem": (vals.std() / math.sqrt(len(vals))).item(),
                               "recon_nats": torch.cat(recon).mean().item(), "kl_nats": torch.cat(kl).mean().item(),
                               "n": len(vals)}
                path.write_text(json.dumps(out, indent=1))
                print(f"[bpd] {family}: {out[family]['bpd']:.3f} bits/dim")
        if self.runs["ddpm"] and "ddpm" not in out:
            model = self.model("ddpm")
            n = 8 if self.args.quick else self.args.bpd_images
            totals = {"fixedlarge": [], "fixedsmall": []}
            terms = {"fixedlarge": 0.0, "fixedsmall": 0.0}
            g.manual_seed(0)
            for i in range(0, n, 250):
                x0 = model.preprocess(self.test_x[i:min(n, i + 250)].to(self.dev))
                t0 = time.time()
                # full fp32: the likelihood bound is a precise quantity, so no bf16 here
                res = model.diffusion.calc_bpd(model.net, x0, ("fixedlarge", "fixedsmall"), generator=g)
                for v in totals:
                    totals[v].append(res[v]["total"].cpu())
                    terms[v] = terms[v] + res[v]["terms"].sum(0).cpu()
                print(f"   bpd batch {i // 250 + 1}: {time.time() - t0:.0f}s")
            out["ddpm"] = {}
            for v in totals:
                vals = torch.cat(totals[v])
                out["ddpm"][v] = {"bpd": vals.mean().item(), "bpd_sem": (vals.std() / math.sqrt(len(vals))).item(),
                                  "terms_mean": (terms[v] / len(vals)).tolist(), "n": len(vals)}
            path.write_text(json.dumps(out, indent=1))
            print(f"[bpd] ddpm: {out['ddpm']['fixedlarge']['bpd']:.3f} (sigma^2=beta), "
                  f"{out['ddpm']['fixedsmall']['bpd']:.3f} (sigma^2=beta~) bits/dim")

    # ---- speed ----------------------------------------------------------------------------------------
    @torch.no_grad()
    def timing(self):
        path = self.out / "timing.json"
        out = json.loads(path.read_text()) if path.exists() else {}
        out["gpu"] = torch.cuda.get_device_name()
        configs = [("mlp", None, None), ("vae", None, None), ("ddpm", "ddpm", 1000)] + \
                  [("ddpm", "ddim", k) for k in DDIM_STEPS]
        for family, sampler, steps in configs:
            key = f"{family}_{sampler or 'prior'}{steps or ''}"
            if not self.runs[family] or key in out:
                continue
            model = self.model(family)
            row = {"family": family, "sampler": sampler, "steps": steps, "nfe": steps or 1}
            for batch, name in [(256, "throughput_img_s"), (1, "latency_s")]:
                generate(model, batch, "ddim" if sampler else None, 5 if sampler else None, seed=1,
                         batch_size=batch, device=self.dev)  # warm-up / autotune
                _, sec = generate(model, batch, sampler, steps, seed=0, batch_size=batch, device=self.dev)
                row[name] = batch / sec if batch > 1 else sec
            out[key] = row
            path.write_text(json.dumps(out, indent=1))
            print(f"[timing] {key}: {row['throughput_img_s']:.1f} img/s (batch 256), {row['latency_s'] * 1000:.1f} ms/img (batch 1)")

    def run_info(self):
        info = {}
        for family, run in self.runs.items():
            if run and Path(run, "ema_final.pt").exists():
                f = torch.load(Path(run) / "ema_final.pt", map_location="cpu", weights_only=False)
                info[family] = {"run": str(run), "steps": f["step"], "train_gpu_hours": f["train_seconds"] / 3600,
                                "gpus": f["gpus"], "n_params": f["n_params"], "config": f["cfg"]}
        (self.out / "runs.json").write_text(json.dumps(info, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="runs/eval")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--ddpm", default="runs/ddpm_cifar10")
    ap.add_argument("--vae", default="runs/vae_conv_cifar10")
    ap.add_argument("--mlp", default="runs/vae_mlp_cifar10")
    ap.add_argument("--classifier", default="runs/classifier/classifier.pt")
    ap.add_argument("--stages", nargs="*", default=["reference", "samples", "curves", "bpd", "timing"])
    ap.add_argument("--n", type=int, default=10000, help="samples per set")
    ap.add_argument("--curve-n", type=int, default=5000)
    ap.add_argument("--bpd-images", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=500)
    ap.add_argument("--quick", action="store_true", help="small sets for a smoke test")
    args = ap.parse_args()
    if args.quick and args.eval_dir == "runs/eval":
        args.eval_dir = "runs/eval_quick"  # smoke-test results never mix with the real ones
    for name in ("ddpm", "vae", "mlp"):  # runs that do not exist (yet) are skipped
        if not Path(getattr(args, name), "ema_final.pt").exists():
            print(f"note: no finished run at {getattr(args, name)}; skipping {name}")
            setattr(args, name, None)
    torch.backends.cudnn.benchmark = True

    ev = Evaluator(args)
    ev.run_info()
    if "reference" in args.stages:
        ev.reference()
        ev.real_reference()
    if "samples" in args.stages:
        for spec in sample_set_specs(args):
            ev.run_spec(spec)
    if "curves" in args.stages:
        ev.curves()
    if "bpd" in args.stages:
        ev.bpd()
    if "timing" in args.stages:
        ev.timing()
    print(f"done; results in {ev.out}")


if __name__ == "__main__":
    main()
