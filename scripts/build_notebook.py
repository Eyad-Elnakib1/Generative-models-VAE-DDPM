# Repository file: scripts/build_notebook.py
"""Generate notebooks/colab_runner.ipynb from the repository files.

Every source file becomes one `%%writefile <path>` cell, so the notebook shows the complete
from-scratch implementation and running it recreates the package on the Colab machine. The
repository stays the single source of truth: edit the .py files, then re-run

    python scripts/build_notebook.py

`tests/test_notebook.py` fails if the notebook is out of date.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "colab_runner.ipynb"

# (repository file, explanation shown above its code cell)
SOURCE_FILES = [
    ("pyproject.toml",
     "**Package definition.** Makes `import genmodels` work and lists the dependencies."),
    ("src/genmodels/__init__.py", "Package marker."),
    ("src/genmodels/nn_blocks.py",
     "**Shared building blocks** for both models, following the official DDPM code: a Wide-ResNet residual "
     "block (GroupNorm → Swish → conv, with the timestep embedding added in), single-head self-attention, "
     "down/up-sampling, the sinusoidal timestep embedding and the paper's weight initialisation."),
    ("src/genmodels/unet.py",
     "**DDPM noise-prediction network ε_θ(x_t, t)** — Ho et al. (2020), Appendix B: a PixelCNN++-style "
     "U-Net with four resolutions (32→4), two residual blocks per resolution and self-attention at 16×16. "
     "35.7M parameters, as in the paper."),
    ("src/genmodels/diffusion.py",
     "**The diffusion process** — Ho et al. (2020):\n"
     "* forward process `q_sample`: x_t = √ᾱ_t · x_0 + √(1−ᾱ_t) · ε, with the linear β schedule 1e-4 → 0.02, T = 1000;\n"
     "* training loss `training_loss`: L_simple = ‖ε − ε_θ(x_t, t)‖² (Algorithm 1);\n"
     "* sampling `p_sample_loop`: 1000 ancestral steps (Algorithm 2), plus DDIM (`ddim_sample_loop`) "
     "for fast sampling with the same trained network;\n"
     "* likelihood `calc_bpd`: the full variational bound in bits/dim (paper eq. 5, discretised decoder)."),
    ("src/genmodels/vae.py",
     "**The VAE** — Kingma & Welling (2013): Gaussian encoder q(z|x), reparameterisation z = μ + σ·ε, "
     "prior N(0, I), analytic KL (paper eq. 10), Gaussian decoder p(x|z) with learned variance, one sample "
     "per image. `MLPVAE` is the paper's own network; `ConvVAE` is the same method with a convolutional "
     "network of the U-Net's size, used for the main comparison."),
    ("src/genmodels/models.py", "**Model registry**: builds a model from the `model` section of a config."),
    ("src/genmodels/data.py",
     "**Data**: CIFAR-10 kept on the GPU, with shuffling that is a pure function of the step, so a resumed "
     "run sees exactly the same batches as an uninterrupted one."),
    ("src/genmodels/ema.py", "**EMA** of the weights (decay 0.9999 in the paper); samples come from the EMA model."),
    ("src/genmodels/utils.py", "**Utilities**: configs, seeding, crash-safe saving, precision choice, image grids."),
    ("src/genmodels/trainer.py",
     "**Training loop** shared by every model: bf16 mixed precision, learning-rate warm-up, gradient "
     "clipping, EMA, periodic evaluation / sample grids / snapshots, and checkpoints every 15 minutes "
     "for resuming after a Colab disconnect."),
    ("configs/ddpm_cifar10.yaml", "**DDPM hyper-parameters** (paper values, with sources)."),
    ("configs/vae_conv_cifar10.yaml", "**Convolutional VAE hyper-parameters** (matched to the DDPM run)."),
    ("configs/vae_mlp_cifar10.yaml", "**Paper MLP VAE hyper-parameters** (Kingma & Welling's own settings)."),
    ("scripts/train.py", "**Training script**: train or resume a run from a config."),
    ("scripts/benchmark.py", "**Calibration script**: measures speed on this GPU and projects the cost of the plan."),
]

# Evaluation code, written into the package in Part 3.
EVAL_FILES = [
    ("src/genmodels/eval/__init__.py", "Package marker."),
    ("src/genmodels/eval/metrics.py",
     "**Metrics.** FID, IS and KID use torch-fidelity's reference implementations on TF-compatible Inception-v3 "
     "features (extracted once per sample set). Our own code: precision/recall (realism vs. coverage, "
     "Kynkäänniemi et al. 2019), the nearest-neighbour memorisation check, PSNR/SSIM, and classifier statistics."),
    ("src/genmodels/eval/classifier.py",
     "**CIFAR-10 classifier** (ResNet-18, trained from scratch): a measuring instrument for *thematic consistency*, i.e. "
     "how recognisable generated images are and whether every class appears."),
    ("src/genmodels/eval/sampling.py",
     "**Sampling utilities**: load a trained run and generate reproducible sample sets (fixed noise per batch)."),
    ("scripts/train_classifier.py", "**Classifier training script.**"),
    ("scripts/evaluate.py",
     "**The evaluation protocol** (resumable: finished items are cached on Drive): reference statistics, every sample "
     "set, FID during training, bits/dim, and sampling speed."),
    ("scripts/make_figures.py", "**Result tables and figures** from the cached evaluation."),
    ("scripts/blind_rating.py", "**Blind human rating** of real vs. VAE vs. DDPM images (optional)."),
]


def _lines(text):
    return text.splitlines(True)


def build():
    cells = []

    def md(text):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": _lines(text.strip("\n"))})

    def code(text):
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                      "source": _lines(text.strip("\n"))})

    md("""
# VAE vs. DDPM on CIFAR-10 — implemented from scratch

This notebook contains the **complete implementation** of both models. Each code cell in Part 1 writes one
file of the repository (the first line of every file names it) and running them recreates the package on
the Colab machine. Parts 2–3 train and evaluate the models with it.

**How to use**
1. *Runtime → Change runtime type →* **L4** (or A100). Use the same GPU type for every final run.
2. At the start of **every** session, run everything up to Part 2 (*Runtime → Run before* on the Phase 0 cell).
3. Training is **resumable**: runs are stored on Google Drive. After a disconnect, re-run the setup and the
   same training cell; it continues from the last checkpoint (saved every 15 minutes).
4. Colab Pro has no background execution: keep this tab open while training.

*This notebook is generated from the repository by `scripts/build_notebook.py`; edit the repository files, not these cells.*
""")
    md("## 0. Setup")
    code("""
!nvidia-smi --query-gpu=name,memory.total --format=csv

from google.colab import drive
drive.mount('/content/drive')
DRIVE = '/content/drive/MyDrive/vae-vs-ddpm'

import os
os.makedirs(f'{DRIVE}/runs', exist_ok=True)
os.makedirs('/content/repo', exist_ok=True)
%cd /content/repo
!mkdir -p src/genmodels/eval configs scripts
!ln -sfn {DRIVE}/runs runs
""")
    md("# Part 1 — Implementation (from scratch)")
    for path, explanation in SOURCE_FILES:
        md(f"### `{path}`\n{explanation}")
        code(f"%%writefile {path}\n" + (ROOT / path).read_text(encoding="utf-8"))

    md("### Install the package and load CIFAR-10")
    code("""
!pip install -q -e ".[eval]"

# `pip install -e` is only picked up by Python processes started afterwards (the `!python ...` commands).
# This notebook's own process was already running, so add the source folder to its import path directly.
import sys, importlib
if '/content/repo/src' not in sys.path:
    sys.path.insert(0, '/content/repo/src')
importlib.invalidate_caches()

# Cache the CIFAR-10 archive on Drive so later sessions skip the download
!mkdir -p data && if [ -f {DRIVE}/cifar-10-python.tar.gz ]; then cp {DRIVE}/cifar-10-python.tar.gz data/; fi
!python -c "from genmodels.data import load_cifar10; load_cifar10('data'); load_cifar10('data', train=False)"
!if [ ! -f {DRIVE}/cifar-10-python.tar.gz ]; then cp data/cifar-10-python.tar.gz {DRIVE}/; fi
""")
    md("### Quick check of the implementation")
    code("""
import torch
from genmodels.models import build_model
from genmodels.utils import load_config

for name in ['ddpm_cifar10', 'vae_conv_cifar10', 'vae_mlp_cifar10']:
    model = build_model(load_config(f'configs/{name}.yaml')['model'])
    print(f'{name:<18} {sum(p.numel() for p in model.parameters()) / 1e6:6.2f}M parameters')
""")
    md("""
# Part 2 — Training

## Phase 0 — calibration (≈10 min)
Read this GPU's compute-unit rate from *Runtime → View resources* and enter it below. The benchmark measures
training speed and sampling throughput and projects the total cost of the plan. It runs with and without
`--compile`; use compile for training only if it is clearly faster.
""")
    code("""
UNITS_PER_HOUR = 1.7   # <- replace with the rate Colab shows
!python scripts/benchmark.py --units-per-hour {UNITS_PER_HOUR}
!python scripts/benchmark.py --units-per-hour {UNITS_PER_HOUR} --compile
""")
    md("""
## Phase 1 — VAE with the paper's own network (MLP)
Small model (4.9M parameters): about 20–40 minutes, under 1 compute unit.
""")
    code("""
!python scripts/train.py --config configs/vae_mlp_cifar10.yaml --run-dir runs/vae_mlp_cifar10
""")
    md("## Phase 2 — VAE (convolutional network, Kingma & Welling method)")
    code("""
COMPILE = 'false'   # 'true' if the benchmark showed a clear speed-up
!python scripts/train.py --config configs/vae_conv_cifar10.yaml --run-dir runs/vae_conv_cifar10 --set train.compile={COMPILE}
""")
    md("""
## Phase 3 — DDPM (Ho et al., 2020)
`DDPM_STEPS` is set from the calibration (paper: 800k). It can be raised later and the run resumed; the model
config itself must not change between sessions (the trainer refuses to resume if it does).
""")
    code("""
DDPM_STEPS = 150000
!python scripts/train.py --config configs/ddpm_cifar10.yaml --run-dir runs/ddpm_cifar10 --set train.total_steps={DDPM_STEPS} train.compile={COMPILE}
""")
    md("## Progress monitor (run any time)")
    code("""
import json, glob
import matplotlib.pyplot as plt
from IPython.display import Image, display

RUN = 'runs/ddpm_cifar10'   # or 'runs/vae_conv_cifar10'
rows = [json.loads(l) for l in open(f'{RUN}/metrics.jsonl')]
train_rows = [r for r in rows if 'loss' in r]
eval_rows = [r for r in rows if any(k.startswith('ema_') for k in r)]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
ax[0].plot([r['step'] for r in train_rows], [r['loss'] for r in train_rows]); ax[0].set_title('training loss'); ax[0].set_yscale('log')
for k in [k for k in eval_rows[-1] if k != 'step' and 'active' not in k] if eval_rows else []:
    ax[1].plot([r['step'] for r in eval_rows], [r[k] for r in eval_rows], label=k)
ax[1].legend(); ax[1].set_title('held-out metrics'); plt.show()
print(train_rows[-1])
display(Image(sorted(glob.glob(f'{RUN}/samples/*.png'))[-1]))
""")
    md("""
# Part 3 — Evaluation

**Quantitative** (10,000 samples per set, against the 50,000 training images):
FID and IS (the brief's metrics), KID, precision/recall (realism vs. diversity), likelihood in bits/dim,
FID vs. number of sampling steps (DDIM 10–250 and the paper's 1000-step sampler), sampling speed,
EMA on/off, VAE reconstruction quality, memorisation check, and a real-data reference row for every metric.

**Qualitative**: uncurated sample grids, samples grouped by class, least-recognisable samples, nearest
training images, VAE reconstructions and interpolations, the DDPM denoising process, and an optional
blind self-rating.

All results are cached in `runs/eval` on Drive; re-running a cell skips everything already done.
""")
    for path, explanation in EVAL_FILES:
        md(f"### `{path}`\n{explanation}")
        code(f"%%writefile {path}\n" + (ROOT / path).read_text(encoding="utf-8"))
    md("## Step 1 — classifier for thematic consistency (≈10 min)")
    code("""
!python scripts/train_classifier.py --out runs/classifier
""")
    md("""
## Step 2 — run the evaluation protocol (resumable)
Roughly 1.5–2.5 GPU-hours on an L4, dominated by DDPM sampling. Re-run this cell after a disconnect.
""")
    code("""
!python scripts/evaluate.py
""")
    md("## Step 3 — tables and figures")
    code("""
!python scripts/make_figures.py --model-figures

import glob
from IPython.display import Markdown, Image, display
display(Markdown(open('runs/eval/report/results.md').read()))
for f in sorted(glob.glob('runs/eval/report/*.png')):
    print(f); display(Image(f))
""")
    md("""
## Optional — blind self-rating (≈20 min, no GPU)
150 shuffled images (50 real, 50 VAE, 50 DDPM) are shown one at a time without their source. Rate each one;
answers are saved after every image, so you can stop and continue later. The sources are revealed only at
the end. Afterwards re-run Step 3 to add the results to the report.
""")
    code("""
import sys; sys.path.insert(0, 'scripts')
import blind_rating
if not os.path.exists('runs/eval/blind/set.npz'):
    blind_rating.prepare('runs/eval', per_source=50, data_root='data')
blind_rating.rate('runs/eval')
""")
    code("""
blind_rating.score('runs/eval')
""")
    md("""
## Results location
Everything is on Google Drive under `vae-vs-ddpm/runs/eval/report/` (results.md, results.csv, figures).
""")
    return {"cells": cells,
            "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "L4"},
                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 0}


def render():
    return json.dumps(build(), indent=1, ensure_ascii=False) + "\n"


if __name__ == "__main__":
    NOTEBOOK.parent.mkdir(exist_ok=True)
    NOTEBOOK.write_text(render(), encoding="utf-8", newline="\n")
    print(f"wrote {NOTEBOOK.relative_to(ROOT)} ({len(build()['cells'])} cells)")
