# Results

Outputs of the reported runs: three models trained and evaluated on one NVIDIA A100-SXM4-80GB (Google Colab).
Every file here was produced by the repository's scripts; nothing was edited by hand.

| Path | Contents | Produced by |
|---|---|---|
| [`results.md`](results.md), [`results.csv`](results.csv) | All result tables, and every metric of every sample set in one CSV | `scripts/make_figures.py` |
| [`figures/`](figures/) | All 16 figures used in the report | `scripts/make_figures.py`, `scripts/plot_training_logs.py` |
| [`eval/items/*.json`](eval/items/) | Metrics of each sample set (FID, IS, KID, precision/recall, classifier statistics, memorisation, PSNR/SSIM) | `scripts/evaluate.py` |
| [`eval/curves/*.json`](eval/curves/) | FID-5k of the EMA snapshots every 10k training steps | `scripts/evaluate.py` |
| [`eval/bpd.json`](eval/bpd.json) | Test-set likelihood bounds; for the DDPM, all 1000 per-timestep terms | `scripts/evaluate.py` |
| [`eval/timing.json`](eval/timing.json) | Sampling throughput (batch 256) and single-image latency | `scripts/evaluate.py` |
| [`eval/runs.json`](eval/runs.json) | Parameters, steps, training GPU-hours and full config of each run | `scripts/evaluate.py` |
| [`eval/blind/`](eval/blind/) | Blind rating by Gemini (see below) | `scripts/blind_rating.py` |
| [`training/<run>/`](training/) | Complete training log (`metrics.jsonl`) and the exact config used | `scripts/train.py` |

## Sample-set naming

| Key | Meaning |
|---|---|
| `real_test` | The 10,000 real CIFAR-10 test images (reference row) |
| `vae_mlp_s{0,1,2}`, `vae_conv_s{0,1,2}` | VAE samples from the prior, 3 independent seeds |
| `vae_conv_raw_s0` | VAE-Conv samples with the raw (non-EMA) weights |
| `vae_*_recon` | VAE reconstructions of the test set |
| `ddpm_ancestral1000_s0` | DDPM, the paper's 1000-step ancestral sampler |
| `ddpm_ddim{10,25,50,100,250}_s0` | DDPM, deterministic DDIM sampling with that many steps |
| `ddpm_ddim100_s{1,2}` | Extra seeds for the DDIM-100 setting |
| `ddpm_raw_ddim100_s0` | DDPM with the raw (non-EMA) weights |

## Blind rating (`eval/blind/`)

| File | Contents |
|---|---|
| `set.npz` | The 150 shuffled images that were rated (50 real, 50 VAE-Conv, 50 DDPM), uint8 `[150, 3, 32, 32]` |
| `key.json` | Answer key: source and (for real images) true class of every image |
| `prompt.txt` | The rating instructions given to Gemini |
| `gemini_answers.csv` | Gemini's ratings: realism (1–5), class, real/generated (originally saved as `my_eval.csv`) |
| `blind_evaluation.xlsx` | Gemini's spreadsheet comparing its ratings with the key, with every image embedded |
| `make_xlsx.py` | Gemini's script that built the spreadsheet (reads `set.npz`, `key.json`, `my_eval.csv`) |
| `score.json`, `rater.json` | Summary scores computed by `scripts/blind_rating.py --score` |

## Notes

All sample-based metrics use 10,000 samples against the 50,000 CIFAR-10 training images (FID-10k). The generated
images themselves (about 30 MB per set) and the model checkpoints are too large for git. They are available in the
project's [Google Drive folder](https://drive.google.com/drive/folders/1SH42and7yRTIOOh9D9E-YndSEQ0Ilfb4?usp=sharing)
(see the main README).
