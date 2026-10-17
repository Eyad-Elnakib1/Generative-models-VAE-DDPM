# Results

All sample-based metrics use 10,000 samples against the 50,000 CIFAR-10 training images (FID-10k: systematically a little higher than the FID-50k reported in papers). ± is the standard deviation over 3 independent sample sets (evaluation noise, not training-seed variance).

## Main comparison

| Model | FID ↓ | IS ↑ | KID ×10³ ↓ | Precision ↑ | Recall ↑ | NLL bound (bits/dim) ↓ | NFE / image | Images/s ↑ | Params | Train GPU-h |
|---|---|---|---|---|---|---|---|---|---|---|
| *Real test images (reference)* | 3.15 | 10.97 | -0.01 | 0.681 | 0.693 | – | – | – | – | – |
| VAE — paper MLP | 135.97 ± 0.52 | 3.11 ± 0.02 | 127.57 ± 0.62 | 0.323 ± 0.006 | 0.003 ± 0.001 | 6.313 | 1 | 350,102.2 | 4.9M | 0.44 |
| VAE — conv | 121.85 ± 0.30 | 3.88 ± 0.05 | 151.18 ± 0.45 | 0.404 ± 0.010 | 0.073 ± 0.020 | 5.826 | 1 | 7,017.3 | 30.4M | 1.32 |
| DDPM — 1000 ancestral steps (paper sampler) | 6.58 | 8.59 | 1.54 | 0.694 | 0.583 | 3.794 | 1000 | 4.2 | 35.7M | 3.94 |
| DDPM — 100 DDIM steps | 9.66 ± 0.06 | 8.31 ± 0.10 | 6.35 ± 0.14 | 0.641 ± 0.007 | 0.592 ± 0.008 | 3.794 | 100 | 41.6 | 35.7M | 3.94 |

Speed measured on NVIDIA A100-SXM4-80GB (batch 256, bf16). Paper reference (DDPM, FID-50k): FID 3.17, IS 9.46, NLL ≤ 3.75 bits/dim after 800k steps.

## DDPM: sampling steps vs. quality (same network, same starting noise)

| Sampler | NFE | FID ↓ | IS ↑ | Precision | Recall | Images/s (batch 256) | Latency, 1 image |
|---|---|---|---|---|---|---|---|
| Ancestral (paper) | 1000 | 6.58 | 8.59 | 0.694 | 0.583 | 4.2 | 21.75 s |
| DDIM | 10 | 18.55 | 7.72 | 0.595 | 0.509 | 415.6 | 0.23 s |
| DDIM | 25 | 12.70 | 8.02 | 0.626 | 0.555 | 166.5 | 0.55 s |
| DDIM | 50 | 10.84 | 8.11 | 0.632 | 0.572 | 83.3 | 1.12 s |
| DDIM | 100 | 9.63 | 8.24 | 0.639 | 0.584 | 41.6 | 2.20 s |
| DDIM | 250 | 8.61 | 8.36 | 0.643 | 0.595 | 16.6 | 5.41 s |
| VAE (conv), for comparison | 1 | | | | | 7,017.3 | 14.2 ms |
| VAE (MLP), for comparison | 1 | | | | | 350,102.2 | 0.7 ms |

## Ablations that need no extra training

| Comparison | FID ↓ | IS ↑ | Precision | Recall |
|---|---|---|---|---|
| DDPM, DDIM-100, EMA weights | 9.63 | 8.24 | 0.639 | 0.584 |
| DDPM, DDIM-100, raw weights | 19.50 | 7.53 | 0.568 | 0.608 |
| VAE (conv), EMA weights | 121.51 | 3.87 | 0.415 | 0.059 |
| VAE (conv), raw weights | 118.90 | 3.90 | 0.412 | 0.076 |
| VAE (conv) reconstructions of test images | 57.86 | 6.27 | 0.440 | 0.318 |
| VAE (MLP) reconstructions | 99.37 | 4.55 | 0.316 | 0.097 |

VAE (conv) reconstruction quality on the test set: PSNR 24.07 dB, SSIM 0.772.

VAE (MLP) reconstruction quality on the test set: PSNR 23.61 dB, SSIM 0.771.

## Thematic consistency (CIFAR-10 classifier on generated images)

| Images | Mean confidence ↑ | Share with confidence > 0.9 ↑ | Class-histogram entropy (1 = all classes equal) | Rarest class share (0.10 = balanced) |
|---|---|---|---|---|
| Real test images | 0.875 | 0.742 | 1.000 | 0.098 |
| VAE (MLP) | 0.756 | 0.298 | 0.455 | 0.000 |
| VAE (conv) | 0.692 | 0.164 | 0.601 | 0.003 |
| DDPM (1000 steps) | 0.865 | 0.719 | 0.991 | 0.076 |
| DDPM (DDIM 100) | 0.854 | 0.670 | 0.995 | 0.080 |

Classifier accuracy on the real test set: 0.9362.

## Memorisation check

Distance from each sample to its nearest training image (Inception features), relative to the same distance for unseen real test images. A copying model would have many samples closer than 99% of real test images (last column ≫ 0.01).

| Model | Median NN distance / real-test median | Share closer than the real-test 1st percentile |
|---|---|---|
| VAE (MLP) | 1.302 | 0.0000 |
| VAE (conv) | 1.337 | 0.0000 |
| DDPM | 1.024 | 0.0009 |

## Likelihood (upper bounds on the test-set NLL, bits/dim)

| Model | bits/dim ↓ |
|---|---|
| VAE (MLP), ELBO | 6.313 ± 0.006 |
| VAE (conv), ELBO | 5.826 ± 0.007 |
| DDPM, variational bound, σ²=β (1000 test images) | 3.794 ± 0.015 |
| DDPM, variational bound, σ²=β̃ (1000 test images) | 3.781 ± 0.019 |

## Blind rating (rater: Gemini, 150 images)

| Source | Mean realism (1–5) | Named a class | Judged 'real' |
|---|---|---|---|
| Real images | 4.58 | 1.00 | 0.96 |
| VAE (conv) | 1.34 | 0.24 | 0.00 |
| DDPM | 4.28 | 0.98 | 0.96 |

Real-vs-generated accuracy: 0.67.
