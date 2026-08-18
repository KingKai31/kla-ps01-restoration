| Stage | Split | PSNR | SSIM | LPIPS | n |
|---|---|---|---|---|---|
| A (KLA-only), epoch 40 | Train (seen) | 28.63 | 0.786 | 0.236 | 508 |
| A (KLA-only), epoch 40 | Val/OOD-proxy | 28.26 | 0.740 | 0.289 | 506 |
| B (KLA+external), epoch 15 (best by val_psnr) | Train (seen) | 28.50 | 0.782 | 0.105 | 508 |
| B (KLA+external), epoch 15 (best by val_psnr) | Val/OOD-proxy | 28.09 | 0.731 | 0.163 | 506 |

**Gap A->B (val/OOD-proxy):** PSNR -0.18, SSIM -0.009, LPIPS -0.126 (improved - lower LPIPS is better)

Shipped checkpoint selection: **best checkpoint by validation PSNR during training** (epoch 15 of 20) - the standard, defensible checkpointing criterion used throughout this project (Stage A's shipped checkpoint was selected the same way). Per-epoch history was not retained (no history JSON was written during the Kaggle run, and log retrieval was attempted but not recoverable) - closed as a line of investigation; the best-by-val-PSNR checkpoint stands as the shipped artifact.

## Composite score sensitivity (Stage A vs Stage B, val/OOD-proxy)

KLA's actual SSIM/pSNR/LPIPS weighting is unknown. Stage B's raw numbers show a real tradeoff (LPIPS much better, PSNR/SSIM slightly worse), so this checks how that tradeoff holds up under different plausible weightings rather than assuming LPIPS improvement = win.

PSNR normalized to [0,1] via a fixed 20-35 dB reference range (not min-max across just these two values, which would be circular).

| Scenario | Stage A score | Stage B score | Winner | Margin |
|---|---|---|---|---|
| Equal weighting (1/3 SSIM + 1/3 norm-PSNR + 1/3 (1-LPIPS)) | 0.6675 | 0.7025 | **B** | 4.98% |
| Quality-only (1/2 SSIM + 1/2 norm-PSNR, LPIPS ignored) | 0.6457 | 0.6351 | **A** | 1.64% |
| LPIPS-weighted (50% (1-LPIPS) + 25% SSIM + 25% norm-PSNR) | 0.6784 | 0.7362 | **B** | 7.85% |

**Stage B wins 2/3 scenarios** (equal-weighting and LPIPS-weighted). It only loses when LPIPS is ignored entirely (quality-only scenario, margin 1.64%) - a narrow margin, not a decisive regression. This is an honest read of a real tradeoff, not a guarantee of KLA's actual scoring outcome.

## Statistical significance (Stage A vs Stage B, paired, n=506)

Paired Wilcoxon signed-rank test (non-parametric, makes no normality assumption) on the same val images scored by both models - answers whether each metric's change is statistically real or could be noise, not just which mean is bigger. Bootstrap 95% CI (1000 resamples) for the mean difference (B-A) reported alongside.

| Metric | Mean A | Mean B | Mean diff (B-A) | 95% CI | Wilcoxon p-value | Significant (p<0.05)? |
|---|---|---|---|---|---|---|
| PSNR | 28.2650 | 28.0859 | -0.1791 | [-0.2027, -0.1568] | 3.54e-47 | **Yes** |
| SSIM | 0.7404 | 0.7312 | -0.0093 | [-0.0105, -0.0080] | 2.80e-45 | **Yes** |
| LPIPS | 0.2889 | 0.1627 | -0.1262 | [-0.1362, -0.1162] | 9.67e-79 | **Yes** |

**PSNR, SSIM, LPIPS changes are statistically significant (p<0.05).**

## Inference time (feasibility slide)

**76.4 ms/image on NVIDIA H100 SXM 80GB** (3.819s total for a 50-image batch)

Measured on real H100 SXM 80GB hardware (clean on-demand pod). Used a synthetic batch matching Test_NoisyLR's exact shape (128x128 float32) and value distribution (mean ~0.4-0.7, std ~0.1-0.3) rather than real Kaggle test data, since pulling that onto a throwaway pod would have needed credentials - timing is compute-graph-driven, not pixel-content-driven, so this is a faithful proxy. All 50 images succeeded via the real model path (zero fallback triggers), output shape/range confirmed compliant.
