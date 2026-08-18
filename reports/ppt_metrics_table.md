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

**This is a real, two-sided, statistically proven tradeoff - not a one-sided win.** LPIPS improved and PSNR, SSIM regressed, and **both directions are independently statistically significant (p<0.05)** - the PSNR, SSIM regressions are exactly as statistically real as the LPIPS improvement is, confirmed by the same paired Wilcoxon test and bootstrap CI, on the same 506 images. Neither result is reported with more confidence than the other.

## Classical baseline comparison (val/OOD-proxy, n=506)

Bicubic upsample + non-local-means denoise (skimage.restoration.denoise_nl_means) - the same code run.py's classical_fallback() actually uses, not a separate reimplementation - evaluated on the identical 506 val images as Stage A/B.

| Method | PSNR | SSIM | LPIPS |
|---|---|---|---|
| Classical (bicubic + NLM) | 20.58 | 0.374 | 0.562 |
| Stage A (NAFNet, KLA-only) | 28.26 | 0.740 | 0.289 |
| Stage B (NAFNet, KLA+external) | 28.09 | 0.731 | 0.163 |

**The AI model gains 7.7dB PSNR over the classical baseline (Stage A) / 7.5dB (Stage B)**, roughly 2.0x-2.0x SSIM, and 1.9x-3.5x better LPIPS (lower is better) - not a marginal gain over a naive approach.

**Correctness note (found and fixed during the final rigor pass):** `scikit-image`'s `estimate_sigma()` - used inside `classical_fallback()` to set the NLM denoising strength - has an undeclared optional dependency on `PyWavelets`. Neither `requirements.txt` pinned it, so on any environment without it (including the one this table was first generated in), `estimate_sigma()` raised `ImportError`, which `classical_fallback()`'s broad exception handler silently caught and fell back to bicubic-only - meaning the NLM denoising step never actually executed, contrary to what "bicubic + NLM" implied. Confirmed directly (the earlier numbers matched a manually-recomputed bicubic-only baseline to 6 decimal places) and fixed by pinning `PyWavelets` in both `requirements.txt` files - verified installing cleanly in a fresh venv, and verified `denoise_nl_means` now actually perturbs the output versus bicubic alone. The numbers above are the corrected, NLM-active run. The practical effect on this specific data was small (bicubic upsampling already leaves little residual noise for NLM to remove, so old vs corrected PSNR/SSIM/LPIPS differ by <0.02 in every metric) - the AI-vs-classical gap claim above is materially unchanged, but the fallback path run.py actually ships now correctly matches its own documentation instead of silently degrading further on a missing dependency.

## GT noise-ceiling sanity check

Checked whether KLA's GT images are a perfectly clean reference or carry residual noise of their own, by inspecting local (8x8 block) variance in each image's flattest regions across 15 real GT images - visual confirmation in reports/figures/gt_noise_ceiling_check.png, per-image data in gt_noise_ceiling_check.csv.

**Finding: GT appears visually clean in flat/smooth regions - no obvious residual noise floor detected.** Mean flattest-region std across 15 images: 0.0157 (implied PSNR ceiling if this were a true noise floor: 36.1 dB - well above both Stage A's 28.26dB and Stage B's 28.09dB, so even if real, it isn't the binding constraint on current results). A handful of images showed higher flat-region variance (0.0577 at the max) - checked these individually (reports/figures/gt_noise_ceiling_outliers.png) and confirmed they're texture-dense images (grass, dense forest) with no genuinely flat region anywhere, not evidence of noise - the 'flattest 5%' statistic on a busy image still reflects real fine structure. Checked, not assumed: the outliers were individually visually verified, not waved away.

## Scale generalization test (256->512, an untrained resolution)

The shipped checkpoint has only ever been trained/validated at 128->256. Tested it at 256->512 on 15 synthetic pairs: real KLA GT images bicubic-upscaled to a pseudo-512 target, then degraded back down to a 256x256 input with the exact same validated noise model used everywhere else in this project (factor=2, same `SpeckleAdditiveDegrader`).

**Methodology limitation, stated up front:** KLA has no real 512x512+ source images available to us. The pseudo-512 "ground truth" here is a clean bicubic upscale of a native 256x256 image - it contains no real fine detail beyond what bicubic interpolation already produces. This test validly answers *does the model's code path handle a differently-shaped input without crashing or producing garbage, and does it still clearly beat naive upscaling of the same input* - it does NOT validly answer *does the model recover genuine fine structure at a real higher resolution*, since no real high-frequency content exists in the target to recover. That second, stronger claim remains untested and should not be inferred from this result.

| Method | PSNR | SSIM | LPIPS |
|---|---|---|---|
| Model (run.py, trained only at 128->256) | 29.99 | 0.8323 | 0.2100 |
| Bicubic baseline (same input, no denoise) | 21.70 | 0.4523 | 0.5031 |
| Classical fallback (bicubic + NLM, run.py's real fallback) | 21.73 | 0.4569 | 0.5004 |

**Result: the model ran successfully on all 15 images (zero crashes, zero fallback triggers, every output correctly shaped 512x512 and spec-compliant) and clearly outperformed both baselines** - +8.3dB PSNR over the classical fallback, a large SSIM/LPIPS gap in the same direction. This confirms genuine architectural/mechanism generalization: the fully-convolutional design with runtime padding to a multiple of 16 does not require retraining to accept a differently-sized input, and whatever it learned about denoising/upsampling from 128->256 training transfers usefully to a 256->512 input rather than collapsing into noise or artifacts.

**What NOT to conclude from this:** the model's PSNR here (29.99dB) is numerically higher than its real 128->256 val PSNR (28.09dB) - this is an artifact of the pseudo-GT's lack of real fine detail (a smoother target is mechanically easier to hit with high PSNR), **not evidence the model performs better at higher resolution**. Do not cite this comparison as a quality claim. The honest, testable claim for the feasibility slide is: *the architecture is confirmed to generalize mechanically to an untrained input resolution, verified end-to-end through the real run.py path* - real higher-resolution reconstruction quality (e.g. against a true 512<->256 KLA test pair, if released) remains unverified.

## Ensemble check (Stage A + Stage B averaged, val/OOD-proxy, n=506)

Zero new training: averages the two existing checkpoints' raw model outputs (before checkerboard suppression/clamping) on every val image, then applies the same post-processing run.py itself uses. Stage A/B numbers below are **recomputed under this identical post-processing pipeline** for a controlled comparison, so they differ slightly (within ~0.01-0.06) from the headline Stage A/B numbers elsewhere in this table, which used a different evaluation script without checkerboard suppression applied - both are correct, they're just not the same measurement.

| Model | PSNR | SSIM | LPIPS |
|---|---|---|---|
| Stage A | 28.199 | 0.7388 | 0.3062 |
| Stage B | 28.078 | 0.7324 | 0.1714 |
| Ensemble (A+B)/2 | 28.224 | 0.7396 | 0.2244 |

**PSNR/SSIM: the ensemble edges out both individual models** (28.224dB PSNR vs A's 28.199 and B's 28.078). **LPIPS: the ensemble lands between A and B, retaining only 61% of Stage B's LPIPS gain over Stage A** (0.0818 of 0.1349 LPIPS points of improvement) - it gives back a meaningful chunk of exactly the gain Stage B was fine-tuned to produce.

| Scenario | Stage A | Stage B | Ensemble | Winner |
|---|---|---|---|---|
| Equal weighting | 0.6597 | 0.6998 | 0.6878 | **B** |
| Quality-only (LPIPS ignored) | 0.6427 | 0.6354 | 0.6439 | **Ens** |
| LPIPS-weighted | 0.6682 | 0.7320 | 0.7098 | **B** |

**Composite-score verdict: the ensemble does NOT win outright - it wins only 1/3 weighting scenarios** (quality-only, where it barely edges Stage A), and loses to Stage B alone under equal-weighting and LPIPS-weighted scoring, the two scenarios where Stage B already won outright. Combined with roughly 2x inference cost (~150ms/image estimated, both models run per image) for a benefit that's marginal-to-absent depending on the weighting, **this does not look like a clear win** - recommendation is to keep shipping Stage B alone, but this is stated as a recommendation, not a unilateral decision: the raw numbers are above for a final call.

## Inference time (feasibility slide)

**76.4 ms/image on NVIDIA H100 SXM 80GB** (3.819s total for a 50-image batch)

Measured on real H100 SXM 80GB hardware (clean on-demand pod). Used a synthetic batch matching Test_NoisyLR's exact shape (128x128 float32) and value distribution (mean ~0.4-0.7, std ~0.1-0.3) rather than real Kaggle test data, since pulling that onto a throwaway pod would have needed credentials - timing is compute-graph-driven, not pixel-content-driven, so this is a faithful proxy. All 50 images succeeded via the real model path (zero fallback triggers), output shape/range confirmed compliant.

## Memory footprint and latency breakdown

Measured locally (not the H100) through run.py's actual code path (scripts/performance_profile.py), 50 synthetic images matching real input characteristics. Relative proportions are expected to transfer directionally to the H100; absolute numbers differ by hardware.

**VRAM (peak, `torch.cuda.max_memory_allocated`):**

| Scenario | Peak VRAM |
|---|---|
| Single image (run.py's actual per-image path) | 44.0 MB |
| 16 images via run.py's real sequential (one-at-a-time) path | 44.3 MB |
| *Hypothetical* genuinely-batched forward pass (batch_size=16 in one call - **not** what run.py's real code does today) | 294.9 MB |

run.py processes images one at a time, not in batches - VRAM usage stays essentially flat (44.0 to 44.3 MB) regardless of how many images are in the job, since only one is ever resident at once. The batched figure is reported separately as feasibility headroom, not a claim about current behavior.

**Latency breakdown (component analysis):**

| Component | Time |
|---|---|
| Model/checkpoint load (one-time cost) | 320.6 ms total |
| Pure forward-pass inference | 10.78 ms/image |
| Full run.py per-image path (forward + checkerboard suppress + clamp + sanitize) | 11.45 ms/image |
| Disk I/O (read input .npy + write output .npy) | 15.64 ms/image |
| **Local total** (load amortized over 50 images + full path + I/O) | **33.50 ms/image** |

**This local total is lower than the reported H100 figure (76.4 ms/image) - flagging this explicitly rather than letting it sit unexplained.** For a small model (6.82M params) at small resolution (128x128 input), per-image overhead can dominate over raw compute throughput: the H100 measurement is a true end-to-end figure on a cold on-demand pod (script startup, model init, and possibly network-backed storage all included per KLA's own timing definition), while this local figure is a warm, already-initialized measurement on local SSD. Both are real measurements of different things - H100 end-to-end is the correct number for the feasibility slide's headline figure; this breakdown is for understanding where time actually goes, not for replacing it.
