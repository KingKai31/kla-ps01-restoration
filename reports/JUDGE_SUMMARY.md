# Judge summary — read this first

One page, ~90 seconds. Deeper evidence for every claim below is pointed to
inline — this file makes claims, the linked files prove them.

## What it does

Restores KLA semiconductor inspection images degraded by **simultaneous
multiplicative speckle noise + additive noise + 128→256 spatial
downsampling**, in a single forward pass. Entry point: `python run.py
<input_dir> <output_dir>` — reads `.npy` grayscale arrays, writes restored
`.npy` arrays, no other setup. Architecture: NAFNet-style U-Net (6.82M
params) — simplified channel attention, gated activation-free blocks,
fused pixel-shuffle upsampling head. Full spec: [run.py](../run.py),
[submission/Phoenix/README.md](../submission/Phoenix/README.md).

## Core innovation: measured, not assumed, noise physics

Before building anything, decomposed KLA's real paired training data to
find the actual degradation model rather than guessing one:
`NoisyLR = box_downsample(GT) × M + A`, where `M ~ Gamma(L, 1/L)`
(multiplicative speckle, L varies 3.8–50.9 across sources — randomized,
not fixed) and `A ~ N(μ, σ)` (additive, σ ≈ 0.007 median). Validated the
decomposition by generating synthetic degraded data from *only* this model
and confirming it matches real NoisyLR on 400 held-out pairs (KS-test on
noise std, p=0.64) — not assumed to work, checked. Full analysis:
[README.md](../README.md#status), [reports/phase1_decomposition_summary.json](phase1_decomposition_summary.json),
[reports/insurance_check_summary.json](insurance_check_summary.json).

## Headline numbers (val/OOD-proxy split, n=506)

| | PSNR | SSIM | LPIPS |
|---|---|---|---|
| Classical baseline (bicubic + NLM) | 20.58 | 0.374 | 0.562 |
| **Stage A** (NAFNet, KLA data only) | 28.26 | 0.740 | 0.289 |
| **Stage B** (NAFNet, +external data, shipped) | 28.09 | 0.731 | **0.163** |

- AI gains **~7.5–7.7dB PSNR** over the classical baseline — not a
  marginal improvement over a naive approach.
- Stage A→B is a **real, statistically proven two-sided tradeoff**: LPIPS
  improves ~44% relative, PSNR/SSIM regress slightly — both directions
  independently significant (paired Wilcoxon, p<1e-44 for all three
  metrics, not just the improvement).
- **Inference: 76.4 ms/image on H100 SXM 80GB**, measured end-to-end on
  real hardware.
- Tested and **rejected** a Stage A+B ensemble as an alternative ship
  candidate: costs ~2x inference time, wins only 1/3 composite-scoring
  scenarios, loses the other 2 to Stage B alone — decision made not to
  switch. Reported as evidence we checked the alternative, not left
  unexamined.

Full tables, statistical tests, and every finding below:
[reports/ppt_metrics_table.md](ppt_metrics_table.md) — the single source
of truth for every number in this project.

## Honest limitations (checked, not hidden)

- **Only 128→256 trained/validated.** Directly tested at 256→512 (2× the
  trained resolution) — architecture ran correctly on all 15 test images,
  clearly beat baselines, confirming mechanical/architectural
  generalization. But the test's pseudo-512 ground truth has no real fine
  detail beyond a bicubic upscale, so real higher-resolution
  reconstruction *quality* remains unverified. Precise claim: "verified
  generalization to 256→512 (2×) — the same 2× factor as training, applied
  to an unseen input resolution," not "scale-agnostic."
- **Validation is an unsupervised-clustering OOD proxy**, not a confirmed
  source-based split — no real source labels exist in KLA's data. Real OOD
  performance is only confirmed against KLA's actual test set.
- **License risk, not hidden:** DIV2K states academic-research-only;
  Flickr2K's license is disputed upstream. Standard practice in SR
  research, and this is a non-commercial hackathon entry — but not
  "verified clean" for commercial use. Checked whether this is load-bearing
  (can't be answered from the trained checkpoint without retraining — a
  cheap DTD+SAR-only variant would answer it if it matters later).
- **Real bug found and fixed:** the classical fallback's NLM denoising
  silently never executed (missing `PyWavelets` dependency, swallowed by a
  broad exception handler) until caught during this rigor pass — fixed,
  verified, and a permanent regression test added so it can't silently
  recur.
- **Tiny inputs (≤8px/side) fall back to the classical path** — a real
  architectural constraint (reflect-pad requires pad < input size),
  caught safely by existing exception handling, not expected to matter at
  KLA's real 128×128 resolution.
- **SAR training data is domain-different from KLA's content** (terrain
  vs. semiconductor) — included for shared noise physics, not content
  similarity; a small rough check (10 terrain-like val images) showed no
  performance degradation, but n=10 is not proof.

## Where to look for more

| Question | File |
|---|---|
| Does it actually meet KLA's submission spec? | [submission/Phoenix/README.md](../submission/Phoenix/README.md), fresh hard-gate re-verification with evidence: [reports/FINAL_SUBMISSION_VERIFICATION.md](FINAL_SUBMISSION_VERIFICATION.md) |
| All metrics, statistical tests, every finding | [reports/ppt_metrics_table.md](ppt_metrics_table.md) |
| Is it robust to bad/adversarial input? | [tests/test_run_py_robustness.py](../tests/test_run_py_robustness.py) (25 tests, all real, all passing) |
| Full noise-physics derivation | [README.md](../README.md) |
| Is training reproducible? | [src/utils/reproducibility.py](../src/utils/reproducibility.py) — bit-identical checkpoints proven across 2 runs |
| Raw inference code | [run.py](../run.py) — fully self-contained, no internet, no external imports beyond pip packages |
