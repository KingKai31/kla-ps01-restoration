# KLA AI Hackathon — PS01: AI-Based Restoration of Degraded Images

Restores semiconductor inspection images degraded by simultaneous speckle
noise, Gaussian blur, and spatial downsampling (512→256 or 256→128),
scored on SSIM / pSNR / LPIPS and end-to-end inference time on an H100.

## Status

**Phase 1 (noise-physics analysis) complete, Phase 2 not started.** No model
has been trained yet. Measured on all 3200 local train pairs
(`reports/phase1_decomposition_summary.json`):

The degradation is **not** pure multiplicative speckle — NoisyLR pixel values
go negative, which multiplicative noise on a non-negative GT cannot produce.
Decomposed as `NoisyLR = GT_down * M + A`:

- Multiplicative speckle `M ~ Gamma(L, 1/L)`: L mean 28.0, median 27.9,
  std 9.1 (range 3.8–50.9) across sources — varies a lot, randomize across
  this full range when generating synthetic data, don't use one fixed L.
- Additive noise `A ~ N(mu, sigma)`: mu approx 0 (mean -0.0009), sigma mean
  0.0073 (median 0.0072, std 0.0024 across sources). Approximately Gaussian
  but with mild left skew (-0.32) and heavier-than-Gaussian tails (excess
  kurtosis 2.69) — Shapiro-Wilk formally rejects exact normality, treat as
  "Gaussian-like, not exactly Gaussian."
- Decomposition validated by comparing predicted vs measured residual
  variance across GT brightness bins: matches within ~1% for GT in
  [0.02, 0.5], predicted overshoots by up to ~15% in the brightest bins
  (0.5-1.0) — the two-parameter model is a good but imperfect fit.
- **Resolved: no separate blur kernel is modeled.** The FFT-based blur
  estimate couldn't cleanly isolate a blur component (injected noise
  dominates the spectrum). Insurance check: generated synthetic NoisyLR with
  the multiplicative+additive model only (no blur), compared against real
  NoisyLR on 400 held-out pairs (`reports/insurance_check_summary.json`,
  `reports/figures/insurance_check_*.png`). Central noise level matches
  (KS-test on per-image std, p=0.64); visual and full-spectrum comparison
  match well. KLA's "Gaussian noise" degradation is explained by the
  combined multiplicative+additive effect, not a separate blur kernel.

### Known minor approximations (documented, not blocking)

- Additive noise is modeled as Gaussian; the real residual has mildly
  heavier tails (excess kurtosis ~2.7), which shows up as real images
  occasionally hitting more extreme min/max pixel values than synthetic
  ones. Revisit only if OOD performance on high-noise sources is weak.
- Synthetic noise is applied per-pixel i.i.d. at the downsampled
  resolution; the insurance check's FFT comparison shows real noise has
  slightly more mid-frequency spatial correlation (~10-22% power gap in a
  mid-frequency band, closing back up near both DC and Nyquist) - plausibly
  because KLA injects noise before downsampling, which box-averaging would
  mildly correlate. Not modeled; judged low-impact.
- A brightness-dependent correction is applied to the multiplicative
  noise's effective L (fit directly from the heteroscedasticity data,
  `L_eff(x) ~= 16.8 + 23.1*x`) to correct Phase 1's ~15% high-brightness
  variance overprediction.

## Known limitations

- **The train/val split is not a real source split — it's an unsupervised
  clustering proxy, and that's a meaningfully weaker claim.** No source/
  category labels exist in the KLA data (just sequential filenames), so
  `scripts/cluster_sources.py` clusters GT images by cheap statistical
  features (intensity histogram, gradient energy, coarse thumbnail) and
  holds out whole clusters for validation. This approximates an OOD split
  better than a random split does, but it is **not** guaranteed to align
  with KLA's actual acquisition sources, and val-split metrics should be
  described as "OOD-proxy validation based on unsupervised clustering,"
  never as confirmed OOD generalization. Real OOD performance is only
  confirmed once evaluated against KLA's actual released test set.
- **Local training hardware (RTX 3050, 4GB VRAM) is a real constraint, not
  a footnote.** Stage A (KLA-only, ~3200 pairs) fits at batch_size=4;
  batch_size=8 OOM'd. Phase 3 adds substantially more external data
  (DIV2K/Flickr2K/DTD/SAR) - training that locally on this GPU is expected
  to be slow or infeasible. A cloud GPU (Colab Pro / Kaggle / rented A100)
  needs to be lined up before Phase 3 starts; this is a manual step (account
  + billing), not something automatable here. Before Stage B training,
  also sanity-check one training step at a larger batch size on whatever
  cloud GPU is used - the goal is confirming the code actually benefits
  from more VRAM, not just that it survives on 4GB, since final eval runs
  on an H100.
- **Only the 256→128 scale factor is present in the currently available
  training data.** The problem statement describes two degradation pairs
  (512→256 and 256→128), but every GT/NoisyLR pair sourced so far is
  256×256 GT / 128×128 degraded — no 512×512 examples have turned up locally.
  Phase 2's architecture infers the target upsampling factor from the input
  tensor's shape at runtime (rather than hardcoding a fixed 2x/4x factor), so
  it is forward-compatible if 512↔256 data is released later, but it has
  only been trained/validated on 256↔128 so far.
- **The external-dataset case-duplicate dedup logic
  (`src/datasets/external_image_dataset.py`) exists but was never exercised
  by the actual Stage B run.** It guards against double-counting images
  when a dataset ships both a canonical folder (e.g. `DIV2K_train_HR`) and
  a lowercase mirror of the same content - confirmed to exist as a
  possibility on the attached DIV2K Kaggle dataset. In practice, the
  `--external-dirs` path used for the real Stage B run was already the
  exact canonical folder (`.../DIV2K_train_HR/DIV2K_train_HR`), not a
  lowercase duplicate, so there was no collision for the dedup logic to
  resolve on this run - **the correct folder path was used explicitly**,
  this is not a claim that the dedup logic itself was verified in
  production. It remains unit-tested against mock objects
  (`scripts/_test_dedup_logic.py`) and end-to-end on a non-colliding real
  tree, but the actual case-collision resolution path has still never run
  against real duplicate directories (can't be built on this dev machine's
  Windows/NTFS filesystem either - case-insensitive, so `DIV2K_train_HR`
  and `div2k_train_hr` resolve to the same directory here). State this
  precisely if it comes up: the run avoided the ambiguity by using the
  right path, not by the dedup logic proving itself.

## Repo layout

```
scripts/analyze_noise.py   Phase 1: per-file stats, Gamma speckle fit, FFT blur estimate
src/                        model / dataset / loss code (Phase 2+)
eval.py                     standalone inference script: eval.py <input_dir> <output_dir>
train.py                    training entry point, reproduces the submitted checkpoint
requirements.txt            pip freeze from the training environment
checkpoints/                trained weights (not committed - see checkpoints/README)
outputs/                    restored images produced by eval.py
reports/                    Phase 1 analysis outputs, metrics tables, figures
data/                       not committed - see data/README.md for expected layout
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

`requirements.txt` pins `torch`/`torchvision` to `+cu121` builds and includes
a `--extra-index-url https://download.pytorch.org/whl/cu121` line so a plain
`pip install -r requirements.txt` resolves them - without that line, pip
fails outright (`+cu121`-tagged wheels don't exist on default PyPI), which is
exactly what happened the first time this was tested end-to-end in a fresh
venv. **If KLA's H100 benchmarking environment has a different CUDA/driver
version than cu121, this install may still fail or silently fall back to a
mismatched build.** Fallback: install the matching build manually first,
e.g. `pip install torch torchvision --index-url
https://download.pytorch.org/whl/cu124` (or `cu118`, or CPU-only
`https://download.pytorch.org/whl/cpu`), then `pip install -r
requirements.txt` for the rest - it will see torch/torchvision already
satisfied and skip them. Regenerate via `python
scripts/freeze_requirements.py`, not a bare `pip freeze`, or this line gets
silently dropped again.

Verified so far: fresh-venv install on cu121 (this dev machine) and a
CPU-only fresh venv (see reports/ for the eval.py verification log). Not yet
tested against a different CUDA minor version (e.g. cu124/cu126) - flagging
this as unverified, not as working.

## Data

See [data/README.md](data/README.md). Training pairs (GT + NoisyLR) are not
yet the official KLA release — currently using a local sample set while
waiting on the official paired training data and test set per the hackathon
schedule.

## Reproducing Phase 1 analysis

```bash
python scripts/analyze_noise.py \
  --test-noisy-dir <path to degraded-only test samples> \
  --train-gt-dir <path to paired GT> \
  --train-noisy-dir <path to paired NoisyLR> \
  --out-dir reports
```
