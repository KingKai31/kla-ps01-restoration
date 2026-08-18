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
  Correction to an earlier version of this note: the architecture's `upscale`
  factor is a **fixed 2× baked into the checkpoint**, not inferred from input
  shape — what's inferred at runtime is only the input's H/W, which the
  model pads to a 16px alignment multiple before its fixed-2× head. So this
  is a resolution-generalization question (does 256→512 work, given the
  model only saw 128→256), not a ratio question (both pairs are 2×). Tested
  directly, not just claimed forward-compatible: ran the real `run.py` path
  against 15 synthetic 256×256 inputs and it produced correct, spec-compliant
  512×512 output on every one, clearly beating both a bicubic and a
  bicubic+NLM baseline at the task - see `reports/ppt_metrics_table.md`'s
  Scale generalization section. That confirms mechanical/architectural
  generalization; it does **not** confirm real reconstruction quality at a
  true higher resolution, since no real 512×512+ KLA source data exists to
  test against (the test's pseudo-GT is a bicubic upscale with no real fine
  detail beyond what bicubic already produces - stated explicitly in the
  table so the result isn't overread). Precise framing for slide/report
  use: **verified generalization to 256→512 (2×) - the same 2× factor used
  in training, applied to an unseen 256-input resolution.** Not
  "scale-agnostic," not "infers arbitrary scale."
- **Any input 8px or smaller on either side falls back to the classical
  (lower-quality) restoration path instead of the trained model.** Found
  via `tests/test_run_py_robustness.py` (formal, re-runnable suite - see
  below), not previously known: `NAFNetSR`'s forward pass reflect-pads
  each input up to a 16px alignment multiple, and PyTorch's reflect
  padding requires the pad amount to be strictly less than the input
  dimension - an 8px side needs an 8px pad, which violates that
  constraint and raises inside the model. `run.py`'s per-image exception
  handling catches this and correctly falls back to classical bicubic+NLM,
  so the batch never crashes and the output stays spec-compliant, but
  quality silently degrades for any image at or below this threshold. Not
  expected to matter for KLA's actual 128x128 data (far above the
  threshold), but disclosed precisely rather than left undiscovered.
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
run.py                      standalone inference script: run.py <input_dir> <output_dir> (renamed from eval.py)
train.py                    Stage A training entry point
train_stageB.py             Stage B training entry point (fine-tunes from a Stage A checkpoint)
submission/Phoenix/         the actual submission package - self-contained run.py, requirements.txt, README.md, models/
requirements.txt            pip freeze from the training environment
checkpoints/                trained weights (not committed - see checkpoints/README)
outputs/                    restored images produced by run.py
tests/                      formal pytest suite (run.py robustness/edge-case coverage,
                             including the PyWavelets dependency-gap regression check)
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
CPU-only fresh venv (see reports/ for the run.py verification log). Not yet
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

## Data sources & licensing

Stage B's training mix included KLA's provided data plus four external
datasets (DIV2K, Flickr2K, DTD, SAR/Sentinel-1&2). Checked against each
dataset's own stated terms, not assumed fine:

| Dataset | Stated terms | Source |
|---|---|---|
| DIV2K | **"Academic research purpose only."** Not licensed for commercial use; copyright remains with original owners. | [data.vision.ee.ethz.ch/cvl/DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) |
| Flickr2K | **Ambiguous** - no authoritative license statement found; an open, unresolved GitHub issue in the original NTIRE2017 repo questions whether its use even complies with CC-BY 4.0. | [github.com/limbee/NTIRE2017#39](https://github.com/limbee/NTIRE2017/issues/39) |
| DTD | Oxford VGG's page states only "made available... for research purposes" - no formal license text found. | [robots.ox.ac.uk/~vgg/data/dtd](https://www.robots.ox.ac.uk/~vgg/data/dtd/) |
| SAR (Sentinel-1&2) | Underlying data confirmed genuinely open (EU Copernicus free/full/open policy, no commercial restriction). The specific Kaggle repackaging used could **not** be directly verified for its own license badge (JS-rendered page, not accessible to automated fetch) - flagged as unverified. | [esa.int](https://www.esa.int/Applications/Observing_the_Earth/Copernicus/Free_access_to_Copernicus_Sentinel_satellite_data), [Kaggle dataset](https://www.kaggle.com/datasets/requiemonk/sentinel12-image-pairs-segregated-by-terrain) |

**Context: this is a non-commercial student hackathon submission, not a
commercial product** - no revenue, no end-user distribution, evaluation-only
use under the competition's own rules. That's the correct frame for how
much risk "academic research purposes only" terms actually represent right
now; it doesn't erase the question, but it bounds it.

Checked (without retraining) whether DIV2K/Flickr2K's license risk is
load-bearing - i.e. whether dropping them would cost Stage B meaningfully:
the training code samples every external image uniformly regardless of
source (no per-source balancing), and no per-source metric is logged
anywhere, so precise attribution isn't answerable from the existing
checkpoint alone. A DTD+SAR-only variant (reusing the loss-ablation
infrastructure already built) would give a real answer - flagged as a
ready, cheap next step, not run in this pass since retraining was out of
scope. Full detail: `submission/Phoenix/README.md`'s Data sources &
licensing section (same content, submission-package copy).

## Reproducibility

`train.py`/`train_stageB.py` fully seed Python's `random`, NumPy, torch,
CUDA, and cuDNN's determinism flags, plus DataLoader worker processes
(previously the worker-process gap meant augmentation wasn't reproducibly
seeded when `--num-workers>0` - a real, now-fixed gap, not previously
verified). Proof: two identical runs of `train.py` with `--seed 42
--num-workers 2` produced a **bit-identical model `state_dict`**
(`torch.equal` on every tensor) and matching `val_psnr` to full float
precision. See `src/utils/reproducibility.py` and
`submission/Phoenix/README.md`'s Reproducibility section (including the
shipped Stage B checkpoint's actual seed, `seed=0`, and the honest caveat
that it predates this fix and isn't independently re-derivable from the
checkpoint file itself) for full detail.
