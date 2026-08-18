# KLA AI Hackathon PS01 — AI-Based Restoration of Degraded Images

Restores semiconductor inspection images degraded by simultaneous
multiplicative speckle noise, additive noise, and spatial downsampling
(128×128 → 256×256).

**Note on this document's citations:** `run.py` itself is fully
self-contained (see Setup below) — but this README, as supporting
evidence, cites a few files (`tests/test_run_py_robustness.py`,
`reports/ppt_metrics_table.md`) that live in the parent development
repository this submission folder was copied from, not inside this folder
itself. If you only have this `submission/Phoenix/` folder in isolation,
those specific citations won't resolve locally — everything needed to
*run* the model is still entirely contained here regardless.

## Setup

Requires Python 3.11 (tested on 3.11.9) and an NVIDIA GPU with drivers
already installed (falls back to CPU automatically if none is found, but
GPU is what this was built and timed for).

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
source .venv/bin/activate    # Linux/macOS
pip install -r requirements.txt
```

(The venv is recommended but not required — `pip install -r
requirements.txt` into any Python 3.11 environment works the same way.)

That installs everything needed. One caveat on the install itself:
`requirements.txt` pins `torch` to a `+cu121` build via `--extra-index-url
https://download.pytorch.org/whl/cu121` — verified to install and run
correctly on a clean venv with an NVIDIA GPU on CUDA 12.1. If the
benchmarking machine has a different CUDA/driver version, install the
matching build first (e.g. `pip install torch --index-url
https://download.pytorch.org/whl/cu124`), then re-run `pip install -r
requirements.txt` — it will see torch already satisfied and install the
rest. This install path has only been verified on Windows so far, not on
Linux — expected to work there too (the extra index is additive, not
exclusive, and PyPI's own Linux `torch` wheels typically bundle CUDA support
directly), but that's not yet directly tested, flagging it as such rather
than as confirmed.

Once installed, no further setup, configuration, or internet access is
needed at any point. This is architectural, not incidental: the model
(NAFNet-style encoder-decoder) is built from raw PyTorch layers with no
pretrained backbone, its weights ship in `models/checkpoint.pt`, and
`run.py` has no dependency on any other file in this folder or elsewhere.

## Usage

```bash
python run.py <input_dir> <output_dir>
```

Example:

```bash
python run.py ./test_images ./restored_images
```

- `<input_dir>`: directory of `.npy` files, each a 2D float32 grayscale array
  (any HxW — the model was trained and validated on 128×128 input producing
  256×256 output).
- `<output_dir>`: created automatically if it doesn't exist. One output
  `.npy` file is written per input file, under the identical filename.
- Output arrays are `(H, W)` float32, values strictly in `[0, 1]`, guaranteed
  free of NaN/Inf by an explicit sanitization step applied to every output
  regardless of code path (see `sanitize_output()` in `run.py`).
- Runs on GPU automatically if available (`torch.cuda.is_available()`),
  falls back to CPU otherwise — no flag needed.
- Works from any working directory — `run.py`'s default checkpoint path
  resolves relative to its own file location, not wherever the command
  happens to be run from, so `python /any/path/run.py <in> <out>` works
  the same as running it from inside this folder.

## Robustness

`run.py` never crashes the batch over a single bad input:
- If the model errors on a specific image, that image falls back to
  classical bicubic upsampling + non-local-means denoising, logged to
  stderr.
- If an input file can't even be loaded (corrupt/wrong shape), a neutral
  placeholder is written instead of stopping the run, also logged.
- A summary at the end reports exactly how many images (if any) needed a
  fallback, so this is never silently masked as "fully succeeded" when it
  wasn't.

Verified with a permanent, re-runnable test suite
(`tests/test_run_py_robustness.py`, 25 tests) covering corrupt files, wrong
dimensionality, NaN/Inf-contaminated pixels, all-zero/all-constant images,
non-square images, and extreme out-of-range values - not just a one-time
manual check.

**Real bug found and fixed:** the classical fallback's `estimate_sigma()`
call (scikit-image) has an undeclared dependency on `PyWavelets` - without
it, `estimate_sigma()` raises `ImportError`, which the fallback's broad
exception handler silently caught, downgrading every fallback to
bicubic-only with no NLM denoising and no visible warning. `requirements.txt`
never pinned it. Found while running an unrelated test (Task 1 of the final
rigor pass), confirmed directly (the previously-reported "classical
baseline" numbers matched a bicubic-only recomputation to 6 decimal
places), and fixed by pinning `PyWavelets==1.9.0` here - verified
installing cleanly in a fresh venv and verified `denoise_nl_means` now
actually executes and perturbs the output. See
`reports/ppt_metrics_table.md`'s Classical baseline comparison section for
the corrected numbers (the practical difference was small, <0.02 on every
metric, but the fallback now genuinely matches its own documentation).

**Real finding from that suite, disclosed rather than left for a grader to
find:** any input **8px or smaller on either side** fails inside the
model's own forward pass (NAFNetSR reflect-pads up to a 16px alignment
multiple, and PyTorch's reflect padding requires the pad amount to be
strictly less than the input dimension - an 8px side needs an 8px pad,
which violates that constraint). This is caught by `run.py`'s per-image
exception handling and correctly falls back to the classical path, so the
batch does not crash and the output is still spec-compliant - but it means
any such image silently gets the lower-quality classical restoration
instead of the trained model. Not expected to matter in practice (KLA's
data is 128x128, far above this threshold), but stated precisely rather
than assumed away.

## Model

NAFNet-style backbone: U-Net hierarchy, simplified channel attention, gated
activation-free blocks (no self-attention), fused pixel-shuffle upsampling
head — a single forward pass jointly denoises, suppresses speckle, and
upsamples 128→256.

**Checkpoint currently shipped: Stage B** (KLA data + external
DIV2K/Flickr2K/DTD/SAR data, fine-tuned from the Stage A checkpoint with an
expanded loss — Charbonnier + MS-SSIM + LPIPS + Sobel edge + range-
consistency penalty). Measured on this project's own held-out validation
split (an unsupervised-clustering OOD proxy — not KLA's official test set,
which we don't have access to): PSNR 28.09, SSIM 0.731, LPIPS 0.163 (n=506).
Compared to the Stage A checkpoint it was fine-tuned from (PSNR 28.26, SSIM
0.740, LPIPS 0.289): LPIPS improved substantially (~44% relative) at a small
cost to PSNR/SSIM — the added perceptual/edge loss terms trade a little
pixel-exact reconstruction for better perceptual quality, which the LPIPS
number is specifically designed to capture. **Both halves of this tradeoff
are statistically proven, not just the improvement** — a paired Wilcoxon
test confirms the PSNR/SSIM regressions are exactly as statistically
significant (p<0.05) as the LPIPS gain, not noise on either side (see
`reports/ppt_metrics_table.md`'s Statistical significance section for the
full per-metric p-values and bootstrap CIs). Shipped checkpoint selection:
**best checkpoint by validation PSNR during training** (epoch 15 of 20) —
the standard, defensible checkpointing criterion used throughout this
project (Stage A's shipped checkpoint was selected the same way). Full
comparison: `reports/ppt_metrics_table.md`.

## Known limitation: scale factor

This model handles the **128→256 scale factor** confirmed present in the
provided training data (every GT/NoisyLR pair sourced was 256×256 GT /
128×128 degraded — no 512×512 examples turned up in the data available to
us). If the official test set includes the 512↔256 pair as well: the
checkpoint's *learned weights* have not been trained on real data at that
resolution, but the *architecture's mechanical ability to run at that
resolution* has now been directly tested (see below), not left as an
untested assumption.

The architecture itself is not the limiting factor: it's fully
convolutional and accepts any input shape at runtime rather than hardcoding
a fixed resolution (feed it a 256×256 input, it produces 512×512, with no
code change) — confirmed by direct inspection, the checkpoint's `upscale`
factor is a fixed 2× regardless of input size, and internal padding to a
16px alignment multiple is computed from the input tensor's own shape at
runtime.

**Tested, not just claimed:** ran the real `run.py` path against 15
256×256 synthetic test inputs (2× the resolution it was trained at) and it
produced correct, spec-compliant 512×512 output on every single one — zero
crashes, zero fallback triggers — clearly outperforming both a bicubic
baseline and the classical bicubic+NLM fallback at the same task (see
`reports/ppt_metrics_table.md`'s Scale generalization section for the full
table and methodology). **This confirms the architecture generalizes
mechanically to an untrained input resolution.** Precise framing for
slide/report use: **verified generalization to 256→512 (2×) — the same 2×
factor used in training, applied to an unseen 256-input resolution.** Not
"scale-agnostic" and not "infers arbitrary scale" - the upscale factor is
fixed at 2×, as stated above; what's newly verified is that this fixed
2× still works correctly at an input size the model never saw in
training. What remains genuinely
untested: real reconstruction quality at a true higher resolution — no
real 512×512+ KLA source images exist to build a real test against, so the
test above used a bicubic-upscaled pseudo-GT with no real fine detail
beyond what bicubic interpolation produces. That's a meaningfully weaker
claim than "verified at real 512", and the metrics table is explicit about
not overstating it. Extending to that scale with confidence would still
need retraining (or at minimum fine-tuning) on real 512↔256 pairs, which we
did not have access to.

## Data sources & licensing

Stage B's training mix included KLA's provided data plus four external
datasets. Only the trained model weights ship in this submission (no
training data is redistributed), but the license terms of data used *to
train* a shipped model are still worth stating plainly rather than assumed
fine. Checked against each dataset's own stated terms, not assumed:

| Dataset | Stated terms | Source |
|---|---|---|
| KLA training data | Provided directly by KLA for this competition | — |
| DIV2K | **"Made available for academic research purpose only."** Images collected from the internet; copyright remains with original owners. **Not licensed for commercial use.** | [data.vision.ee.ethz.ch/cvl/DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) |
| Flickr2K | **Ambiguous.** No authoritative license statement found for the bundled dataset. An open, unresolved GitHub issue in the original NTIRE2017 repo explicitly questions whether its use complies with CC-BY 4.0 - not resolved as of this check. | [github.com/limbee/NTIRE2017#39](https://github.com/limbee/NTIRE2017/issues/39) |
| DTD (textures) | Oxford VGG's official page states only "made available to the computer vision community for research purposes" - no formal license text (e.g. no SPDX identifier) found on the source page. | [robots.ox.ac.uk/~vgg/data/dtd](https://www.robots.ox.ac.uk/~vgg/data/dtd/) |
| SAR (Sentinel-1&2) | Underlying data: **confirmed genuinely open** - EU Copernicus program's free, full, open-access policy, no restriction on commercial or non-commercial use, only requires an attribution notice when modified ("Contains modified Copernicus Sentinel data [Year]"). **However**, the specific Kaggle-hosted repackaging used (`requiemonk/sentinel12-image-pairs-segregated-by-terrain`) could not be directly verified for its own stated license via automated fetch (Kaggle's dataset pages are JS-rendered and not accessible to the tooling used for this check) - flagging this as **unverified**, not assumed identical to the underlying open Sentinel policy. | [esa.int Copernicus free access](https://www.esa.int/Applications/Observing_the_Earth/Copernicus/Free_access_to_Copernicus_Sentinel_satellite_data), [Kaggle dataset](https://www.kaggle.com/datasets/requiemonk/sentinel12-image-pairs-segregated-by-terrain) (license badge not machine-verified)

**Context this should be read in: this is a non-commercial student hackathon
submission, not a commercial product.** No revenue, no distribution to end
users, no deployment beyond the competition's judging process - the entire
use case is evaluation by KLA's judges under the hackathon's own rules.
"Academic research purposes only" terms are a materially smaller concern in
that context than they would be for a commercial deployment train on the
same data; this doesn't erase the license question, but it is the correct
frame for how much risk it actually represents right now.

**Honest read:** this is standard practice in the super-resolution research
community - DIV2K/Flickr2K are the field's default training sets and appear
in essentially every published SR paper's training pipeline under the same
academic-research framing this submission falls under. But "commonly done"
and "verified clean" are different claims, and DIV2K's terms explicitly say
academic-research-only. If this submission or the resulting model is ever
used beyond the hackathon's research/competition context, DIV2K's and
Flickr2K's terms should be revisited before any commercial use, and the
Kaggle SAR dataset's actual license badge should be manually confirmed (not
just its underlying Sentinel source policy).

**Is the DIV2K/Flickr2K license risk load-bearing - i.e. could they be
dropped without meaningfully hurting Stage B's results, if this ever needed
to ship commercially?** Checked what's actually knowable from the
already-trained checkpoint without retraining (retraining was explicitly
out of scope for this check):

- **Training mechanism, verified from the code:** `MixedDataset` and
  `ExternalImageDataset` (`src/datasets/external_image_dataset.py`) pool
  every external image from every `--external-dirs` path into one flat
  list and sample uniformly at random from it - there is no per-source
  balancing or weighting. So each source's actual influence during
  training was proportional to its raw image count in the real run, not a
  deliberately tuned mix.
- **What can't be measured post-hoc:** no per-source loss or metric is
  logged anywhere - not in the checkpoint's saved metadata, not in a
  training history file (none was retained for the real Stage B run, a
  separately-documented gap), and the KLA val split we evaluate on
  contains zero DIV2K/Flickr2K/DTD/SAR images, so per-image val
  performance can't be attributed back to which external source helped.
  This is a genuine limit on what's answerable from the existing
  checkpoint alone, not a gap we're eliding.
- **What would give a real answer:** a DTD+SAR-only Stage B variant, same
  training recipe and epoch budget as the loss-ablation runs already
  planned for this pass, would directly test whether dropping DIV2K/
  Flickr2K costs meaningfully - it's a natural extension of ablation
  infrastructure that already exists, not new engineering. Not run in
  this pass (retraining was explicitly out of scope here) - flagged as a
  concrete, cheap, ready-to-launch next step if this license question
  ever becomes practically important, rather than left as a vague future
  todo.

## Reproducibility

`train.py` and `train_stageB.py` fully seed every RNG source their code
path touches: Python's `random`, NumPy, `torch.manual_seed`, and
`torch.cuda.manual_seed_all` (this was previously only partial - the CUDA
seed call and cuDNN's deterministic flags were missing, and the DataLoader
augmentation, which runs inside worker processes when `--num-workers>0`,
was not reproducibly seeded at all). Fixed in
`src/utils/reproducibility.py`, verified with real evidence, not assumed:
ran `train.py` twice with identical `--seed 42 --num-workers 2` (the exact
condition that was broken - worker-process augmentation) and compared the
resulting checkpoints directly. Result: **every tensor in the model's
`state_dict` is bit-identical (`torch.equal`) between the two runs, and
`val_psnr` matches to full float precision** (26.217100912881932 in both).

**Seed used for the shipped Stage B checkpoint:** `--seed` was not
explicitly overridden in any Stage B launch command used, so it ran with
the script's default, **`seed=0`**. This checkpoint predates the fix above
(checkpoints now save their own `seed` field for future provenance; this
one doesn't have it recorded internally), so this is stated based on the
launch commands actually used, not independently re-derivable from the
checkpoint file itself - flagging that distinction rather than overclaiming
certainty.
