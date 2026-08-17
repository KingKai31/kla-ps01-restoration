# KLA AI Hackathon PS01 — AI-Based Restoration of Degraded Images

Restores semiconductor inspection images degraded by simultaneous
multiplicative speckle noise, additive noise, and spatial downsampling
(128×128 → 256×256).

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
number is specifically designed to capture. This is the checkpoint with the
highest val PSNR seen during Stage B training (epoch 15 of 20), not
necessarily the literal final epoch. Full comparison:
`reports/ppt_metrics_table.md`.
