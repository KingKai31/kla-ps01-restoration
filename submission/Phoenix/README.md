# KLA AI Hackathon PS01 — AI-Based Restoration of Degraded Images

Restores semiconductor inspection images degraded by simultaneous
multiplicative speckle noise, additive noise, and spatial downsampling
(128×128 → 256×256).

## Setup

```bash
pip install -r requirements.txt
```

No other setup is needed. `requirements.txt` pins `torch` to a `+cu121`
build via `--extra-index-url https://download.pytorch.org/whl/cu121` —
verified to install and run correctly on a clean venv with an NVIDIA GPU on
CUDA 12.1. If the benchmarking machine has a different CUDA/driver version,
install the matching build first (e.g. `pip install torch --index-url
https://download.pytorch.org/whl/cu124`), then re-run `pip install -r
requirements.txt` — it will see torch already satisfied and install the
rest.

The model runs entirely from local files — no internet access, no API keys,
no additional downloads at any point. This is architectural, not incidental:
the model (NAFNet-style encoder-decoder) is built from raw PyTorch layers
with no pretrained backbone, its weights ship in `models/checkpoint.pt`, and
`run.py` has no dependency on any other file in this folder or elsewhere.

## Usage

```bash
python run.py <input_dir> <output_dir>
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

**Checkpoint currently shipped: Stage A** (KLA training data only). Measured
on the held-out OOD-proxy validation split: PSNR 28.26, SSIM 0.740, LPIPS
0.289 (n=506). Stage B (KLA data + external DIV2K/Flickr2K/DTD/SAR data,
fine-tuned from this Stage A checkpoint with an expanded loss) is in
progress; if it completes and validates before submission, its checkpoint
replaces this one at `models/checkpoint.pt` with no other change to this
folder.
