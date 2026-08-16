"""
Standalone inference script for submission/benchmarking.

Usage:
    python eval.py <input_dir> <output_dir> [--checkpoint checkpoints/stageA_best.pt]

Reads every .npy in input_dir (float32 grayscale, any HxW), restores it, and
writes the result to output_dir under the same filename.

Guardrails (never let one bad input crash the whole batch):
  - Output is clipped to [0, 1].
  - A cheap, low-blend fixed blur suppresses pixel-shuffle checkerboard
    periodicity without materially softening real detail.
  - If the model errors on a given image, falls back to classical bicubic
    upsampling + non-local-means denoising, logging the trigger.
  - If even loading the input file fails, writes a neutral placeholder
    instead of crashing the run - logged as a hard failure, not silent.
  - A final summary reports fallback/failure counts so this is never
    silently masked as "it worked".
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from skimage.transform import resize as sk_resize
from skimage.restoration import denoise_nl_means, estimate_sigma

from src.models.nafnet import NAFNetSR

DEFAULT_SHAPE = (256, 256)  # best-effort placeholder shape when input can't even be read


def load_model(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    upscale = ckpt.get("upscale", 2)
    model = NAFNetSR(img_channel=1, width=ckpt.get("width", 32), upscale=upscale)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, upscale


def suppress_checkerboard(y: torch.Tensor, blend: float = 0.15) -> torch.Tensor:
    kernel = torch.ones(1, 1, 3, 3, device=y.device, dtype=y.dtype) / 9.0
    blurred = F.conv2d(y, kernel, padding=1)
    return (1 - blend) * y + blend * blurred


def classical_fallback(noisy: np.ndarray, scale: int) -> np.ndarray:
    h, w = noisy.shape
    up = sk_resize(noisy, (h * scale, w * scale), order=3, mode="reflect", anti_aliasing=True)
    up = np.clip(up, 0.0, 1.0)
    try:
        sigma_est = float(np.mean(estimate_sigma(up)))
        denoised = denoise_nl_means(up, h=1.15 * sigma_est, fast_mode=True, patch_size=5, patch_distance=6)
    except Exception:
        denoised = up  # bicubic-only if NLM itself errors (e.g. degenerate constant image)
    return np.clip(denoised, 0.0, 1.0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir", type=Path)
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--checkpoint", type=Path, default=Path("checkpoints/stageA_best.pt"))
    ap.add_argument("--checkerboard-blend", type=float, default=0.15)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, upscale = load_model(args.checkpoint, device)

    files = sorted(args.input_dir.glob("*.npy"))
    print(f"Found {len(files)} input files. Device: {device}")

    load_failures, model_fallbacks, fallback_failures = [], [], []

    with torch.no_grad():
        for f in files:
            try:
                arr = np.load(f).astype(np.float32)
                if arr.ndim != 2:
                    raise ValueError(f"expected 2D grayscale array, got shape {arr.shape}")
            except Exception as e:
                print(f"ERROR: could not load {f.name} ({e}); writing placeholder, skipping", file=sys.stderr)
                load_failures.append(f.name)
                out = np.full(DEFAULT_SHAPE, 0.5, dtype=np.float32)
                np.save(args.output_dir / f.name, out)
                continue

            try:
                arr_clean = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
                x = torch.from_numpy(arr_clean).unsqueeze(0).unsqueeze(0).to(device)
                y = model(x)
                y = suppress_checkerboard(y, blend=args.checkerboard_blend)
                y = y.clamp(0.0, 1.0)
                out = y.squeeze(0).squeeze(0).cpu().numpy()
                if not np.all(np.isfinite(out)):
                    raise RuntimeError("model produced non-finite output")
            except Exception as e:
                model_fallbacks.append(f.name)
                print(f"WARNING: model failed on {f.name} ({e}); falling back to bicubic+NLM", file=sys.stderr)
                try:
                    out = classical_fallback(np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0), scale=upscale)
                except Exception as e2:
                    fallback_failures.append(f.name)
                    print(f"ERROR: fallback also failed on {f.name} ({e2}); writing placeholder", file=sys.stderr)
                    h, w = arr.shape
                    out = np.full((h * upscale, w * upscale), 0.5, dtype=np.float32)

            np.save(args.output_dir / f.name, out)

    print(f"Wrote {len(files)} restored images to {args.output_dir}")
    print(f"Load failures: {len(load_failures)} {load_failures}")
    print(f"Classical-fallback triggers: {len(model_fallbacks)} {model_fallbacks}")
    print(f"Fallback-also-failed: {len(fallback_failures)} {fallback_failures}")
    if not load_failures and not model_fallbacks:
        print("Model succeeded on every image, no fallback triggered.")


if __name__ == "__main__":
    main()
