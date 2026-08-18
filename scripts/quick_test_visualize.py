"""
Quick local visual test: run any single image through the ACTUAL shipped
run.py inference path and save a side-by-side comparison image. For our
own confidence-checking and pulling fresh PPT examples - not for judges
(they use run.py directly, per spec).

Deliberately does not reimplement model loading/inference: writes the
prepared input to a temp directory and invokes run.py's real main()
(imported by file path, same technique used in tests/test_run_py_robustness.py
and scripts/scale_generalization_test.py) so this can never silently drift
from what the actual submission does.

Usage:
    python scripts/quick_test_visualize.py <input_path> [gt_path]

<input_path>: any common image format (png/jpg/jpeg/bmp/tif) or .npy.
    If not already a 128x128 grayscale .npy, it's converted to grayscale
    and resized to 128x128 (matching real input shape) before inference.
gt_path (optional): ground-truth image (same format flexibility). If
    given, prints PSNR/SSIM/LPIPS for this specific image and adds a
    third panel to the comparison figure.

Output: reports/figures/quick_test_<timestamp>.png
"""
import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.transform import resize as sk_resize

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from src.datasets.synthetic_degrade import to_unit_grayscale  # noqa: E402

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _load_run_module():
    spec = importlib.util.spec_from_file_location("run_py_under_test", REPO_ROOT / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_and_prepare(path: Path, target_shape: tuple) -> np.ndarray:
    """Loads any supported image/array format and returns a float32 array
    in [0,1] at target_shape. If the file is already a .npy at exactly
    target_shape, passed through unchanged - otherwise converted to
    grayscale (to_unit_grayscale) and resized."""
    if path.suffix.lower() == ".npy":
        raw = np.load(path)
        if raw.ndim == 2 and raw.shape == target_shape:
            return raw.astype(np.float32)
        normalized = to_unit_grayscale(raw)
    else:
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {path.suffix} (expected .npy or {IMAGE_EXTENSIONS})")
        with Image.open(path) as im:
            normalized = to_unit_grayscale(np.array(im))

    if normalized.shape != target_shape:
        normalized = sk_resize(normalized, target_shape, order=3, mode="reflect", anti_aliasing=True)
        normalized = np.clip(normalized, 0.0, 1.0).astype(np.float32)
    return normalized


def upsample_for_display(img: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Purely visual upsample so the low-res input can sit side-by-side
    with the higher-res output at the same panel size - does not touch
    the actual inference pipeline."""
    if img.shape == target_shape:
        return img
    return np.clip(sk_resize(img, target_shape, order=0, mode="reflect", anti_aliasing=False), 0.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_path", type=Path)
    ap.add_argument("gt_path", type=Path, nargs="?", default=None)
    ap.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "models" / "checkpoint.pt")
    ap.add_argument("--out-dir", type=Path, default=Path("reports/figures"))
    args = ap.parse_args()

    if not args.input_path.exists():
        raise FileNotFoundError(f"Input not found: {args.input_path}")

    print(f"Loading and preparing input: {args.input_path}")
    prepared_input = load_and_prepare(args.input_path, target_shape=(128, 128))
    print(f"  Prepared shape: {prepared_input.shape}, range [{prepared_input.min():.3f}, {prepared_input.max():.3f}]")

    work_dir = args.out_dir / "_quick_test_tmp"
    input_dir = work_dir / "input"
    output_dir = work_dir / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    for f in input_dir.glob("*.npy"):
        f.unlink()
    for f in output_dir.glob("*.npy"):
        f.unlink()

    np.save(input_dir / "quick_test.npy", prepared_input)

    print(f"Running the real run.py path (checkpoint: {args.checkpoint})...")
    run_module = _load_run_module()
    argv_backup = sys.argv
    sys.argv = ["run.py", str(input_dir), str(output_dir), "--checkpoint", str(args.checkpoint)]
    try:
        run_module.main()
    finally:
        sys.argv = argv_backup

    restored = np.load(output_dir / "quick_test.npy").astype(np.float32)
    print(f"  Restored shape: {restored.shape}, range [{restored.min():.3f}, {restored.max():.3f}]")

    gt = None
    metrics = None
    if args.gt_path is not None:
        if not args.gt_path.exists():
            raise FileNotFoundError(f"Ground-truth path not found: {args.gt_path}")
        print(f"Loading ground truth: {args.gt_path}")
        gt = load_and_prepare(args.gt_path, target_shape=restored.shape)

        from skimage.metrics import peak_signal_noise_ratio as sk_psnr
        from skimage.metrics import structural_similarity as sk_ssim
        import torch
        import lpips

        psnr = sk_psnr(gt, restored, data_range=1.0)
        ssim = sk_ssim(gt, restored, data_range=1.0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lpips_fn = lpips.LPIPS(net="alex").to(device)
        with torch.no_grad():
            p = torch.from_numpy(restored).float().unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(device) * 2 - 1
            g = torch.from_numpy(gt).float().unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(device) * 2 - 1
            lpips_val = lpips_fn(p, g).item()

        metrics = {"psnr": psnr, "ssim": ssim, "lpips": lpips_val}
        print(f"\nMetrics vs ground truth: PSNR={psnr:.2f}  SSIM={ssim:.4f}  LPIPS={lpips_val:.4f}\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_panels = 3 if gt is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 5))

    input_display = upsample_for_display(prepared_input, restored.shape)
    axes[0].imshow(input_display, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"Input ({prepared_input.shape[0]}x{prepared_input.shape[1]}, upsampled for display)", fontsize=10)
    axes[0].axis("off")

    restored_title = "Restored (run.py output)"
    if metrics:
        restored_title += f"\nPSNR={metrics['psnr']:.2f} SSIM={metrics['ssim']:.4f} LPIPS={metrics['lpips']:.4f}"
    axes[1].imshow(restored, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(restored_title, fontsize=10)
    axes[1].axis("off")

    if gt is not None:
        axes[2].imshow(gt, cmap="gray", vmin=0, vmax=1)
        axes[2].set_title("Ground truth", fontsize=10)
        axes[2].axis("off")

    fig.suptitle(f"Quick test: {args.input_path.name}", fontsize=12, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = args.out_dir / f"quick_test_{timestamp}.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)

    print(f"Saved comparison image to {out_path}")


if __name__ == "__main__":
    main()
