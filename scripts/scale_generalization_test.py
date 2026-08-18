"""
Task 1 (final rigor pass): tests whether the shipped checkpoint - trained
and validated ONLY at 128->256 - produces sensible output when run.py is
fed a 256x256 input instead (i.e. at 2x the absolute resolution it was
trained at). The architecture is fully convolutional and always applies a
fixed 2x upsample (baked into the checkpoint's `upscale` field, confirmed
via direct inspection - it does not "infer" a ratio, it infers only the
input H/W to pad/crop around), so this specifically tests resolution
generalization, not ratio generalization (128->256 and 256->512 are both
exactly 2x).

METHODOLOGY LIMITATION - stated up front, not buried in results: KLA has
no real 512x512 (or higher) source images available to us. This script
builds a "pseudo-512" ground truth by bicubic-upscaling real 256x256 KLA
GT images. That pseudo-GT contains NO real fine detail beyond what a
clean bicubic upscale can produce - it cannot test whether the model
recovers genuine high-frequency structure at a real higher resolution.
What it CAN validly test: does the model's code path handle a
differently-shaped input without crashing/producing garbage, and does its
output still meaningfully outperform a naive upscale of the SAME
degraded input at the SAME pseudo-scale. That is a real, useful, but
narrower claim than "verified at real 512x512" - do not overstate it.

The degraded 256x256 test input is built with the exact same validated
noise model used throughout this project (src/datasets/synthetic_degrade.py's
SpeckleAdditiveDegrader), applied with factor=2 to the pseudo-512 GT -
identical physics to how the real 128x128 NoisyLR data relates to the
real 256x256 GT, just one octave up.
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import lpips
import numpy as np
import pandas as pd
import torch
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim
from skimage.transform import resize as sk_resize

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from src.datasets.synthetic_degrade import SpeckleAdditiveDegrader  # noqa: E402


def _load_run_module():
    spec = importlib.util.spec_from_file_location("run_py_under_test", REPO_ROOT / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--n-images", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "models" / "checkpoint.pt")
    ap.add_argument("--reports-dir", type=Path, default=Path("reports"))
    ap.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    files = sorted(args.gt_dir.glob("*.npy"))
    chosen = [files[i] for i in rng.choice(len(files), size=args.n_images, replace=False)]

    degrader = SpeckleAdditiveDegrader(args.reports_dir, seed=args.seed)

    work_dir = args.out_dir / "_scale_gen_test_tmp"
    input_dir = work_dir / "inputs_256"
    output_dir = work_dir / "outputs_512"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    pseudo_gt512 = {}
    for f in chosen:
        gt256 = np.load(f).astype(np.float64)
        assert gt256.shape == (256, 256), f"expected 256x256 KLA GT, got {gt256.shape} for {f.name}"
        gt512 = sk_resize(gt256, (512, 512), order=3, mode="reflect", anti_aliasing=False)
        gt512 = np.clip(gt512, 0.0, 1.0).astype(np.float32)
        pseudo_gt512[f.name] = gt512

        noisy256 = degrader.degrade(gt512, factor=2)  # box-downsample 512->256, apply noise model
        np.save(input_dir / f.name, noisy256.astype(np.float32))

    print(f"Built {len(chosen)} synthetic 512->256 test pairs in {input_dir}")

    # --- run the REAL run.py against these 256x256 inputs (not a mock) ---
    run_module = _load_run_module()
    argv_backup = sys.argv
    sys.argv = ["run.py", str(input_dir), str(output_dir), "--checkpoint", str(args.checkpoint)]
    try:
        run_module.main()
    finally:
        sys.argv = argv_backup

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_fn = lpips.LPIPS(net="alex").to(device)

    def lpips_score(pred: np.ndarray, gt: np.ndarray) -> float:
        p = torch.from_numpy(pred).float().unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(device) * 2 - 1
        g = torch.from_numpy(gt).float().unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(device) * 2 - 1
        with torch.no_grad():
            return lpips_fn(p, g).item()

    rows = []
    for f in chosen:
        gt512 = pseudo_gt512[f.name]
        noisy256 = np.load(input_dir / f.name)
        model_out = np.load(output_dir / f.name).astype(np.float32)
        assert model_out.shape == (512, 512), f"expected model to produce 512x512, got {model_out.shape}"

        bicubic512 = np.clip(
            sk_resize(noisy256, (512, 512), order=3, mode="reflect", anti_aliasing=False), 0.0, 1.0
        ).astype(np.float32)

        classical512 = run_module.classical_fallback(noisy256, scale=2)

        rows.append({
            "file": f.name,
            "model_psnr": sk_psnr(gt512, model_out, data_range=1.0),
            "model_ssim": sk_ssim(gt512, model_out, data_range=1.0),
            "model_lpips": lpips_score(model_out, gt512),
            "bicubic_psnr": sk_psnr(gt512, bicubic512, data_range=1.0),
            "bicubic_ssim": sk_ssim(gt512, bicubic512, data_range=1.0),
            "bicubic_lpips": lpips_score(bicubic512, gt512),
            "classical_psnr": sk_psnr(gt512, classical512, data_range=1.0),
            "classical_ssim": sk_ssim(gt512, classical512, data_range=1.0),
            "classical_lpips": lpips_score(classical512, gt512),
        })

    df = pd.DataFrame(rows)
    out_csv = args.out_dir / "scale_generalization_256to512_test.csv"
    df.to_csv(out_csv, index=False)

    print(f"\nSaved per-image results to {out_csv}\n")
    print(df.to_string(index=False))
    print(f"\nMeans across {len(df)} synthetic 512-scale test images:")
    print(f"  Model (run.py, trained only at 128->256): "
          f"PSNR={df['model_psnr'].mean():.2f}  SSIM={df['model_ssim'].mean():.4f}  LPIPS={df['model_lpips'].mean():.4f}")
    print(f"  Bicubic baseline (same input, no denoise):  "
          f"PSNR={df['bicubic_psnr'].mean():.2f}  SSIM={df['bicubic_ssim'].mean():.4f}  LPIPS={df['bicubic_lpips'].mean():.4f}")
    print(f"  Classical fallback (bicubic+NLM, run.py's actual fallback path): "
          f"PSNR={df['classical_psnr'].mean():.2f}  SSIM={df['classical_ssim'].mean():.4f}  LPIPS={df['classical_lpips'].mean():.4f}")


if __name__ == "__main__":
    main()
