"""
Classical baseline (bicubic upsample + non-local-means denoise) evaluated
on the same 506 val/OOD-proxy images used for Stage A/B, for a direct
three-way comparison. Reuses run.py's actual classical_fallback() function
(the same code path run.py falls back to on model failure) rather than
reimplementing it, so the "classical baseline" number is guaranteed to
match what the real fallback would actually produce.
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import lpips
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.datasets.kla_dataset import KLAPairDataset  # noqa: E402

# import classical_fallback from run.py without executing its __main__ block
spec = importlib.util.spec_from_file_location("run_module", ROOT / "run.py")
run_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_module)
classical_fallback = run_module.classical_fallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--noisy-dir", type=Path, required=True)
    ap.add_argument("--split-csv", type=Path, default=Path("reports/source_clusters.csv"))
    ap.add_argument("--out-csv", type=Path, default=Path("reports/classical_baseline_val_per_image_metrics.csv"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_fn = lpips.LPIPS(net="alex").to(device)

    val_ds = KLAPairDataset(args.gt_dir, args.noisy_dir, args.split_csv, "val", augment=False)
    print(f"Running classical baseline on {len(val_ds)} val images...")

    rows = []
    for i in range(len(val_ds)):
        noisy_t, gt_t, fname = val_ds[i]
        noisy = noisy_t.squeeze(0).numpy()
        gt = gt_t.squeeze(0).numpy()

        pred = classical_fallback(noisy, scale=2)

        psnr = sk_psnr(gt, pred, data_range=1.0)
        ssim = sk_ssim(gt, pred, data_range=1.0)

        pred_t = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).to(device)
        gt_t2 = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0).to(device)
        pred_lp = pred_t.repeat(1, 3, 1, 1) * 2 - 1
        gt_lp = gt_t2.repeat(1, 3, 1, 1) * 2 - 1
        with torch.no_grad():
            lp = lpips_fn(pred_lp, gt_lp).item()

        rows.append({"file": fname, "psnr": psnr, "ssim": ssim, "lpips": lp})
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(val_ds)}")

    df = pd.DataFrame(rows).sort_values("file").reset_index(drop=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Saved {len(df)} rows to {args.out_csv}")
    print(f"Mean: PSNR={df['psnr'].mean():.3f} SSIM={df['ssim'].mean():.4f} LPIPS={df['lpips'].mean():.4f}")


if __name__ == "__main__":
    main()
