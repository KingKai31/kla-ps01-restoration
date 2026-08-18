"""
Task 3 (final rigor pass): given the statistically-significant PSNR/SSIM
regression Stage B traded for its LPIPS gain, checks whether a simple
Stage A + Stage B ensemble (average the two models' raw output tensors,
then apply the same post-processing run.py itself uses - checkerboard
suppression, clamp, sanitize) recovers some of Stage A's quality while
keeping most of Stage B's LPIPS improvement. Zero new training - both
checkpoints already exist.

Averaging happens BEFORE checkerboard suppression/clamping (i.e. on the
models' raw forward-pass output), matching how an ensemble is normally
built - both models see the same input, run independently, and their
predictions are combined before any single-model post-processing.
"""
import argparse
import sys
from pathlib import Path

import lpips
import numpy as np
import pandas as pd
import torch
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from src.datasets.kla_dataset import KLAPairDataset  # noqa: E402
from src.models.nafnet import NAFNetSR  # noqa: E402

sys.path.insert(0, str(REPO_ROOT))
import importlib.util
spec = importlib.util.spec_from_file_location("run_module", REPO_ROOT / "run.py")
run_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_module)
suppress_checkerboard = run_module.suppress_checkerboard


def load_model(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = NAFNetSR(img_channel=1, width=ckpt.get("width", 32), upscale=ckpt.get("upscale", 2)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--noisy-dir", type=Path, required=True)
    ap.add_argument("--split-csv", type=Path, default=Path("reports/source_clusters.csv"))
    ap.add_argument("--stageA-checkpoint", type=Path, default=Path("checkpoints/stageA_best.pt"))
    ap.add_argument("--stageB-checkpoint", type=Path, default=Path("models/checkpoint.pt"))
    ap.add_argument("--out-csv", type=Path, default=Path("reports/ensemble_val_per_image_metrics.csv"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelA = load_model(args.stageA_checkpoint, device)
    modelB = load_model(args.stageB_checkpoint, device)
    lpips_fn = lpips.LPIPS(net="alex").to(device)

    val_ds = KLAPairDataset(args.gt_dir, args.noisy_dir, args.split_csv, "val", augment=False)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)
    print(f"Running Stage A, Stage B, and their ensemble on {len(val_ds)} val images...")

    rows = []
    with torch.no_grad():
        for noisy, gt, fnames in val_loader:
            noisy, gt = noisy.to(device), gt.to(device)

            raw_A = modelA(noisy)
            raw_B = modelB(noisy)
            raw_ens = (raw_A + raw_B) / 2.0  # average BEFORE post-processing

            pred_A = suppress_checkerboard(raw_A).clamp(0.0, 1.0)
            pred_B = suppress_checkerboard(raw_B).clamp(0.0, 1.0)
            pred_ens = suppress_checkerboard(raw_ens).clamp(0.0, 1.0)

            def lpips_batch(pred):
                p = pred.repeat(1, 3, 1, 1) * 2 - 1
                g = gt.repeat(1, 3, 1, 1) * 2 - 1
                lp = lpips_fn(p, g).squeeze()
                return lp.cpu().tolist() if lp.dim() > 0 else [lp.item()]

            lpA, lpB, lpE = lpips_batch(pred_A), lpips_batch(pred_B), lpips_batch(pred_ens)
            gt_np = gt.cpu().numpy()
            predA_np, predB_np, predE_np = pred_A.cpu().numpy(), pred_B.cpu().numpy(), pred_ens.cpu().numpy()

            for i, fname in enumerate(fnames):
                g = gt_np[i, 0]
                a, b, e = predA_np[i, 0], predB_np[i, 0], predE_np[i, 0]
                rows.append({
                    "file": fname,
                    "psnr_A": sk_psnr(g, a, data_range=1.0), "ssim_A": sk_ssim(g, a, data_range=1.0), "lpips_A": lpA[i],
                    "psnr_B": sk_psnr(g, b, data_range=1.0), "ssim_B": sk_ssim(g, b, data_range=1.0), "lpips_B": lpB[i],
                    "psnr_ens": sk_psnr(g, e, data_range=1.0), "ssim_ens": sk_ssim(g, e, data_range=1.0), "lpips_ens": lpE[i],
                })

    df = pd.DataFrame(rows).sort_values("file").reset_index(drop=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    print(f"\nSaved {len(df)} rows to {args.out_csv}\n")
    print(f"{'Model':<20}{'PSNR':>8}{'SSIM':>9}{'LPIPS':>9}")
    print(f"{'Stage A':<20}{df['psnr_A'].mean():>8.3f}{df['ssim_A'].mean():>9.4f}{df['lpips_A'].mean():>9.4f}")
    print(f"{'Stage B':<20}{df['psnr_B'].mean():>8.3f}{df['ssim_B'].mean():>9.4f}{df['lpips_B'].mean():>9.4f}")
    print(f"{'Ensemble (A+B)/2':<20}{df['psnr_ens'].mean():>8.3f}{df['ssim_ens'].mean():>9.4f}{df['lpips_ens'].mean():>9.4f}")


if __name__ == "__main__":
    main()
