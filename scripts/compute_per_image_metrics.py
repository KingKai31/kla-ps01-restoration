"""
Computes per-image PSNR/SSIM/LPIPS on the val/OOD-proxy split for a given
checkpoint and saves to CSV. Shared by Stage A and Stage B so the same
file-order/methodology produces directly-comparable paired data (needed for
paired statistical tests - same 506 images, same order, scored by each
model).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import lpips
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.datasets.kla_dataset import KLAPairDataset  # noqa: E402
from src.models.nafnet import NAFNetSR  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--noisy-dir", type=Path, required=True)
    ap.add_argument("--split-csv", type=Path, default=Path("reports/source_clusters.csv"))
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = NAFNetSR(img_channel=1, width=ckpt.get("width", 32), upscale=ckpt.get("upscale", 2)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    lpips_fn = lpips.LPIPS(net="alex").to(device)

    val_ds = KLAPairDataset(args.gt_dir, args.noisy_dir, args.split_csv, "val", augment=False)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)

    rows = []
    with torch.no_grad():
        for noisy, gt, fnames in val_loader:
            noisy, gt = noisy.to(device), gt.to(device)
            pred = model(noisy).clamp(0.0, 1.0)

            pred_lp = pred.repeat(1, 3, 1, 1) * 2 - 1
            gt_lp = gt.repeat(1, 3, 1, 1) * 2 - 1
            lp = lpips_fn(pred_lp, gt_lp).squeeze()
            lp_vals = lp.cpu().tolist() if lp.dim() > 0 else [lp.item()]

            pred_np, gt_np = pred.cpu().numpy(), gt.cpu().numpy()
            for i, fname in enumerate(fnames):
                p, g = pred_np[i, 0], gt_np[i, 0]
                rows.append({
                    "file": fname,
                    "psnr": sk_psnr(g, p, data_range=1.0),
                    "ssim": sk_ssim(g, p, data_range=1.0),
                    "lpips": lp_vals[i],
                })

    df = pd.DataFrame(rows).sort_values("file").reset_index(drop=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Saved {len(df)} rows to {args.out_csv}")
    print(f"Mean: PSNR={df['psnr'].mean():.3f} SSIM={df['ssim'].mean():.4f} LPIPS={df['lpips'].mean():.4f}")


if __name__ == "__main__":
    main()
