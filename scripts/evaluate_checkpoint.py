"""
Full metrics report for a trained checkpoint: SSIM/PSNR/LPIPS on the train
split (in-distribution, seen during training) and the val split (held-out
pseudo-source clusters - this project's OOD proxy, per
scripts/cluster_sources.py), plus the gap between them.

Note: "val split" and "the held-out-cluster OOD-proxy split" are the same
set here - the split was built by holding out whole clusters specifically
to serve as the OOD proxy (see README's "Known limitations" on why this is
an approximation, not a confirmed OOD split). Train-split metrics serve as
the in-distribution reference point for the train/val gap.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import lpips
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.datasets.kla_dataset import KLAPairDataset  # noqa: E402
from src.models.nafnet import NAFNetSR  # noqa: E402


def evaluate_split(model, loader, device, lpips_fn, max_samples=None):
    model.eval()
    psnrs, ssims, lpipss = [], [], []
    n_seen = 0
    with torch.no_grad():
        for noisy, gt, _ in loader:
            noisy, gt = noisy.to(device), gt.to(device)
            pred = model(noisy).clamp(0.0, 1.0)

            # LPIPS expects 3-channel [-1, 1] input
            pred_lp = pred.repeat(1, 3, 1, 1) * 2 - 1
            gt_lp = gt.repeat(1, 3, 1, 1) * 2 - 1
            lp = lpips_fn(pred_lp, gt_lp).squeeze()
            if lp.dim() == 0:
                lpipss.append(lp.item())
            else:
                lpipss.extend(lp.cpu().tolist())

            pred_np = pred.cpu().numpy()
            gt_np = gt.cpu().numpy()
            for i in range(pred_np.shape[0]):
                p, g = pred_np[i, 0], gt_np[i, 0]
                psnrs.append(sk_psnr(g, p, data_range=1.0))
                ssims.append(sk_ssim(g, p, data_range=1.0))

            n_seen += noisy.shape[0]
            if max_samples and n_seen >= max_samples:
                break
    return {
        "n": len(psnrs),
        "psnr_mean": float(np.mean(psnrs)), "psnr_std": float(np.std(psnrs)),
        "ssim_mean": float(np.mean(ssims)), "ssim_std": float(np.std(ssims)),
        "lpips_mean": float(np.mean(lpipss)), "lpips_std": float(np.std(lpipss)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--noisy-dir", type=Path, required=True)
    ap.add_argument("--split-csv", type=Path, default=Path("reports/source_clusters.csv"))
    ap.add_argument("--checkpoint", type=Path, default=Path("checkpoints/stageA_best.pt"))
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--train-eval-samples", type=int, default=506,
                     help="Subsample train split to this many images for a runtime-comparable eval (matches val split size)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("reports/stageA_metrics.json"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = NAFNetSR(img_channel=1, width=ckpt.get("width", 32), upscale=ckpt.get("upscale", 2)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch')}, "
          f"train-time val_psnr={ckpt.get('val_psnr'):.3f}, val_ssim={ckpt.get('val_ssim'):.4f}")

    lpips_fn = lpips.LPIPS(net="alex").to(device)

    train_ds = KLAPairDataset(args.gt_dir, args.noisy_dir, args.split_csv, "train", augment=False)
    val_ds = KLAPairDataset(args.gt_dir, args.noisy_dir, args.split_csv, "val", augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"Evaluating train split (in-distribution, seen; subsampled to {args.train_eval_samples})...")
    train_metrics = evaluate_split(model, train_loader, device, lpips_fn, max_samples=args.train_eval_samples)
    print(train_metrics)

    print(f"Evaluating val split (held-out clusters = OOD proxy, {len(val_ds)} images)...")
    val_metrics = evaluate_split(model, val_loader, device, lpips_fn, max_samples=None)
    print(val_metrics)

    gap = {
        "psnr_gap_train_minus_val": train_metrics["psnr_mean"] - val_metrics["psnr_mean"],
        "ssim_gap_train_minus_val": train_metrics["ssim_mean"] - val_metrics["ssim_mean"],
        "lpips_gap_val_minus_train": val_metrics["lpips_mean"] - train_metrics["lpips_mean"],
    }

    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": ckpt.get("epoch"),
        "train_split_in_distribution_seen": train_metrics,
        "val_split_ood_proxy_held_out_clusters": val_metrics,
        "gap": gap,
        "note": "train split metrics are on data the model was trained on (in-distribution ceiling), "
                "not a held-out in-distribution set - val split doubles as this project's OOD proxy. "
                "See README known limitations.",
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
