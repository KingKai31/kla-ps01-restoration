"""
Stage A training: NAFNet-style backbone, Charbonnier + MS-SSIM loss, KLA
pairs only (256 GT / 128 NoisyLR), split by pseudo-source cluster so
validation approximates OOD generalization.

Usage:
    python train.py --gt-dir <GT dir> --noisy-dir <NoisyLR dir> \
        --split-csv reports/source_clusters.csv --epochs 40
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

from src.datasets.kla_dataset import KLAPairDataset
from src.models.nafnet import NAFNetSR
from src.losses.charbonnier_msssim import CharbonnierMSSSIMLoss
from src.utils.reproducibility import set_full_determinism, seed_worker, make_seeded_generator


def evaluate(model, loader, device):
    model.eval()
    psnrs, ssims = [], []
    with torch.no_grad():
        for noisy, gt, _ in loader:
            noisy, gt = noisy.to(device), gt.to(device)
            pred = model(noisy).clamp(0.0, 1.0)
            pred_np = pred.cpu().numpy()
            gt_np = gt.cpu().numpy()
            for i in range(pred_np.shape[0]):
                p = pred_np[i, 0]
                g = gt_np[i, 0]
                psnrs.append(sk_psnr(g, p, data_range=1.0))
                ssims.append(sk_ssim(g, p, data_range=1.0))
    model.train()
    return float(np.mean(psnrs)), float(np.mean(ssims))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--noisy-dir", type=Path, required=True)
    ap.add_argument("--split-csv", type=Path, default=Path("reports/source_clusters.csv"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    ap.add_argument("--report-dir", type=Path, default=Path("reports"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_full_determinism(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds = KLAPairDataset(args.gt_dir, args.noisy_dir, args.split_csv, "train", augment=True)
    val_ds = KLAPairDataset(args.gt_dir, args.noisy_dir, args.split_csv, "val", augment=False)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    # worker_init_fn + generator: without these, np.random calls inside
    # KLAPairDataset's augmentation (which runs in worker processes when
    # num_workers>0) are not reproducibly seeded - see
    # src/utils/reproducibility.py for why this is a real, separate gap
    # from seeding the main process's RNGs.
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True, drop_last=True,
                               worker_init_fn=seed_worker, generator=make_seeded_generator(args.seed))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True,
                             worker_init_fn=seed_worker)

    model = NAFNetSR(img_channel=1, width=args.width, upscale=2).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params / 1e6:.2f}M")

    criterion = CharbonnierMSSSIMLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_psnr = -1.0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, epoch_char, epoch_msssim = 0.0, 0.0, 0.0
        for noisy, gt, _ in train_loader:
            noisy, gt = noisy.to(device), gt.to(device)
            pred = model(noisy)
            loss, parts = criterion(pred, gt)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_char += parts["charbonnier"]
            epoch_msssim += parts["ms_ssim_loss"]

        scheduler.step()
        n_batches = len(train_loader)
        val_psnr, val_ssim = evaluate(model, val_loader, device)

        row = {
            "epoch": epoch,
            "train_loss": epoch_loss / n_batches,
            "train_charbonnier": epoch_char / n_batches,
            "train_ms_ssim_loss": epoch_msssim / n_batches,
            "val_psnr": val_psnr,
            "val_ssim": val_ssim,
            "lr": scheduler.get_last_lr()[0],
            "elapsed_sec": time.time() - t0,
        }
        history.append(row)
        print(f"Epoch {epoch}/{args.epochs}  loss={row['train_loss']:.4f}  "
              f"val_psnr={val_psnr:.3f}  val_ssim={val_ssim:.4f}")

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save({"model_state_dict": model.state_dict(),
                        "epoch": epoch, "val_psnr": val_psnr, "val_ssim": val_ssim,
                        "width": args.width, "upscale": 2, "seed": args.seed},
                       args.checkpoint_dir / "stageA_best.pt")

        with open(args.report_dir / "train_history.json", "w") as f:
            json.dump(history, f, indent=2)

    torch.save({"model_state_dict": model.state_dict(), "epoch": args.epochs,
                "val_psnr": val_psnr, "val_ssim": val_ssim,
                "width": args.width, "upscale": 2, "seed": args.seed},
               args.checkpoint_dir / "stageA_last.pt")

    print(f"Done. Best val PSNR={best_psnr:.3f}. Checkpoints in {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
