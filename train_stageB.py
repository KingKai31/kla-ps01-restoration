"""
Stage B training: fine-tunes from the Stage A checkpoint on KLA data mixed
with externally-sourced clean images (DIV2K/Flickr2K/DTD/SAR or similar),
degraded on the fly with the same validated noise model, using the full
composite loss (Charbonnier + MS-SSIM + LPIPS + Sobel edge + range penalty).

All paths are CLI arguments - nothing hardcoded to this dev machine, so this
drops onto Kaggle (or any other environment) unchanged; only the argument
values differ, e.g.:

  python train_stageB.py \
    --kla-gt-dir /kaggle/input/kla-ps01/train/GT \
    --kla-noisy-dir /kaggle/input/kla-ps01/train/NoisyLR \
    --external-dirs /kaggle/input/div2k/DIV2K_train_HR /kaggle/input/flickr2k/Flickr2K \
      /kaggle/input/dtd/images /kaggle/input/sentinel12-image-pairs-segregated-by-terrain \
    --stageA-checkpoint /kaggle/input/kla-ps01-stagea/stageA_best.pt \
    --checkpoint-dir /kaggle/working/checkpoints \
    --report-dir /kaggle/working/reports

Validation is exclusively on the KLA val/OOD-proxy split (same
source_clusters.csv used for Stage A) - external data is training-only, so
Stage A vs Stage B metrics stay directly comparable on the same held-out set.

Resumable: pass --resume to continue from checkpoints/stageB_last.pt
(model + optimizer + epoch) if present, rather than restarting from the
Stage A checkpoint - Kaggle sessions have a wall-clock limit, and a run
getting cut off shouldn't mean starting over.
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
import lpips

from src.datasets.kla_dataset import KLAPairDataset
from src.datasets.external_image_dataset import ExternalImageDataset, MixedDataset
from src.datasets.synthetic_degrade import SpeckleAdditiveDegrader
from src.models.nafnet import NAFNetSR
from src.losses.stageB_composite import StageBCompositeLoss
from src.utils.reproducibility import set_full_determinism, seed_worker, make_seeded_generator


def evaluate(model, loader, device, lpips_fn):
    model.eval()
    psnrs, ssims, lpipss = [], [], []
    with torch.no_grad():
        for noisy, gt, _ in loader:
            noisy, gt = noisy.to(device), gt.to(device)
            pred = model(noisy).clamp(0.0, 1.0)

            pred_lp = pred.repeat(1, 3, 1, 1) * 2 - 1
            gt_lp = gt.repeat(1, 3, 1, 1) * 2 - 1
            lp = lpips_fn(pred_lp, gt_lp).squeeze()
            lpipss.extend(lp.cpu().tolist() if lp.dim() > 0 else [lp.item()])

            pred_np, gt_np = pred.cpu().numpy(), gt.cpu().numpy()
            for i in range(pred_np.shape[0]):
                p, g = pred_np[i, 0], gt_np[i, 0]
                psnrs.append(sk_psnr(g, p, data_range=1.0))
                ssims.append(sk_ssim(g, p, data_range=1.0))
    model.train()
    return float(np.mean(psnrs)), float(np.mean(ssims)), float(np.mean(lpipss))


def main():
    ap = argparse.ArgumentParser()
    # --- data paths (all required except external-dirs, nothing hardcoded) ---
    ap.add_argument("--kla-gt-dir", type=Path, required=True)
    ap.add_argument("--kla-noisy-dir", type=Path, required=True)
    ap.add_argument("--split-csv", type=Path, default=Path("reports/source_clusters.csv"))
    ap.add_argument("--external-dirs", type=Path, nargs="*", default=[],
                     help="One or more directories of external clean images (DIV2K/Flickr2K/DTD/SAR/...). "
                          "Each is scanned recursively for png/jpg/jpeg/bmp/tif/npy.")
    ap.add_argument("--noise-model-dir", type=Path, default=Path("reports"),
                     help="Where SpeckleAdditiveDegrader reads its fitted noise params from "
                          "(corrected_mult_fits.csv, corrected_add_fits.csv, phase1_decomposition_summary.json)")

    # --- checkpoints ---
    ap.add_argument("--stageA-checkpoint", type=Path, default=Path("checkpoints/stageA_best.pt"))
    ap.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    ap.add_argument("--report-dir", type=Path, default=Path("reports"))
    ap.add_argument("--resume", action="store_true",
                     help="Resume from checkpoint-dir/stageB_last.pt instead of starting from --stageA-checkpoint")

    # --- training config ---
    ap.add_argument("--epochs", type=int, default=20, help="Fine-tuning typically needs fewer epochs than Stage A's 40")
    ap.add_argument("--batch-size", type=int, default=16,
                     help="Default tuned for Kaggle's T4 (16GB), not this dev machine's 4GB card - "
                          "drop to ~4 if running locally, adjust once real VRAM usage is observed on Kaggle.")
    ap.add_argument("--lr", type=float, default=5e-5, help="Lower than Stage A's 2e-4 - fine-tuning, not training from scratch")
    ap.add_argument("--kla-weight", type=float, default=0.5, help="Fraction of training samples drawn from KLA vs external per step")
    ap.add_argument("--tile-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=2,
                     help="Stage A hit a Windows-specific pagefile issue with num_workers>0 (DataLoader workers "
                          "re-importing CUDA). That's a Windows quirk, not expected on Kaggle's Linux runtime, but "
                          "if this errors similarly there, drop to 0.")
    ap.add_argument("--seed", type=int, default=0)

    # --- loss weights (override to tune from validation curves, not frozen) ---
    ap.add_argument("--charbonnier-weight", type=float, default=1.0)
    ap.add_argument("--msssim-weight", type=float, default=0.2)
    ap.add_argument("--lpips-weight", type=float, default=0.075)
    ap.add_argument("--sobel-weight", type=float, default=0.1)
    ap.add_argument("--range-weight", type=float, default=0.05)

    args = ap.parse_args()

    # Path sanity log - first thing printed, before any data/model loading.
    # --split-csv, --stageA-checkpoint, --checkpoint-dir, --report-dir, and
    # --noise-model-dir all default to relative paths that resolve against
    # the process's cwd, not this script's location. On a platform this
    # can't be watched running live (Kaggle), a wrong cwd should show up
    # here as the first log line, not as a FileNotFoundError several steps
    # in or, worse, a silent wrong-file pickup.
    print(f"cwd: {Path.cwd()}")
    for name in ("split_csv", "stageA_checkpoint", "checkpoint_dir", "report_dir", "noise_model_dir"):
        value = getattr(args, name)
        resolved = value.resolve()
        tag = "exists" if resolved.exists() else "MISSING"
        print(f"  --{name.replace('_', '-')}: {value} -> {resolved} [{tag}]")

    # --checkpoint-dir and --report-dir are the only two of the five that
    # this script writes to (checkpoints, train_history_stageB.json). On
    # Kaggle, /kaggle/input/ is mounted read-only - only /kaggle/working/ can
    # be written. If either resolves under a read-only mount this fails here,
    # with an actionable message, instead of a bare PermissionError after
    # data/model loading has already spent several minutes.
    for name in ("checkpoint_dir", "report_dir"):
        resolved = getattr(args, name).resolve()
        resolved.mkdir(parents=True, exist_ok=True)
        probe = resolved / ".write_test"
        try:
            probe.write_text("ok")
            probe.unlink()
        except OSError as e:
            raise SystemExit(
                f"--{name.replace('_', '-')} resolves to {resolved}, which is not writable ({e}). "
                f"On Kaggle, /kaggle/input/ is read-only - point this at /kaggle/working/... instead."
            )

    set_full_determinism(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}, count: {torch.cuda.device_count()} "
              f"(this script uses 1 GPU by default; multi-GPU would need explicit DataParallel/DDP setup - "
              f"not added without confirming that's wanted)")

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    # --- data ---
    kla_train_ds = KLAPairDataset(args.kla_gt_dir, args.kla_noisy_dir, args.split_csv, "train", augment=True)
    val_ds = KLAPairDataset(args.kla_gt_dir, args.kla_noisy_dir, args.split_csv, "val", augment=False)

    external_ds = None
    if args.external_dirs:
        degrader = SpeckleAdditiveDegrader(args.noise_model_dir, seed=args.seed)
        external_ds = ExternalImageDataset(args.external_dirs, degrader, tile_size=args.tile_size, augment=True)
        print(f"External images found: {len(external_ds)} across {len(args.external_dirs)} dir(s)")
    else:
        print("No --external-dirs given - training on KLA data only (composite loss ablation, no external mix)")

    train_ds = MixedDataset(kla_train_ds, external_ds, kla_weight=args.kla_weight)
    print(f"Train (mixed): {len(train_ds)}  KLA-train pool: {len(kla_train_ds)}  "
          f"External pool: {len(external_ds) if external_ds else 0}  Val (KLA OOD-proxy): {len(val_ds)}")

    # worker_init_fn + generator: without these, np.random calls inside
    # MixedDataset/KLAPairDataset's augmentation (which run in worker
    # processes when num_workers>0) are not reproducibly seeded - see
    # src/utils/reproducibility.py.
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True, drop_last=True,
                               worker_init_fn=seed_worker, generator=make_seeded_generator(args.seed))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             worker_init_fn=seed_worker)

    # --- model: fine-tune from Stage A, or resume Stage B ---
    resume_path = args.checkpoint_dir / "stageB_last.pt"
    start_epoch = 1
    history = []

    if args.resume and resume_path.exists():
        print(f"Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        width, upscale = ckpt.get("width", 32), ckpt.get("upscale", 2)
        model = NAFNetSR(img_channel=1, width=width, upscale=upscale).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        history_path = args.report_dir / "train_history_stageB.json"
        if history_path.exists():
            with open(history_path) as f:
                history = json.load(f)
    else:
        print(f"Starting Stage B from Stage A checkpoint: {args.stageA_checkpoint}")
        ckpt = torch.load(args.stageA_checkpoint, map_location=device)
        width, upscale = ckpt.get("width", 32), ckpt.get("upscale", 2)
        model = NAFNetSR(img_channel=1, width=width, upscale=upscale).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Stage A checkpoint: epoch={ckpt.get('epoch')}, val_psnr={ckpt.get('val_psnr'):.3f}, "
              f"val_ssim={ckpt.get('val_ssim'):.4f}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params / 1e6:.2f}M")

    criterion = StageBCompositeLoss(
        charbonnier_weight=args.charbonnier_weight, msssim_weight=args.msssim_weight,
        lpips_weight=args.lpips_weight, sobel_weight=args.sobel_weight, range_weight=args.range_weight,
    ).to(device)
    val_lpips_fn = lpips.LPIPS(net="alex").to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    if args.resume and resume_path.exists() and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        for _ in range(start_epoch - 1):
            scheduler.step()

    best_psnr = max([h["val_psnr"] for h in history], default=-1.0)
    t0 = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_totals = {"loss": 0.0, "charbonnier": 0.0, "ms_ssim_loss": 0.0, "lpips": 0.0, "sobel": 0.0, "range_penalty": 0.0}

        for noisy, gt, _ in train_loader:
            noisy, gt = noisy.to(device), gt.to(device)
            pred = model(noisy)
            loss, parts = criterion(pred, gt)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_totals["loss"] += loss.item()
            for k, v in parts.items():
                epoch_totals[k] += v

        scheduler.step()
        n_batches = len(train_loader)
        val_psnr, val_ssim, val_lpips = evaluate(model, val_loader, device, val_lpips_fn)

        row = {"epoch": epoch, **{f"train_{k}": v / n_batches for k, v in epoch_totals.items()},
               "val_psnr": val_psnr, "val_ssim": val_ssim, "val_lpips": val_lpips,
               "lr": scheduler.get_last_lr()[0], "elapsed_sec": time.time() - t0}
        history.append(row)
        print(f"Epoch {epoch}/{args.epochs}  loss={row['train_loss']:.4f}  "
              f"val_psnr={val_psnr:.3f}  val_ssim={val_ssim:.4f}  val_lpips={val_lpips:.4f}")

        ckpt_out = {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch, "val_psnr": val_psnr, "val_ssim": val_ssim, "val_lpips": val_lpips,
                    "width": width, "upscale": upscale, "stage": "B", "seed": args.seed,
                    "base_checkpoint": str(args.stageA_checkpoint)}

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(ckpt_out, args.checkpoint_dir / "stageB_best.pt")
        torch.save(ckpt_out, args.checkpoint_dir / "stageB_last.pt")

        with open(args.report_dir / "train_history_stageB.json", "w") as f:
            json.dump(history, f, indent=2)

    print(f"Done. Best val PSNR={best_psnr:.3f}. Checkpoints in {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
