"""
Pulls the worst-performing Stage B val/OOD-proxy samples (bottom-N by SSIM
union top-N by LPIPS) and renders an input/output/GT grid for visual
inspection - specifically checking whether failures look like natural-photo
hallucination (fake organic texture, photographic-looking artifacts from
the DIV2K/Flickr2K training mix) as distinct from the already-known
texture-oversmoothing failure mode, or whether oversmoothing explains all
of them.
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
    ap.add_argument("--checkpoint", type=Path, default=Path("models/checkpoint.pt"))
    ap.add_argument("--n-worst-ssim", type=int, default=5)
    ap.add_argument("--n-worst-lpips", type=int, default=5)
    ap.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fig_dir = args.out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

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

            pred_np, gt_np, noisy_np = pred.cpu().numpy(), gt.cpu().numpy(), noisy.cpu().numpy()
            for i, fname in enumerate(fnames):
                p, g, n = pred_np[i, 0], gt_np[i, 0], noisy_np[i, 0]
                rows.append({
                    "file": fname,
                    "psnr": sk_psnr(g, p, data_range=1.0),
                    "ssim": sk_ssim(g, p, data_range=1.0),
                    "lpips": lp_vals[i],
                    "gt": g, "noisy": n, "pred": p,
                })

    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("gt", "noisy", "pred")} for r in rows])
    df.to_csv(args.out_dir / "stageB_val_per_image_metrics.csv", index=False)

    worst_ssim = df.nsmallest(args.n_worst_ssim, "ssim")
    worst_lpips = df.nlargest(args.n_worst_lpips, "lpips")
    combined = pd.concat([worst_ssim, worst_lpips]).drop_duplicates(subset="file").sort_values("ssim")

    print(f"Worst {args.n_worst_ssim} by SSIM:\n{worst_ssim[['file','psnr','ssim','lpips']].to_string(index=False)}\n")
    print(f"Worst {args.n_worst_lpips} by LPIPS:\n{worst_lpips[['file','psnr','ssim','lpips']].to_string(index=False)}\n")
    print(f"Combined unique worst-case set ({len(combined)} images):\n{combined[['file','psnr','ssim','lpips']].to_string(index=False)}")

    file_to_row = {r["file"]: r for r in rows}
    chosen = [file_to_row[f] for f in combined["file"]]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_show = len(chosen)
    fig, axes = plt.subplots(n_show, 3, figsize=(9, 2.8 * n_show))
    if n_show == 1:
        axes = axes[None, :]
    for row_i, r in enumerate(chosen):
        axes[row_i, 0].imshow(r["noisy"], cmap="gray")
        axes[row_i, 1].imshow(r["pred"], cmap="gray")
        axes[row_i, 2].imshow(r["gt"], cmap="gray")
        if row_i == 0:
            axes[row_i, 0].set_title("NoisyLR input")
            axes[row_i, 1].set_title("Model output")
            axes[row_i, 2].set_title("GT")
        axes[row_i, 0].set_ylabel(f"{r['file']}\nPSNR={r['psnr']:.2f} SSIM={r['ssim']:.3f} LPIPS={r['lpips']:.3f}",
                                   fontsize=8)
        for a in axes[row_i]:
            a.set_xticks([])
            a.set_yticks([])
    fig.suptitle("Stage B: worst-case val samples (bottom-SSIM union top-LPIPS) - hallucination check")
    fig.tight_layout()
    out_path = fig_dir / "stageB_worst_case_hallucination_check.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
