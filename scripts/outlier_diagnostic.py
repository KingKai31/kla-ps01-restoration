import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GT_DIR = Path(sys.argv[1])
NOISY_DIR = Path(sys.argv[2])
OUT_DIR = Path(sys.argv[3])
FILES = sys.argv[4:]


def box_downsample(arr, factor):
    h, w = arr.shape
    return arr.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))


def bilinear_up(arr, target_shape):
    from scipy.ndimage import zoom
    zf = (target_shape[0] / arr.shape[0], target_shape[1] / arr.shape[1])
    return zoom(arr, zf, order=1)


for fname in FILES:
    gt = np.load(GT_DIR / fname).astype(np.float64)
    deg = np.load(NOISY_DIR / fname).astype(np.float64)
    gt_down = box_downsample(gt, gt.shape[0] // deg.shape[0])
    deg_up = bilinear_up(deg, gt.shape)

    bright_mask_new = gt_down > 0.05 * gt_down.max()
    ratio_new = np.where(bright_mask_new, deg / np.maximum(gt_down, 1e-6), np.nan)

    bright_mask_old = gt > 0.05 * gt.max()
    ratio_old = np.where(bright_mask_old, deg_up / np.maximum(gt, 1e-6), np.nan)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    im0 = axes[0, 0].imshow(gt, cmap="gray"); axes[0, 0].set_title(f"GT (native {gt.shape})"); plt.colorbar(im0, ax=axes[0, 0])
    im1 = axes[0, 1].imshow(deg, cmap="gray"); axes[0, 1].set_title(f"NoisyLR (native {deg.shape})"); plt.colorbar(im1, ax=axes[0, 1])
    im2 = axes[0, 2].imshow(gt_down, cmap="gray"); axes[0, 2].set_title("GT box-downsampled (new method)"); plt.colorbar(im2, ax=axes[0, 2])

    im3 = axes[1, 0].imshow(ratio_old, cmap="inferno", vmin=0, vmax=np.nanmax(ratio_old))
    axes[1, 0].set_title(f"OLD ratio map (deg_upsampled/GT)\nmax={np.nanmax(ratio_old):.2f}")
    plt.colorbar(im3, ax=axes[1, 0])

    im4 = axes[1, 1].imshow(ratio_new, cmap="inferno", vmin=0, vmax=np.nanmax(ratio_old))
    axes[1, 1].set_title(f"NEW ratio map (deg/GT_down)\nmax={np.nanmax(ratio_new):.2f}  (same color scale as OLD)")
    plt.colorbar(im4, ax=axes[1, 1])

    im5 = axes[1, 2].imshow(deg_up, cmap="gray")
    axes[1, 2].set_title("NoisyLR bilinear-upsampled (old method input)")
    plt.colorbar(im5, ax=axes[1, 2])

    fig.suptitle(f"Outlier diagnostic: {fname}")
    fig.tight_layout()
    out_path = OUT_DIR / f"outlier_diagnostic_{fname.replace('.npy', '')}.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Saved {out_path}")
    print(f"  old ratio_max={np.nanmax(ratio_old):.3f} at {np.unravel_index(np.nanargmax(ratio_old), ratio_old.shape)}")
    print(f"  new ratio_max={np.nanmax(ratio_new):.3f} at {np.unravel_index(np.nanargmax(np.nan_to_num(ratio_new, nan=-1)), ratio_new.shape)}")
    print(f"  GT range [{gt.min():.4f},{gt.max():.4f}]  deg range [{deg.min():.4f},{deg.max():.4f}]  gt_down range [{gt_down.min():.4f},{gt_down.max():.4f}]")
