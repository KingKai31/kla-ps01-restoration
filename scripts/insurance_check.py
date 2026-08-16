"""
Insurance check: does the calibrated multiplicative+additive synthetic
generator (no separate blur kernel) actually look like real NoisyLR?

Compares, on real GT images with paired real NoisyLR available:
  - per-image intensity stats (min/max/mean/std): real vs synthetic
  - radial-averaged FFT power spectrum: real vs synthetic
  - a visual grid of GT / real NoisyLR / synthetic NoisyLR side by side

If these track well, the "no separate blur kernel" decision is closed and
documented. If there's a systematic gap (e.g. synthetic spectrum missing
mid-frequency energy real data has), that's the signal a blur kernel (or
something else) is still missing.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as spstats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.datasets.synthetic_degrade import SpeckleAdditiveDegrader  # noqa: E402


def radial_power_spectrum(img: np.ndarray, n_bins: int = 32):
    f = np.fft.fftshift(np.fft.fft2(img))
    mag2 = np.abs(f) ** 2
    h, w = img.shape
    cy, cx = h / 2, w / 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    rmax = np.sqrt(cy ** 2 + cx ** 2)
    bins = np.linspace(0, rmax, n_bins + 1)
    idx = np.clip(np.digitize(r.ravel(), bins) - 1, 0, n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins)
    prof = np.bincount(idx, weights=mag2.ravel(), minlength=n_bins) / np.maximum(counts, 1)
    centers = (bins[:-1] + bins[1:]) / 2
    return centers, prof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-gt-dir", type=Path, required=True)
    ap.add_argument("--train-noisy-dir", type=Path, required=True)
    ap.add_argument("--reports-dir", type=Path, default=Path("reports"))
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--n-visual", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    degrader = SpeckleAdditiveDegrader(args.reports_dir, seed=args.seed)
    print(f"Brightness-correction fit: L_eff(x) ~= {degrader.brightness_a:.2f} + {degrader.brightness_b:.2f}*x "
          f"(L_ref={degrader.L_ref:.2f})")

    gt_files = sorted(args.train_gt_dir.glob("*.npy"))
    n = min(args.n_samples, len(gt_files))
    chosen_idx = rng.choice(len(gt_files), size=n, replace=False)
    chosen = [gt_files[i] for i in chosen_idx]

    real_rows, synth_rows = [], []
    real_radial, synth_radial = [], []
    centers = None

    for gt_path in chosen:
        gt = np.load(gt_path).astype(np.float64)
        real_noisy = np.load(args.train_noisy_dir / gt_path.name).astype(np.float64)
        synth_noisy = degrader.degrade(gt).astype(np.float64)

        real_rows.append({"file": gt_path.name, "min": real_noisy.min(), "max": real_noisy.max(),
                           "mean": real_noisy.mean(), "std": real_noisy.std()})
        synth_rows.append({"file": gt_path.name, "min": synth_noisy.min(), "max": synth_noisy.max(),
                            "mean": synth_noisy.mean(), "std": synth_noisy.std()})

        c, p_real = radial_power_spectrum(real_noisy)
        _, p_synth = radial_power_spectrum(synth_noisy)
        centers = c
        real_radial.append(p_real)
        synth_radial.append(p_synth)

    df_real = pd.DataFrame(real_rows)
    df_synth = pd.DataFrame(synth_rows)
    df_real.to_csv(args.reports_dir / "insurance_check_real_stats.csv", index=False)
    df_synth.to_csv(args.reports_dir / "insurance_check_synth_stats.csv", index=False)

    real_radial_mean = np.mean(real_radial, axis=0)
    synth_radial_mean = np.mean(synth_radial, axis=0)

    ks_std = spstats.ks_2samp(df_real["std"], df_synth["std"])
    ks_max = spstats.ks_2samp(df_real["max"], df_synth["max"])
    ks_min = spstats.ks_2samp(df_real["min"], df_synth["min"])

    with np.errstate(divide="ignore", invalid="ignore"):
        log_spectrum_ratio = np.log10((synth_radial_mean + 1e-12) / (real_radial_mean + 1e-12))

    summary = {
        "n_samples": n,
        "brightness_fit": {"a": degrader.brightness_a, "b": degrader.brightness_b, "L_ref": degrader.L_ref},
        "real": {k: float(df_real[k].mean()) for k in ["min", "max", "mean", "std"]},
        "synth": {k: float(df_synth[k].mean()) for k in ["min", "max", "mean", "std"]},
        "real_std_of_per_image_std": float(df_real["std"].std()),
        "synth_std_of_per_image_std": float(df_synth["std"].std()),
        "ks_test_std": {"stat": float(ks_std.statistic), "p": float(ks_std.pvalue)},
        "ks_test_max": {"stat": float(ks_max.statistic), "p": float(ks_max.pvalue)},
        "ks_test_min": {"stat": float(ks_min.statistic), "p": float(ks_min.pvalue)},
        "radial_spectrum_centers": centers.tolist(),
        "radial_spectrum_real_mean": real_radial_mean.tolist(),
        "radial_spectrum_synth_mean": synth_radial_mean.tolist(),
        "log10_spectrum_ratio_synth_over_real": log_spectrum_ratio.tolist(),
        "max_abs_log10_spectrum_ratio_excl_dc": float(np.max(np.abs(log_spectrum_ratio[1:]))),
    }

    fig_dir = args.reports_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, ["std", "max", "min"]):
        ax.hist(df_real[col], bins=40, alpha=0.6, label="real", color="steelblue")
        ax.hist(df_synth[col], bins=40, alpha=0.6, label="synthetic", color="darkorange")
        ax.set_title(f"per-image {col}: real vs synthetic")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "insurance_check_stat_histograms.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(centers[1:], real_radial_mean[1:], label="real NoisyLR", color="steelblue")
    axes[0].plot(centers[1:], synth_radial_mean[1:], label="synthetic NoisyLR", color="darkorange", linestyle="--")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("spatial frequency radius (bin)")
    axes[0].set_ylabel("mean power (log scale)")
    axes[0].set_title(f"Radial power spectrum (n={n})")
    axes[0].legend(fontsize=8)

    axes[1].axhline(0, color="gray", linewidth=0.8)
    axes[1].plot(centers[1:], log_spectrum_ratio[1:], color="crimson", marker="o", markersize=3)
    axes[1].set_xlabel("spatial frequency radius (bin)")
    axes[1].set_ylabel("log10(synthetic / real) power")
    axes[1].set_title("Spectrum mismatch (0 = perfect match)")
    fig.tight_layout()
    fig.savefig(fig_dir / "insurance_check_radial_spectrum.png", dpi=150)
    plt.close(fig)

    n_vis = min(args.n_visual, len(chosen))
    fig, axes = plt.subplots(n_vis, 3, figsize=(9, 2.7 * n_vis))
    if n_vis == 1:
        axes = axes[None, :]
    for i in range(n_vis):
        gt_path = chosen[i]
        gt = np.load(gt_path).astype(np.float64)
        real_noisy = np.load(args.train_noisy_dir / gt_path.name).astype(np.float64)
        synth_noisy = degrader.degrade(gt).astype(np.float64)
        vmin = min(real_noisy.min(), synth_noisy.min())
        vmax = max(real_noisy.max(), synth_noisy.max())
        axes[i, 0].imshow(gt, cmap="gray")
        axes[i, 1].imshow(real_noisy, cmap="gray", vmin=vmin, vmax=vmax)
        axes[i, 2].imshow(synth_noisy, cmap="gray", vmin=vmin, vmax=vmax)
        if i == 0:
            axes[i, 0].set_title("GT")
            axes[i, 1].set_title("real NoisyLR")
            axes[i, 2].set_title("synthetic NoisyLR")
        axes[i, 0].set_ylabel(gt_path.name, fontsize=7)
        for a in axes[i]:
            a.set_xticks([])
            a.set_yticks([])
    fig.tight_layout()
    fig.savefig(fig_dir / "insurance_check_visual_grid.png", dpi=130)
    plt.close(fig)

    with open(args.reports_dir / "insurance_check_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
