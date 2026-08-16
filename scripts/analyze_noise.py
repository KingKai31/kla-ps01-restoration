"""
Phase 1 noise-physics analysis.

1. Per-file stats (shape/dtype/min/max/mean/std) for every .npy sample in a
   given directory (works on GT-only, NoisyLR-only, or paired dirs).
2. If a paired GT dir is given: upsample NoisyLR to GT size (bilinear) and
   fit a Gamma-style multiplicative speckle model per pair, aggregating
   L (number of looks) and sigma across the whole set.
3. FFT high-frequency energy ratio between degraded (upsampled) and GT, as
   a proxy for the blur/Gaussian-softening component.

No noise model is assumed a priori - every number here is measured.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from scipy import stats as spstats


def load_npy_stats(path: Path) -> dict:
    arr = np.load(path)
    return {
        "file": path.name,
        "shape": str(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }


def scan_directory(dir_path: Path) -> pd.DataFrame:
    files = sorted(dir_path.glob("*.npy"))
    rows = [load_npy_stats(f) for f in files]
    return pd.DataFrame(rows)


def upsample_to(arr: np.ndarray, target_shape: tuple) -> np.ndarray:
    if arr.shape == target_shape:
        return arr
    zoom_factors = (target_shape[0] / arr.shape[0], target_shape[1] / arr.shape[1])
    return zoom(arr, zoom_factors, order=1)  # bilinear


def fit_gamma_speckle(gt: np.ndarray, degraded_up: np.ndarray, eps: float = 1e-6):
    """
    Multiplicative speckle model: degraded = GT * N, N ~ Gamma(shape=L, scale=1/L)
    so E[N]=1, Var[N]=1/L. Estimate ratio = degraded / GT (where GT is not
    near-zero, to avoid blowing up), fit a Gamma distribution to the ratio,
    and report L = shape parameter, plus sigma = std of the ratio.
    """
    mask = gt > (0.05 * gt.max() + eps)
    if mask.sum() < 100:
        return None
    ratio = degraded_up[mask] / (gt[mask] + eps)
    ratio = ratio[(ratio > 0) & np.isfinite(ratio)]
    if ratio.size < 100:
        return None
    shape_hat, loc_hat, scale_hat = spstats.gamma.fit(ratio, floc=0)
    return {
        "L_shape": float(shape_hat),
        "gamma_scale": float(scale_hat),
        "ratio_mean": float(ratio.mean()),
        "ratio_std": float(ratio.std()),
        "ratio_max": float(ratio.max()),
    }


def fft_high_freq_energy_ratio(gt: np.ndarray, degraded_up: np.ndarray, cutoff_frac: float = 0.5):
    """
    Ratio of high-frequency spectral energy (degraded_up / GT) as a blur proxy.
    cutoff_frac selects the outer fraction of the frequency radius treated as
    "high frequency" (0.5 = outer half of the spectrum by radius).
    """
    def high_freq_energy(img):
        f = np.fft.fftshift(np.fft.fft2(img))
        mag2 = np.abs(f) ** 2
        h, w = img.shape
        cy, cx = h / 2, w / 2
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
        rmax = np.sqrt(cy ** 2 + cx ** 2)
        hf_mask = r >= (cutoff_frac * rmax)
        return mag2[hf_mask].sum(), mag2.sum()

    hf_gt, tot_gt = high_freq_energy(gt)
    hf_deg, tot_deg = high_freq_energy(degraded_up)
    return {
        "hf_energy_frac_gt": float(hf_gt / (tot_gt + 1e-12)),
        "hf_energy_frac_degraded": float(hf_deg / (tot_deg + 1e-12)),
        "hf_ratio_degraded_over_gt": float((hf_deg + 1e-12) / (hf_gt + 1e-12)),
    }


def main():
    ap = argparse.ArgumentParser(description="Phase 1 noise-physics analysis")
    ap.add_argument("--test-noisy-dir", type=Path, default=None,
                     help="Directory of degraded-only .npy samples (no GT)")
    ap.add_argument("--train-gt-dir", type=Path, default=None,
                     help="Directory of paired GT .npy files")
    ap.add_argument("--train-noisy-dir", type=Path, default=None,
                     help="Directory of paired NoisyLR .npy files (matched filenames to GT)")
    ap.add_argument("--max-pairs", type=int, default=None,
                     help="Optional cap on number of pairs analyzed (for a quick pass)")
    ap.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "figures").mkdir(parents=True, exist_ok=True)

    summary = {}

    # --- Step 1: per-file stats ---
    if args.test_noisy_dir is not None:
        print(f"Scanning test NoisyLR dir: {args.test_noisy_dir}")
        df_test = scan_directory(args.test_noisy_dir)
        df_test.to_csv(args.out_dir / "stats_test_noisylr.csv", index=False)
        print(f"  {len(df_test)} files. Shape counts:\n{df_test['shape'].value_counts()}")
        summary["test_noisylr"] = {
            "n_files": int(len(df_test)),
            "shape_counts": df_test["shape"].value_counts().to_dict(),
            "min_of_min": float(df_test["min"].min()),
            "max_of_max": float(df_test["max"].max()),
            "mean_of_mean": float(df_test["mean"].mean()),
            "mean_of_std": float(df_test["std"].mean()),
        }

    if args.train_gt_dir is not None:
        print(f"Scanning train GT dir: {args.train_gt_dir}")
        df_gt = scan_directory(args.train_gt_dir)
        df_gt.to_csv(args.out_dir / "stats_train_gt.csv", index=False)
        print(f"  {len(df_gt)} files. Shape counts:\n{df_gt['shape'].value_counts()}")
        summary["train_gt"] = {
            "n_files": int(len(df_gt)),
            "shape_counts": df_gt["shape"].value_counts().to_dict(),
            "min_of_min": float(df_gt["min"].min()),
            "max_of_max": float(df_gt["max"].max()),
            "mean_of_mean": float(df_gt["mean"].mean()),
            "mean_of_std": float(df_gt["std"].mean()),
        }

    if args.train_noisy_dir is not None:
        print(f"Scanning train NoisyLR dir: {args.train_noisy_dir}")
        df_noisy = scan_directory(args.train_noisy_dir)
        df_noisy.to_csv(args.out_dir / "stats_train_noisylr.csv", index=False)
        print(f"  {len(df_noisy)} files. Shape counts:\n{df_noisy['shape'].value_counts()}")
        summary["train_noisylr"] = {
            "n_files": int(len(df_noisy)),
            "shape_counts": df_noisy["shape"].value_counts().to_dict(),
            "min_of_min": float(df_noisy["min"].min()),
            "max_of_max": float(df_noisy["max"].max()),
            "mean_of_mean": float(df_noisy["mean"].mean()),
            "mean_of_std": float(df_noisy["std"].mean()),
        }

    # --- Steps 2 & 3: paired analysis ---
    if args.train_gt_dir is not None and args.train_noisy_dir is not None:
        gt_files = sorted(args.train_gt_dir.glob("*.npy"))
        if args.max_pairs:
            gt_files = gt_files[: args.max_pairs]

        gamma_rows = []
        fft_rows = []
        skipped = 0
        for gt_path in gt_files:
            noisy_path = args.train_noisy_dir / gt_path.name
            if not noisy_path.exists():
                skipped += 1
                continue
            gt = np.load(gt_path).astype(np.float64)
            degraded = np.load(noisy_path).astype(np.float64)
            degraded_up = upsample_to(degraded, gt.shape)

            gres = fit_gamma_speckle(gt, degraded_up)
            if gres:
                gres["file"] = gt_path.name
                gres["gt_shape"] = str(gt.shape)
                gres["degraded_shape"] = str(degraded.shape)
                gres["scale_factor"] = gt.shape[0] / degraded.shape[0]
                gamma_rows.append(gres)

            fres = fft_high_freq_energy_ratio(gt, degraded_up)
            fres["file"] = gt_path.name
            fft_rows.append(fres)

        df_gamma = pd.DataFrame(gamma_rows)
        df_fft = pd.DataFrame(fft_rows)
        df_gamma.to_csv(args.out_dir / "speckle_gamma_fits.csv", index=False)
        df_fft.to_csv(args.out_dir / "fft_highfreq_ratios.csv", index=False)

        print(f"\nPaired analysis: {len(df_gamma)} usable pairs, {skipped} skipped (missing match)")

        if len(df_gamma):
            summary["speckle_gamma"] = {
                "n_pairs": int(len(df_gamma)),
                "L_shape_mean": float(df_gamma["L_shape"].mean()),
                "L_shape_median": float(df_gamma["L_shape"].median()),
                "L_shape_std": float(df_gamma["L_shape"].std()),
                "L_shape_min": float(df_gamma["L_shape"].min()),
                "L_shape_max": float(df_gamma["L_shape"].max()),
                "ratio_std_mean": float(df_gamma["ratio_std"].mean()),
                "ratio_std_median": float(df_gamma["ratio_std"].median()),
                "ratio_max_mean": float(df_gamma["ratio_max"].mean()),
                "by_scale_factor": {
                    str(k): {
                        "n": int(len(v)),
                        "L_shape_mean": float(v["L_shape"].mean()),
                        "ratio_std_mean": float(v["ratio_std"].mean()),
                    }
                    for k, v in df_gamma.groupby("scale_factor")
                },
            }

            # Histogram figure
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            axes[0].hist(df_gamma["L_shape"], bins=50, color="steelblue")
            axes[0].set_title("Fitted Gamma shape L across all pairs")
            axes[0].set_xlabel("L (number of looks)")
            axes[0].set_ylabel("count")

            axes[1].hist(df_gamma["ratio_std"], bins=50, color="darkorange")
            axes[1].set_title("Speckle ratio std (sigma) across all pairs")
            axes[1].set_xlabel("sigma = std(degraded_up / GT)")
            axes[1].set_ylabel("count")

            fig.tight_layout()
            fig.savefig(args.out_dir / "figures" / "speckle_L_sigma_histograms.png", dpi=150)
            plt.close(fig)
            print(f"Saved histogram: {args.out_dir / 'figures' / 'speckle_L_sigma_histograms.png'}")

        if len(df_fft):
            summary["fft_blur"] = {
                "n_pairs": int(len(df_fft)),
                "hf_ratio_degraded_over_gt_mean": float(df_fft["hf_ratio_degraded_over_gt"].mean()),
                "hf_ratio_degraded_over_gt_median": float(df_fft["hf_ratio_degraded_over_gt"].median()),
                "hf_energy_frac_gt_mean": float(df_fft["hf_energy_frac_gt"].mean()),
                "hf_energy_frac_degraded_mean": float(df_fft["hf_energy_frac_degraded"].mean()),
            }

    with open(args.out_dir / "phase1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {args.out_dir / 'phase1_summary.json'}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
