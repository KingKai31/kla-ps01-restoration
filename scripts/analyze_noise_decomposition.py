"""
Phase 1 refinement: additive + multiplicative noise decomposition, corrected
FFT blur estimate, and outlier investigation.

Why this supersedes the first-pass analyze_noise.py numbers:
  1. The original script upsampled NoisyLR (128x128) to GT's size (256x256)
     with bilinear interpolation before comparing. Bilinear upsampling
     SMOOTHS the noisy signal itself (it averages neighboring noisy pixels),
     which biases every downstream statistic - the Gamma L fit, ratio sigma,
     ratio_max, and the FFT high-frequency ratio all inherit that artifact.
  2. This version instead box-downsamples GT to NoisyLR's native resolution
     (exact non-overlapping 2x2 mean pooling for the 256->128 factor) and
     compares everything at NoisyLR's native, unmodified resolution. Nothing
     about the noisy signal is touched.
  3. The negative pixel values seen in Phase 1 step 1 proved the degradation
     is NOT pure multiplicative speckle (GT * M can't go negative for
     GT >= 0). This script fits a two-component model instead:
         NoisyLR = GT_down * M + A
         M ~ Gamma(shape=L, scale=1/L)   (mean 1)  -- multiplicative speckle
         A ~ N(mu_A, sigma_A^2)                     -- additive noise
     M is fit on "positive-value" (bright) regions where GT_down is not
     near-zero, so the ratio degraded/GT_down approx= M (additive term is
     small relative to a large denominator there). A is fit on dark regions
     where GT_down approx= 0, so degraded - GT_down approx= A directly
     (the multiplicative term GT_down*(M-1) vanishes when GT_down approx=0).
  4. The decomposition is then validated (not assumed): residual variance
     is binned by GT_down brightness and compared against the model's
     predicted curve sqrt(GT^2/L + sigma_A^2). If decomposition is right,
     empirical and predicted curves should track.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as spstats


def box_downsample(arr: np.ndarray, factor: int) -> np.ndarray:
    h, w = arr.shape
    assert h % factor == 0 and w % factor == 0, f"shape {arr.shape} not divisible by {factor}"
    return arr.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))


def hf_energy_frac(img: np.ndarray, cutoff_frac: float = 0.5) -> float:
    f = np.fft.fftshift(np.fft.fft2(img))
    mag2 = np.abs(f) ** 2
    h, w = img.shape
    cy, cx = h / 2, w / 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    rmax = np.sqrt(cy ** 2 + cx ** 2)
    hf_mask = r >= cutoff_frac * rmax
    return float(mag2[hf_mask].sum() / (mag2.sum() + 1e-12))


def analyze_pair(gt: np.ndarray, degraded: np.ndarray,
                  bright_thresh_frac: float = 0.05, dark_thresh: float = 0.02,
                  dark_reservoir_n: int = 50, rng: np.random.Generator = None):
    factor = gt.shape[0] // degraded.shape[0]
    gt_down = box_downsample(gt, factor) if factor > 1 else gt.copy()

    bright_mask = gt_down > (bright_thresh_frac * gt_down.max() + 1e-9)
    dark_mask = gt_down < dark_thresh

    out = {"gt_down": gt_down}

    if bright_mask.sum() > 50:
        ratio = degraded[bright_mask] / gt_down[bright_mask]
        ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
        if ratio.size > 50:
            shape_hat, _, scale_hat = spstats.gamma.fit(ratio, floc=0)
            out["mult"] = {
                "L": float(shape_hat),
                "scale": float(scale_hat),
                "ratio_mean": float(ratio.mean()),
                "ratio_std": float(ratio.std()),
                "ratio_max": float(ratio.max()),
                "ratio_argmax_flat": int(np.argmax((degraded[bright_mask] / gt_down[bright_mask]))),
                "n": int(ratio.size),
            }

    if dark_mask.sum() > 50:
        residual = (degraded[dark_mask] - gt_down[dark_mask]).astype(np.float64)
        out["add"] = {
            "mean": float(residual.mean()),
            "std": float(residual.std()),
            "skew": float(spstats.skew(residual)),
            "kurtosis": float(spstats.kurtosis(residual)),
            "n": int(residual.size),
        }
        if rng is not None and residual.size > 0:
            k = min(dark_reservoir_n, residual.size)
            out["dark_reservoir"] = rng.choice(residual, size=k, replace=False)

    hf_gt = hf_energy_frac(gt_down)
    hf_deg = hf_energy_frac(degraded)
    out["fft"] = {
        "hf_frac_gt_down": hf_gt,
        "hf_frac_degraded": hf_deg,
        "hf_ratio_degraded_over_gt": float((hf_deg + 1e-12) / (hf_gt + 1e-12)),
    }

    # full-image residual for heteroscedasticity binning
    out["residual_full"] = degraded - gt_down
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-gt-dir", type=Path, required=True)
    ap.add_argument("--train-noisy-dir", type=Path, required=True)
    ap.add_argument("--old-gamma-csv", type=Path, default=None,
                     help="Path to Phase-1-first-pass speckle_gamma_fits.csv for old-vs-new comparison")
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("reports"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    gt_files = sorted(args.train_gt_dir.glob("*.npy"))
    if args.max_pairs:
        gt_files = gt_files[: args.max_pairs]

    mult_rows = []
    add_rows = []
    fft_rows = []
    dark_reservoir_all = []

    n_bins = 20
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_sum = np.zeros(n_bins)
    bin_sumsq = np.zeros(n_bins)
    bin_count = np.zeros(n_bins, dtype=np.int64)

    skipped = 0
    for gt_path in gt_files:
        noisy_path = args.train_noisy_dir / gt_path.name
        if not noisy_path.exists():
            skipped += 1
            continue
        gt = np.load(gt_path).astype(np.float64)
        degraded = np.load(noisy_path).astype(np.float64)

        res = analyze_pair(gt, degraded, rng=rng)
        gt_down = res["gt_down"]

        if "mult" in res:
            row = dict(res["mult"])
            row["file"] = gt_path.name
            mult_rows.append(row)
        if "add" in res:
            row = dict(res["add"])
            row["file"] = gt_path.name
            add_rows.append(row)
            if "dark_reservoir" in res:
                dark_reservoir_all.append(res["dark_reservoir"])
        row = dict(res["fft"])
        row["file"] = gt_path.name
        fft_rows.append(row)

        resid = res["residual_full"]
        bin_idx = np.clip(np.digitize(gt_down.ravel(), bin_edges) - 1, 0, n_bins - 1)
        r = resid.ravel()
        bin_sum += np.bincount(bin_idx, weights=r, minlength=n_bins)
        bin_sumsq += np.bincount(bin_idx, weights=r ** 2, minlength=n_bins)
        bin_count += np.bincount(bin_idx, minlength=n_bins)

    df_mult = pd.DataFrame(mult_rows)
    df_add = pd.DataFrame(add_rows)
    df_fft = pd.DataFrame(fft_rows)
    df_mult.to_csv(args.out_dir / "corrected_mult_fits.csv", index=False)
    df_add.to_csv(args.out_dir / "corrected_add_fits.csv", index=False)
    df_fft.to_csv(args.out_dir / "corrected_fft_ratios.csv", index=False)

    dark_pool = np.concatenate(dark_reservoir_all) if dark_reservoir_all else np.array([])

    summary = {"n_pairs": len(gt_files) - skipped, "skipped": skipped}

    # --- multiplicative component ---
    summary["mult"] = {
        "L_mean": float(df_mult["L"].mean()),
        "L_median": float(df_mult["L"].median()),
        "L_std": float(df_mult["L"].std()),
        "L_min": float(df_mult["L"].min()),
        "L_max": float(df_mult["L"].max()),
        "ratio_mean_mean": float(df_mult["ratio_mean"].mean()),
        "ratio_std_mean": float(df_mult["ratio_std"].mean()),
        "ratio_std_median": float(df_mult["ratio_std"].median()),
        "ratio_max_mean": float(df_mult["ratio_max"].mean()),
        "ratio_max_max": float(df_mult["ratio_max"].max()),
    }

    # --- additive component ---
    summary["add"] = {
        "mean_of_means": float(df_add["mean"].mean()),
        "median_of_means": float(df_add["mean"].median()),
        "mean_of_stds": float(df_add["std"].mean()),
        "median_of_stds": float(df_add["std"].median()),
        "std_of_stds": float(df_add["std"].std()),
        "pooled_reservoir_n": int(dark_pool.size),
    }
    if dark_pool.size > 0:
        summary["add"]["pooled_mean"] = float(dark_pool.mean())
        summary["add"]["pooled_std"] = float(dark_pool.std())
        summary["add"]["pooled_skew"] = float(spstats.skew(dark_pool))
        summary["add"]["pooled_kurtosis_excess"] = float(spstats.kurtosis(dark_pool))
        shapiro_sample = dark_pool if dark_pool.size <= 5000 else rng.choice(dark_pool, 5000, replace=False)
        sh_stat, sh_p = spstats.shapiro(shapiro_sample)
        summary["add"]["shapiro_stat"] = float(sh_stat)
        summary["add"]["shapiro_p"] = float(sh_p)
        summary["add"]["shapiro_n"] = int(shapiro_sample.size)

    # --- FFT (corrected, native resolution, no upsampling) ---
    summary["fft_corrected"] = {
        "hf_ratio_mean": float(df_fft["hf_ratio_degraded_over_gt"].mean()),
        "hf_ratio_median": float(df_fft["hf_ratio_degraded_over_gt"].median()),
        "hf_frac_gt_down_mean": float(df_fft["hf_frac_gt_down"].mean()),
        "hf_frac_degraded_mean": float(df_fft["hf_frac_degraded"].mean()),
    }

    # --- heteroscedasticity validation ---
    with np.errstate(invalid="ignore", divide="ignore"):
        bin_mean = bin_sum / bin_count
        bin_var = bin_sumsq / bin_count - bin_mean ** 2
        bin_std = np.sqrt(np.maximum(bin_var, 0))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    L_med = summary["mult"]["L_median"]
    sigma_A_med = summary["add"]["median_of_stds"]
    predicted_std = np.sqrt((bin_centers ** 2) / L_med + sigma_A_med ** 2)

    summary["heteroscedasticity_check"] = {
        "bin_centers": bin_centers.tolist(),
        "bin_count": bin_count.tolist(),
        "empirical_std": bin_std.tolist(),
        "predicted_std": predicted_std.tolist(),
    }

    # --- old vs new comparison + outlier investigation ---
    if args.old_gamma_csv and args.old_gamma_csv.exists():
        df_old = pd.read_csv(args.old_gamma_csv)
        merged = df_mult.merge(df_old[["file", "L_shape", "ratio_std", "ratio_max"]],
                                on="file", suffixes=("_new", "_old"))
        merged = merged.rename(columns={"L_shape": "L_old"})
        merged.to_csv(args.out_dir / "old_vs_new_comparison.csv", index=False)
        summary["old_vs_new"] = {
            "L_mean_old": float(merged["L_old"].mean()),
            "L_mean_new": float(merged["L"].mean()),
            "ratio_std_mean_old": float(merged["ratio_std_old"].mean()),
            "ratio_std_mean_new": float(merged["ratio_std_new"].mean()),
            "ratio_max_mean_old": float(merged["ratio_max_old"].mean()),
            "ratio_max_mean_new": float(merged["ratio_max_new"].mean()),
        }

        top5 = merged.nlargest(5, "ratio_max_old")[["file", "ratio_max_old", "ratio_max_new"]]
        summary["outlier_top5_old_ratio_max"] = top5.to_dict(orient="records")

    top5_new = df_mult.nlargest(5, "ratio_max")[["file", "ratio_max", "L", "n"]]
    summary["outlier_top5_new_ratio_max"] = top5_new.to_dict(orient="records")

    with open(args.out_dir / "phase1_decomposition_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # --- figures ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(df_mult["L"], bins=50, color="steelblue")
    axes[0].set_title("Corrected multiplicative L (bright-region ratio fit)")
    axes[0].set_xlabel("L")
    axes[1].hist(df_add["std"], bins=50, color="darkorange")
    axes[1].set_title("Additive sigma_A per pair (dark-region residual std)")
    axes[1].set_xlabel("sigma_A")
    fig.tight_layout()
    fig.savefig(fig_dir / "corrected_mult_L_and_additive_sigma.png", dpi=150)
    plt.close(fig)

    if dark_pool.size > 0:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].hist(dark_pool, bins=80, color="seagreen", density=True)
        xs = np.linspace(dark_pool.min(), dark_pool.max(), 200)
        axes[0].plot(xs, spstats.norm.pdf(xs, dark_pool.mean(), dark_pool.std()), "k--",
                     label=f"N({dark_pool.mean():.4f}, {dark_pool.std():.4f}^2)")
        axes[0].set_title("Pooled dark-region residual (degraded - GT_down)")
        axes[0].legend()
        spstats.probplot(dark_pool, dist="norm", plot=axes[1])
        axes[1].set_title("Q-Q plot vs Normal")
        fig.tight_layout()
        fig.savefig(fig_dir / "additive_residual_normality_check.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    valid = bin_count > 50
    ax.plot(bin_centers[valid], bin_std[valid], "o-", label="empirical residual std", color="crimson")
    ax.plot(bin_centers, predicted_std, "--", label=f"predicted: sqrt(GT^2/L={L_med:.1f} + sigma_A^2={sigma_A_med:.4f}^2)",
            color="black")
    ax.set_xlabel("GT_down intensity bin center")
    ax.set_ylabel("std of (degraded - GT_down)")
    ax.set_title("Decomposition validation: residual heteroscedasticity vs model")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "heteroscedasticity_validation.png", dpi=150)
    plt.close(fig)

    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
