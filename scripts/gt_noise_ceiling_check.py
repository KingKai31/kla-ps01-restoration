"""
Checks whether KLA's GT images are a perfectly clean reference or carry
some baseline noise floor of their own - this caps achievable PSNR
regardless of model quality (if GT itself has noise, no restoration can
exceed the implied ceiling). Inspects flat/smooth regions specifically
(where any variance is very unlikely to be real structure) via local
variance maps, on a sample of real GT images.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def local_variance_map(img: np.ndarray, win: int = 8) -> np.ndarray:
    h, w = img.shape
    h2, w2 = h - h % win, w - w % win
    blocks = img[:h2, :w2].reshape(h2 // win, win, w2 // win, win)
    return blocks.var(axis=(1, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--n-images", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    files = sorted(args.gt_dir.glob("*.npy"))
    chosen = [files[i] for i in rng.choice(len(files), size=args.n_images, replace=False)]

    rows = []
    flattest_examples = []
    for f in chosen:
        img = np.load(f).astype(np.float64)
        var_map = local_variance_map(img, win=8)
        flat_thresh = np.percentile(var_map, 5)  # flattest 5% of 8x8 blocks in this image
        flat_var = var_map[var_map <= flat_thresh].mean()
        rows.append({
            "file": f.name,
            "global_std": img.std(),
            "flattest_blocks_mean_var": flat_var,
            "flattest_blocks_mean_std": np.sqrt(flat_var),
            "min_block_var": var_map.min(),
        })
        flattest_examples.append((f.name, img, var_map))

    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "gt_noise_ceiling_check.csv", index=False)

    print(df.to_string(index=False))
    print(f"\nAcross {len(df)} images:")
    print(f"  Mean of (flattest-block std) per image: {df['flattest_blocks_mean_std'].mean():.5f}")
    print(f"  Max of (flattest-block std) per image:  {df['flattest_blocks_mean_std'].max():.5f}")
    print(f"  Min block variance seen anywhere: {df['min_block_var'].min():.8f}")

    # implied PSNR ceiling if this residual std were irreducible noise (data range 1.0)
    typical_std = df["flattest_blocks_mean_std"].mean()
    if typical_std > 1e-8:
        implied_ceiling_psnr = 20 * np.log10(1.0 / typical_std)
        print(f"  Implied PSNR ceiling if flat-region std were the true noise floor: {implied_ceiling_psnr:.1f} dB")
    else:
        print("  Flat-region std is ~0 (below float precision noise) - no measurable ceiling implied")

    # visual check: show a handful of flattest-looking images with their variance maps
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_show = min(6, len(flattest_examples))
    fig, axes = plt.subplots(n_show, 2, figsize=(6, 2.6 * n_show))
    df_sorted_idx = df.sort_values("flattest_blocks_mean_std").index[:n_show]
    for row_i, idx in enumerate(df_sorted_idx):
        fname, img, var_map = flattest_examples[idx]
        axes[row_i, 0].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[row_i, 0].set_title(f"{fname}", fontsize=8)
        im = axes[row_i, 1].imshow(var_map, cmap="viridis")
        axes[row_i, 1].set_title(f"local 8x8 variance map", fontsize=8)
        plt.colorbar(im, ax=axes[row_i, 1], fraction=0.046)
        for a in axes[row_i]:
            a.set_xticks([])
            a.set_yticks([])
    fig.suptitle("GT noise-ceiling check: images + local variance maps (darkest = flattest)")
    fig.tight_layout()
    out_path = args.out_dir / "figures" / "gt_noise_ceiling_check.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
