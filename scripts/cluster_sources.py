"""
No source/category labels exist in the KLA training data (just sequential
000000.npy filenames) - but Phase 1's outlier diagnostics visually showed
very different image types mixed together (fine-grained speckle texture,
natural photos of books/jars/wire, fabric weave, etc). The instruction was
to split train/val by source, not randomly, so validation approximates the
real OOD test split rather than leaking near-duplicate samples across the
split.

This clusters GT images into pseudo-source groups using cheap, fast
whole-image statistics (intensity histogram + multi-scale gradient/texture
energy - no heavy pretrained embedding model, keeps this lightweight) and
KMeans. The resulting clusters are then assigned to train/val as whole
groups (GroupKFold-style), so no cluster appears in both splits.

This is a proxy for true source labels, not a substitute - inspect the
saved thumbnail grid per cluster to sanity-check the grouping makes sense
before trusting the split.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def image_features(img: np.ndarray) -> np.ndarray:
    hist, _ = np.histogram(img, bins=16, range=(0, 1), density=True)

    gy, gx = np.gradient(img)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)

    feats = [
        img.mean(), img.std(),
        float(np.percentile(img, 5)), float(np.percentile(img, 95)),
        grad_mag.mean(), grad_mag.std(),
    ]
    feats.extend(hist.tolist())

    # coarse downsampled thumbnail (8x8) as crude texture/layout signature
    h, w = img.shape
    fh, fw = h // 8, w // 8
    thumb = img[: fh * 8, : fw * 8].reshape(8, fh, 8, fw).mean(axis=(1, 3))
    feats.extend(thumb.ravel().tolist())

    return np.array(feats, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-gt-dir", type=Path, required=True)
    ap.add_argument("--n-clusters", type=int, default=12)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--out-dir", type=Path, default=Path("reports"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.out_dir / "figures"
    fig_dir.mkdir(exist_ok=True, parents=True)

    gt_files = sorted(args.train_gt_dir.glob("*.npy"))
    feats = []
    for p in gt_files:
        img = np.load(p).astype(np.float64)
        feats.append(image_features(img))
    X = np.stack(feats)

    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=args.n_clusters, random_state=args.seed, n_init=10)
    labels = km.fit_predict(Xs)

    df = pd.DataFrame({"file": [p.name for p in gt_files], "cluster": labels})
    df.to_csv(args.out_dir / "source_clusters.csv", index=False)

    cluster_sizes = df["cluster"].value_counts().sort_index()
    print("Cluster sizes:\n", cluster_sizes)

    rng = np.random.default_rng(args.seed)
    clusters = list(cluster_sizes.index)
    rng.shuffle(clusters)

    n_total = len(df)
    target_val = int(n_total * args.val_fraction)
    val_clusters, running = [], 0
    for c in clusters:
        if running >= target_val:
            break
        val_clusters.append(c)
        running += cluster_sizes[c]

    df["split"] = np.where(df["cluster"].isin(val_clusters), "val", "train")
    df.to_csv(args.out_dir / "source_clusters.csv", index=False)

    split_summary = {
        "n_clusters": args.n_clusters,
        "val_clusters": [int(c) for c in val_clusters],
        "n_train": int((df["split"] == "train").sum()),
        "n_val": int((df["split"] == "val").sum()),
        "val_fraction_actual": float((df["split"] == "val").mean()),
        "cluster_sizes": {int(k): int(v) for k, v in cluster_sizes.items()},
    }
    with open(args.out_dir / "source_split_summary.json", "w") as f:
        json.dump(split_summary, f, indent=2)
    print(json.dumps(split_summary, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_show = min(6, args.n_clusters)
    fig, axes = plt.subplots(n_show, 6, figsize=(12, 2 * n_show))
    for row, c in enumerate(cluster_sizes.index[:n_show]):
        members = df[df["cluster"] == c]["file"].tolist()[:6]
        for col in range(6):
            ax = axes[row, col]
            if col < len(members):
                img = np.load(args.train_gt_dir / members[col])
                ax.imshow(img, cmap="gray")
            split_tag = "val" if c in val_clusters else "train"
            if col == 0:
                ax.set_ylabel(f"cluster {c} ({split_tag}, n={cluster_sizes[c]})", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Pseudo-source clusters (sanity check the grouping visually)")
    fig.tight_layout()
    fig.savefig(fig_dir / "source_clusters_sample_grid.png", dpi=130)
    plt.close(fig)
    print(f"Saved cluster sample grid to {fig_dir / 'source_clusters_sample_grid.png'}")


if __name__ == "__main__":
    main()
