"""Paired GT/NoisyLR dataset for KLA-only training, using the pseudo-source
cluster split from scripts/cluster_sources.py (reports/source_clusters.csv)
so validation approximates OOD generalization instead of leaking near-
duplicate samples from the same source across train/val."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class KLAPairDataset(Dataset):
    def __init__(self, gt_dir: Path, noisy_dir: Path, split_csv: Path, split: str,
                 augment: bool = True):
        self.gt_dir = Path(gt_dir)
        self.noisy_dir = Path(noisy_dir)
        self.augment = augment and split == "train"

        df = pd.read_csv(split_csv)
        self.files = df[df["split"] == split]["file"].tolist()
        if len(self.files) == 0:
            raise ValueError(f"No files found for split={split!r} in {split_csv}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        gt = np.load(self.gt_dir / fname).astype(np.float32)
        noisy = np.load(self.noisy_dir / fname).astype(np.float32)

        if self.augment:
            if np.random.rand() < 0.5:
                gt, noisy = gt[:, ::-1].copy(), noisy[:, ::-1].copy()
            if np.random.rand() < 0.5:
                gt, noisy = gt[::-1, :].copy(), noisy[::-1, :].copy()
            k = np.random.randint(0, 4)
            if k:
                gt, noisy = np.rot90(gt, k).copy(), np.rot90(noisy, k).copy()

        gt_t = torch.from_numpy(gt).unsqueeze(0)
        noisy_t = torch.from_numpy(noisy).unsqueeze(0)
        return noisy_t, gt_t, fname
