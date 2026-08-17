"""
On-the-fly degraded dataset for external clean images (DIV2K/Flickr2K/DTD/
SAR or any other source). Each __getitem__ call re-degrades the underlying
clean image with a freshly sampled noise draw from
SpeckleAdditiveDegrader - this means epochs don't repeat the exact same
synthetic pair, which is free augmentation the fixed KLA pairs don't get.

Accepts a directory of arbitrary images (png/jpg/jpeg/bmp/tif) and/or .npy
arrays - deliberately format-agnostic since external sources rarely agree
on a format.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from .synthetic_degrade import SpeckleAdditiveDegrader

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _is_image_file(p: Path) -> bool:
    return p.suffix.lower() in IMAGE_EXTENSIONS or p.suffix.lower() == ".npy"


def _walk_dedup(dir_path: Path, skipped: list) -> list:
    """Recurse through dir_path, collecting image/npy files, but when two
    sibling subdirectories differ only by case (e.g. a dataset shipped with
    both 'DIV2K_train_HR' and a lowercase mirror of the same images), recurse
    into only one of them so the same picture isn't loaded twice under two
    different paths. Prefers the non-all-lowercase name when exactly one
    candidate in the collision group isn't all-lowercase (matches how these
    datasets are actually named); otherwise picks alphabetically-first and
    logs the rest as skipped, since automatic disambiguation isn't certain
    in a genuinely ambiguous case - that should surface for a human to check,
    not get silently guessed.

    UNTESTED on a real case-sensitive filesystem: this was built and unit-
    tested against the DIV2K-style scenario (see scripts/_test_dedup_logic.py)
    using mock objects, not real directories, because this dev machine is
    Windows/NTFS - case-insensitive, so 'DIV2K_train_HR' and 'div2k_train_hr'
    resolve to the same directory here and the real collision can't be built
    locally to test end-to-end. Kaggle's filesystem is Linux (case-sensitive),
    where the real scenario actually occurs. Check the first Kaggle run's
    logs for the "skipped N likely case-duplicate dir(s)" message to confirm
    this actually fires and picks correctly, rather than assuming it does."""
    try:
        entries = list(dir_path.iterdir())
    except (PermissionError, FileNotFoundError):
        return []

    subdirs = [e for e in entries if e.is_dir()]
    by_lower = {}
    for d in subdirs:
        by_lower.setdefault(d.name.lower(), []).append(d)

    to_recurse = []
    for group in by_lower.values():
        if len(group) == 1:
            to_recurse.append(group[0])
            continue
        non_lower = [d for d in group if d.name != d.name.lower()]
        chosen = non_lower[0] if len(non_lower) == 1 else sorted(group, key=lambda d: d.name)[0]
        to_recurse.append(chosen)
        skipped.extend(d for d in group if d != chosen)

    files = [e for e in entries if e.is_file() and _is_image_file(e)]
    for d in to_recurse:
        files.extend(_walk_dedup(d, skipped))
    return files


def list_images(root: Path) -> list:
    root = Path(root)
    skipped = []
    files = _walk_dedup(root, skipped)
    if skipped:
        print(f"[external_image_dataset] skipped {len(skipped)} likely case-duplicate dir(s) "
              f"under {root} (kept one per collision group): {[str(s) for s in skipped]}")
    return sorted(files)


def load_image_any_format(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path)
    with Image.open(path) as im:
        return np.array(im)


class ExternalImageDataset(Dataset):
    def __init__(self, image_dirs, degrader: SpeckleAdditiveDegrader, tile_size: int = 256,
                 factor: int = 2, augment: bool = True):
        self.files = []
        for d in image_dirs:
            found = list_images(Path(d))
            if not found:
                raise ValueError(f"No images found under {d} (looked for {IMAGE_EXTENSIONS} and .npy)")
            self.files.extend(found)
        self.degrader = degrader
        self.tile_size = tile_size
        self.factor = factor
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        try:
            img = load_image_any_format(path)
        except Exception as e:
            raise RuntimeError(f"Failed to load external image {path}: {e}")

        gt, noisy = self.degrader.degrade_external(img, tile_size=self.tile_size, factor=self.factor)

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
        return noisy_t, gt_t, path.name


class MixedDataset(Dataset):
    """Samples from kla_dataset and external_dataset according to
    kla_weight : (1 - kla_weight) at each __getitem__ call. Length is
    len(kla_dataset) + len(external_dataset) so one "epoch" sees roughly
    that many total samples, drawn with replacement from whichever pool
    the coin flip picks - simple and avoids needing the two pools to be
    the same size."""

    def __init__(self, kla_dataset: Dataset, external_dataset: Dataset, kla_weight: float = 0.5):
        self.kla_dataset = kla_dataset
        self.external_dataset = external_dataset
        self.kla_weight = kla_weight
        self._length = len(kla_dataset) + (len(external_dataset) if external_dataset else 0)

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        if self.external_dataset is not None and len(self.external_dataset) > 0 and np.random.rand() > self.kla_weight:
            j = np.random.randint(0, len(self.external_dataset))
            return self.external_dataset[j]
        i = np.random.randint(0, len(self.kla_dataset))
        return self.kla_dataset[i]
