"""
Synthetic NoisyLR generator, calibrated on Phase 1's measured decomposition.

Model: NoisyLR = box_downsample(GT) * M + A
  M ~ Gamma(shape=L, scale=1/L)   multiplicative speckle, mean 1
  A ~ N(mu_add, sigma_add)         additive noise

L and (mu_add, sigma_add) are drawn per synthesized image from the empirical
per-source-image fits measured across all 3200 real train pairs (bootstrap
sampling from the real distribution, not an assumed parametric shape) - the
severity varies a lot across sources (L range ~3.8-50.9), so a single fixed
value would misrepresent the data's true diversity.

Optional brightness correction: Phase 1's heteroscedasticity validation
showed the plain model overpredicts residual variance by up to ~15% at high
GT brightness (empirical noise grows slightly slower than GT^2/L predicts).
This is corrected with a linear L-vs-brightness curve fit directly from the
saved bin statistics in phase1_decomposition_summary.json - no new
measurement pass required.

No separate blur/Gaussian kernel is applied - Phase 1's FFT analysis could
not isolate one cleanly (injected noise dominates the spectrum), and the
multiplicative+additive model was judged sufficient pending the insurance
check in scripts/insurance_check.py.

`degrade()` operates on KLA's own 256x256 GT images (exact factor-2 box
downsample). `degrade_external()` extends this to arbitrary-size external
images (DIV2K/Flickr2K/DTD/SAR) for Stage B: it crops/resizes to a
256x256 tile first (matching KLA's GT resolution and this project's single
confirmed scale factor), then applies the identical noise model - the
degradation physics don't change just because the clean image came from a
different source.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def box_downsample(arr: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return arr
    h, w = arr.shape
    assert h % factor == 0 and w % factor == 0
    return arr.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))


def to_unit_grayscale(img: np.ndarray) -> np.ndarray:
    """Normalize an arbitrary image array (uint8, float, RGB or grayscale) to
    float32 grayscale in [0, 1]. External datasets arrive in mixed formats;
    KLA's own .npy files are already float32 in [0, 1] and pass through
    unchanged (mean-over-channels is a no-op for single-channel input)."""
    arr = img.astype(np.float64)
    if arr.ndim == 3:
        arr = arr.mean(axis=-1)
    if arr.max() > 1.5:  # heuristic: looks like 0-255 range, not 0-1
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def crop_or_resize_to_tile(img: np.ndarray, tile_size: int, rng: np.random.Generator) -> np.ndarray:
    """Random crop if the image is at least tile_size in both dims (typical
    for DIV2K/Flickr2K), else upscale to fit then crop (typical for smaller
    DTD/SAR patches). Never distorts aspect ratio beyond what's needed to
    reach tile_size on the shorter side."""
    h, w = img.shape
    if h < tile_size or w < tile_size:
        scale = tile_size / min(h, w)
        new_h, new_w = int(np.ceil(h * scale)), int(np.ceil(w * scale))
        y_idx = np.clip((np.arange(tile_size) / scale).astype(int), 0, h - 1)
        x_idx = np.clip((np.arange(tile_size) / scale).astype(int), 0, w - 1)
        img = img[y_idx][:, x_idx]
        h, w = img.shape
    top = rng.integers(0, h - tile_size + 1)
    left = rng.integers(0, w - tile_size + 1)
    return img[top: top + tile_size, left: left + tile_size]


class SpeckleAdditiveDegrader:
    def __init__(self, reports_dir: Path, seed: int = 0, brightness_correct: bool = True):
        reports_dir = Path(reports_dir)
        mult_df = pd.read_csv(reports_dir / "corrected_mult_fits.csv")
        add_df = pd.read_csv(reports_dir / "corrected_add_fits.csv")
        merged = mult_df.merge(add_df, on="file", suffixes=("_mult", "_add"))

        self.L_pool = merged["L"].to_numpy()
        self.mu_pool = merged["mean"].to_numpy()
        self.sigma_pool = merged["std"].to_numpy()

        with open(reports_dir / "phase1_decomposition_summary.json") as f:
            summ = json.load(f)

        self.L_ref = summ["mult"]["L_median"]
        self.brightness_correct = brightness_correct
        if brightness_correct:
            hc = summ["heteroscedasticity_check"]
            centers = np.array(hc["bin_centers"])
            emp_std = np.array(hc["empirical_std"])
            sigma_A_med = summ["add"]["median_of_stds"]
            var_mult = np.maximum(emp_std ** 2 - sigma_A_med ** 2, 1e-8)
            L_eff = (centers ** 2) / var_mult
            b, a = np.polyfit(centers, L_eff, 1)  # L_eff(x) ~= a + b*x
            self.brightness_a = float(a)
            self.brightness_b = float(b)
        else:
            self.brightness_a = self.L_ref
            self.brightness_b = 0.0

        self.rng = np.random.default_rng(seed)

    def _apply_noise(self, clean_down: np.ndarray) -> np.ndarray:
        idx = self.rng.integers(0, len(self.L_pool))
        L_base = self.L_pool[idx]
        mu_add = self.mu_pool[idx]
        sigma_add = self.sigma_pool[idx]

        if self.brightness_correct:
            L_curve = self.brightness_a + self.brightness_b * clean_down
            L_curve = np.clip(L_curve, 1.0, None)
            L_local = np.clip(L_base * (L_curve / self.L_ref), 1.0, None)
        else:
            L_local = np.full_like(clean_down, max(L_base, 1.0))

        M = self.rng.gamma(shape=L_local, scale=1.0 / L_local)
        A = self.rng.normal(loc=mu_add, scale=sigma_add, size=clean_down.shape)
        noisy = clean_down * M + A
        return noisy.astype(np.float32)

    def degrade(self, gt: np.ndarray, factor: int = 2) -> np.ndarray:
        """KLA's own 256x256 GT images - exact factor-2 box downsample."""
        clean_down = box_downsample(gt.astype(np.float64), factor)
        return self._apply_noise(clean_down)

    def degrade_external(self, clean_img: np.ndarray, tile_size: int = 256, factor: int = 2) -> tuple:
        """Arbitrary-size external clean image (any dtype/range/channels).
        Returns (gt_tile, noisy_tile) - gt_tile is the 256x256 crop used as
        ground truth, noisy_tile is its degraded 128x128 counterpart, same
        noise model as KLA's own data."""
        normalized = to_unit_grayscale(clean_img)
        gt_tile = crop_or_resize_to_tile(normalized, tile_size, self.rng)
        clean_down = box_downsample(gt_tile.astype(np.float64), factor)
        noisy_tile = self._apply_noise(clean_down)
        return gt_tile.astype(np.float32), noisy_tile
