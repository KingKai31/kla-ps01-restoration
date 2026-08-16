"""Builds a directory of deliberately broken/edge-case inputs and confirms
eval.py survives all of them without crashing the batch."""
import numpy as np
from pathlib import Path

out_dir = Path(r"C:\kla_adversarial_test\inputs")
out_dir.mkdir(parents=True, exist_ok=True)

# 1. normal valid input (control)
np.save(out_dir / "normal_128.npy", np.random.rand(128, 128).astype(np.float32))

# 2. corrupt/truncated file (not a real npy at all)
(out_dir / "corrupt.npy").write_bytes(b"this is not a valid npy file")

# 3. wrong dimensionality (3D)
np.save(out_dir / "wrong_ndim_3d.npy", np.random.rand(128, 128, 3).astype(np.float32))

# 4. wrong dimensionality (1D)
np.save(out_dir / "wrong_ndim_1d.npy", np.random.rand(128 * 128).astype(np.float32))

# 5. NaN/Inf values mixed into an otherwise valid image
arr_nan = np.random.rand(128, 128).astype(np.float32)
arr_nan[10:20, 10:20] = np.nan
arr_nan[50:55, 50:55] = np.inf
arr_nan[60:65, 60:65] = -np.inf
np.save(out_dir / "nan_inf.npy", arr_nan)

# 6. all-zero image (degenerate, could break sigma estimation in NLM fallback)
np.save(out_dir / "all_zero.npy", np.zeros((128, 128), dtype=np.float32))

# 7. all-constant nonzero image
np.save(out_dir / "all_constant.npy", np.full((128, 128), 0.73, dtype=np.float32))

# 8. tiny image
np.save(out_dir / "tiny_8x8.npy", np.random.rand(8, 8).astype(np.float32))

# 9. non-square image
np.save(out_dir / "nonsquare_100x150.npy", np.random.rand(100, 150).astype(np.float32))

# 10. extreme out-of-range values (way beyond normal speckle overshoot)
np.save(out_dir / "extreme_values.npy", (np.random.rand(128, 128).astype(np.float32) * 1000 - 500))

print(f"Built {len(list(out_dir.glob('*.npy')))} adversarial test files in {out_dir}")
for f in sorted(out_dir.glob("*.npy")):
    print(" -", f.name)
