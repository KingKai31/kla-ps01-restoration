# Data layout

This folder is not tracked by git (too large for the repo). Point the scripts
at your local copies via `--data-root`, or place them here matching this layout:

```
data/
  train/
    GT/        # 000000.npy ... paired ground-truth images (512x512 or 256x256, float32)
    NoisyLR/   # 000000.npy ... paired degraded images (256x256 or 128x128, float32)
  test/
    NoisyLR/   # unpaired degraded test samples, same naming
```

Pairing is by identical filename between `GT/` and `NoisyLR/`.

Current known local sources (update if these move):
- Train pairs: `C:\Users\ANANNYA\Downloads\train (1)\train\{GT,NoisyLR}` (3200 pairs)
- Test sample (degraded-only, no GT): `C:\Users\ANANNYA\Downloads\Test_NoisyLR (1)\NoisyLR` (400 samples)
- Official KLA test set: not yet released (per hackathon schedule)
