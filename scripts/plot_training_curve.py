import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--history", type=Path, default=Path("reports/train_history.json"))
ap.add_argument("--out", type=Path, default=Path("reports/figures/stageA_training_curve.png"))
args = ap.parse_args()

with open(args.history) as f:
    history = json.load(f)

epochs = [h["epoch"] for h in history]
train_loss = [h["train_loss"] for h in history]
val_psnr = [h["val_psnr"] for h in history]
val_ssim = [h["val_ssim"] for h in history]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(epochs, train_loss, color="steelblue")
axes[0].set_title("Train loss (Charbonnier + MS-SSIM)")
axes[0].set_xlabel("epoch")

axes[1].plot(epochs, val_psnr, color="darkorange")
axes[1].set_title("Val PSNR (OOD-proxy split)")
axes[1].set_xlabel("epoch")

axes[2].plot(epochs, val_ssim, color="seagreen")
axes[2].set_title("Val SSIM (OOD-proxy split)")
axes[2].set_xlabel("epoch")

fig.tight_layout()
args.out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.out, dpi=150)
print(f"Saved {args.out}")

print(f"epoch 1:  loss={train_loss[0]:.4f} psnr={val_psnr[0]:.3f} ssim={val_ssim[0]:.4f}")
mid = len(epochs) // 2
print(f"epoch {epochs[mid]}: loss={train_loss[mid]:.4f} psnr={val_psnr[mid]:.3f} ssim={val_ssim[mid]:.4f}")
print(f"epoch {epochs[-1]}: loss={train_loss[-1]:.4f} psnr={val_psnr[-1]:.3f} ssim={val_ssim[-1]:.4f}")
