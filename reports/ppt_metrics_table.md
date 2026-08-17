| Stage | Split | PSNR | SSIM | LPIPS | n |
|---|---|---|---|---|---|
| A (KLA-only), epoch 40 | Train (seen) | 28.63 | 0.786 | 0.236 | 508 |
| A (KLA-only), epoch 40 | Val/OOD-proxy | 28.26 | 0.740 | 0.289 | 506 |
| B (KLA+external), epoch 15 (best by val_psnr, not final epoch 20) | Train (seen) | 28.50 | 0.782 | 0.105 | 508 |
| B (KLA+external), epoch 15 (best by val_psnr, not final epoch 20) | Val/OOD-proxy | 28.09 | 0.731 | 0.163 | 506 |

**Gap A->B (val/OOD-proxy):** PSNR -0.18, SSIM -0.009, LPIPS -0.126 (improved - lower LPIPS is better)

Note: shipped Stage B checkpoint is epoch 15 (highest val_psnr seen during training, per standard best-checkpoint selection), not literal final epoch 20. Per-epoch history (train_history_stageB.json) not available locally to show the full curve.
