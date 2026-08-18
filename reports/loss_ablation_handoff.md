# Loss-term ablation — Kaggle handoff

**Goal:** isolate which composite-loss term(s) drove Stage B's LPIPS
improvement (0.289→0.163) — currently only known that the full stack
together did it. Two short (not-fully-converged) runs, directional evidence
only, using the existing `train_stageB.py` (already supports this via
`--sobel-weight`/`--range-weight`, no code changes needed).

**Blocker on my end:** I don't have `train_history_stageB.json` (the real
Stage B run's per-epoch history) — only the final `stageB_best.pt`
checkpoint was handed to me. I need this file to (a) show the real Stage B
trajectory's first ~8 epochs as the reference point for these ablations,
and (b) separately, to answer the "middle-epoch checkpoint" question
(whether an epoch between 15 and 20 offers a better PSNR/SSIM/LPIPS
balance than epoch 15). **Please pull this file from the Kaggle
notebook/working directory and hand it back** — likely at
`/kaggle/working/reports/train_history_stageB.json` per `train_stageB.py`'s
default `--report-dir`, unless the ops assistant set it elsewhere.

## Runs to execute

Both start from the same Stage A checkpoint, same data mix, same
hyperparameters as the real Stage B run — only loss weights and epoch count
differ. **Use separate `--checkpoint-dir`/`--report-dir` per run** so
outputs don't collide with each other or the real Stage B artifacts.

**Ablation 1 — Charbonnier + MS-SSIM + LPIPS only (no Sobel, no range-consistency):**
```bash
python train_stageB.py \
  --kla-gt-dir <same as real Stage B run> \
  --kla-noisy-dir <same as real Stage B run> \
  --external-dirs <same external dirs as real Stage B run> \
  --stageA-checkpoint <path to stageA_best.pt> \
  --checkpoint-dir /kaggle/working/checkpoints_ablation1 \
  --report-dir /kaggle/working/reports_ablation1 \
  --epochs 8 --batch-size 16 --lr 5e-5 \
  --sobel-weight 0 --range-weight 0
```

**Ablation 2 — Charbonnier + MS-SSIM + LPIPS + Sobel (no range-consistency):**
```bash
python train_stageB.py \
  --kla-gt-dir <same as real Stage B run> \
  --kla-noisy-dir <same as real Stage B run> \
  --external-dirs <same external dirs as real Stage B run> \
  --stageA-checkpoint <path to stageA_best.pt> \
  --checkpoint-dir /kaggle/working/checkpoints_ablation2 \
  --report-dir /kaggle/working/reports_ablation2 \
  --epochs 8 --batch-size 16 --lr 5e-5 \
  --range-weight 0
```

**Reference (full stack)** — once `train_history_stageB.json` is handed
back, its first 8 epochs serve as this comparison point directly; no new
run needed for this one.

## What to hand back

- `reports_ablation1/train_history_stageB.json`
- `reports_ablation2/train_history_stageB.json`
- The real Stage B run's `reports/train_history_stageB.json` (the blocker
  above — needed regardless of the ablations)

Each history file has per-epoch `val_psnr`, `val_ssim`, `val_lpips` already
logged (`train_stageB.py`'s existing `evaluate()` call) — no extra
instrumentation needed. Once these three files are back, I'll build the
epoch-by-epoch comparison plot and the middle-checkpoint analysis in one
pass, same rigor as everything else so far.
