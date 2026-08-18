# Loss-term ablation — Kaggle handoff

**STATUS: DROPPED, not pending.** The Kaggle sessions running these three
configs became stuck/unresponsive across multiple attempts. Decision made
to stop waiting rather than keep re-attempting - this was always
supplementary analysis (explaining *why* the composite loss works, one
bullet on the Innovation slide), not a requirement for Stage B itself,
which ships and performs as documented in `reports/ppt_metrics_table.md`
regardless. No further action planned; kept below as a historical record
of what was attempted and the exact commands used, not as an open TODO.

**Goal:** isolate which composite-loss term(s) drove Stage B's LPIPS
improvement (0.289→0.163) — currently only known that the full stack
together did it. Three short (not-fully-converged) runs, directional
evidence only, using the existing `train_stageB.py` (already supports this
via `--sobel-weight`/`--range-weight`, no code changes needed).

**Resolved (not a blocker anymore):** the real Stage B run's per-epoch
history (`train_history_stageB.json`) was never written and isn't
recoverable — closed as a line of investigation, decision made to ship
`stageB_best.pt` as-is (best checkpoint by validation PSNR, a legitimate
criterion on its own). Consequence for this ablation: since there's no
epoch-by-epoch full-stack trajectory to compare against, **a third run —
full stack, same 8 epochs — is added below** so all three conditions are
epoch-matched and directly comparable to each other. Stage A's numbers and
Stage B epoch-15's numbers remain useful as bookend context, but aren't
epoch-matched to these 8-epoch runs, so treat comparisons against them as
directional only, not controlled.

## Runs to execute

All three start from the same Stage A checkpoint, same data mix, same
hyperparameters — only loss weights differ. **Use separate
`--checkpoint-dir`/`--report-dir` per run** so outputs don't collide with
each other.

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

**Ablation 3 — full stack (reference point, epoch-matched to 1 & 2):**
```bash
python train_stageB.py \
  --kla-gt-dir <same as real Stage B run> \
  --kla-noisy-dir <same as real Stage B run> \
  --external-dirs <same external dirs as real Stage B run> \
  --stageA-checkpoint <path to stageA_best.pt> \
  --checkpoint-dir /kaggle/working/checkpoints_ablation3_fullstack \
  --report-dir /kaggle/working/reports_ablation3_fullstack \
  --epochs 8 --batch-size 16 --lr 5e-5
```
(no loss-weight overrides — uses the defaults, i.e. the same full composite
loss as the real Stage B run)

## What to hand back

- `reports_ablation1/train_history_stageB.json`
- `reports_ablation2/train_history_stageB.json`
- `reports_ablation3_fullstack/train_history_stageB.json`

**This time, please confirm the file actually exists at that path on Kaggle
before ending the session** (`ls` or equivalent) — the real Stage B run's
history going missing is exactly the failure mode to avoid repeating here.

Each history file has per-epoch `val_psnr`, `val_ssim`, `val_lpips` already
logged (`train_stageB.py`'s existing `evaluate()` call) — no extra
instrumentation needed. Once these three files are back, I'll build the
epoch-by-epoch comparison plot (1 vs 2 isolates Sobel's marginal effect;
1/2 vs 3 isolates range-consistency's marginal effect) in one pass, same
rigor as everything else so far.
