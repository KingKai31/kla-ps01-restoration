# Final submission verification

**Document history:** built 2026-08-18. A prior instruction to produce this
document was apparently given and then superseded by other work
(`scripts/quick_test_visualize.py`, a separate PPT-building session) before
it was ever created or committed - confirmed via a full search of this
repo's git history (no trace of this filename at any point, on any branch).
Rather than reconstruct a list of items that can't be verified to have
existed, this document is built fresh: every check below was re-run today
against the exact commit currently on `origin/main`, with real command
output as evidence, not carried over from memory of earlier passes.

**Verified against commit `4369a87`** (confirmed synced with `origin/main`
at time of writing - `git log --oneline -1` on both local and origin match).

---

## Verdict: submission-ready

Every KLA hard-gate requirement passes, re-verified fresh today against the
exact shipped checkpoint and the exact `requirements.txt` files as they
currently exist. One item remains open (loss-ablation integration, Task 4)
but it is supplementary analysis, not a hard gate - it does not block
submission readiness.

---

## A. KLA hard-gate requirements (re-verified fresh, today)

All checks below were re-run today, not assumed carried over from earlier
passes - this matters because `requirements.txt` changed since the last
full verification (PyWavelets was added; see Section C).

| # | Requirement | Status | Fresh evidence (today) |
|---|---|---|---|
| 1 | Entry script `run.py`, callable as `python run.py <input_dir> <output_dir>` | **PASS** | Ran exactly this form against 4 real never-seen `Test_NoisyLR` samples (000000, 000123, 000250, 000399) |
| 2 | Reads all `.npy` files from input dir | **PASS** | 4/4 input files found and processed (`Found 4 input files`) |
| 3 | Output dir created if missing | **PASS** | Ran against a directory confirmed nonexistent beforehand; present with 4 files after |
| 4 | Exactly one output per input, exact filename match | **PASS** | `[f.name for f in in_files] == [f.name for f in out_files]` → `True` |
| 5 | Output shape `(H, W)` | **PASS** | All 4: `ndim==2`, shape `(256,256)` for `(128,128)` input |
| 6 | Output values strictly in `[0,1]`, no NaN/Inf, `float32` | **PASS** | All 4: `dtype=float32`, `finite=True`, min/max within `[0.0000, 1.0000]` (exact per-file: 000000 `[0.0078,1.0000]`, 000123 `[0.0031,0.9405]`, 000250 `[0.0102,0.9454]`, 000399 `[0.0000,0.9875]`) |
| 7 | Correct resolution (128→256) | **PASS** | All 4: `resolution_ok = (out.shape == (in.shape[0]*2, in.shape[1]*2))` → `True` |
| 8 | Zero internet access at runtime | **PASS** | Monkey-patched `socket.socket.connect` and `socket.getaddrinfo` to hard-raise on any call, ran the full pipeline end-to-end against the real checkpoint: **0 network calls attempted**, correct output produced |
| 9 | Model weights bundled locally, loaded from disk only | **PASS** | `submission/Phoenix/models/checkpoint.pt` present; `torch.load` reads a local path only, no URL anywhere in the code |
| 10 | Exact folder structure `{run.py, requirements.txt, README.md, models/}` | **PASS** | `find submission/Phoenix -type f` → exactly these 4 paths, nothing extra |
| 11 | `requirements.txt` installs cleanly and produces correct output in total isolation | **PASS** | Built a brand-new venv from scratch, installed **strictly** from `submission/Phoenix/requirements.txt` (no other packages), ran `run.py` from it against the 4 real samples - output **bit-identical** (`np.array_equal` → `True` on all 4) to the dev-venv run |
| 12 | Works regardless of invocation cwd/path style | **PASS** | Re-tested the full combination together in the fresh venv: wrong cwd + absolute script path + real checkpoint - output still bit-identical (`True` on all 4) to the normal-invocation case |
| 13 | `run.py` in `submission/Phoenix/` is the actual shipped code, not a stale copy | **PASS** | `diff run.py submission/Phoenix/run.py` → identical, byte-for-byte |
| 14 | Shipped checkpoint identical between both locations | **PASS** | `sha256sum` on `models/checkpoint.pt` and `submission/Phoenix/models/checkpoint.pt` → both `277de182...66786feeb7`, exact match |
| 15 | Repo state matches what's actually being evaluated | **PASS** | `git status` clean (no uncommitted changes), local `HEAD` and `origin/main` both at `4369a87` |

---

## B. Model quality (val/OOD-proxy split, n=506)

Full detail, every statistical test, and per-image data:
[reports/ppt_metrics_table.md](ppt_metrics_table.md) - unchanged by this
verification pass, cited here for completeness.

| | PSNR | SSIM | LPIPS |
|---|---|---|---|
| Classical baseline (bicubic + NLM) | 20.58 | 0.374 | 0.562 |
| Stage A (KLA-only) | 28.26 | 0.740 | 0.289 |
| **Stage B (shipped)** | 28.09 | 0.731 | **0.163** |

- **AI gains ~7.5-7.7dB PSNR** over the classical baseline.
- Stage A→B is a **statistically proven two-sided tradeoff** (paired
  Wilcoxon, p<1e-44 for all three metrics) - LPIPS improves ~44% relative,
  PSNR/SSIM regress slightly, both directions equally proven, not just the
  improvement.
- **Inference: 76.4 ms/image on H100 SXM 80GB**, real end-to-end hardware
  measurement.
- **Scale generalization (256→512, an untrained resolution):** tested
  directly, ran correctly on all 15 test images, clearly beat both a
  bicubic and bicubic+NLM baseline. Precise claim: verified generalization
  to 256→512 (2×) - the same 2× factor as training, applied to an unseen
  input resolution - not "scale-agnostic."
- **Ensemble check (Stage A + Stage B averaged):** tested, **rejected**.
  Wins only 1/3 composite-scoring scenarios and costs ~2x inference time.
  Decision made: ship Stage B alone.
- **GT noise-ceiling check:** GT confirmed visually and quantitatively
  clean in flat regions - implied 36.1dB ceiling, well above both stages'
  measured ~28dB, not currently limiting either model.

---

## C. Engineering rigor

- **Reproducibility:** full RNG determinism (Python/NumPy/torch/CUDA/cuDNN
  + DataLoader worker seeding) proven via two identical training runs
  producing a **bit-identical checkpoint** (`torch.equal` on every tensor).
- **Formal test suite:** `tests/test_run_py_robustness.py`, **25 tests**,
  re-run today - **25 passed**. Covers corrupt files, wrong dimensionality,
  NaN/Inf-contaminated pixels, degenerate images, extreme values, and a
  permanent regression test for the PyWavelets bug below.
- **Real bug found and fixed (disclosed, not hidden):** `classical_fallback()`'s
  NLM denoising step silently never executed on any environment missing
  `PyWavelets` (an undeclared dependency of scikit-image's
  `estimate_sigma()`) - including every earlier "classical baseline"
  measurement in this project. Fixed by pinning `PyWavelets==1.9.0` in
  both `requirements.txt` files; **re-verified today** in a completely
  fresh venv (`pip install -r submission/Phoenix/requirements.txt` with no
  prior state) that it installs cleanly and the model still runs correctly
  end-to-end (Section A, checks 11-12).
- **License compliance:** DIV2K (academic-only), Flickr2K (disputed
  license), DTD, SAR/Sentinel all checked against their own stated terms,
  not assumed. Framed explicitly as a non-commercial hackathon submission.
  Checked whether the risk is load-bearing (not answerable without
  retraining; a cheap next step is documented if it matters later).
- **Repo cleanliness:** an independent fresh-eyes audit found and fixed a
  stale README status section (previously claimed "no model trained yet"
  while the rest of the project described a shipped Stage B checkpoint),
  two dead scratch scripts, a dead doc pointer, and an orphaned judge
  summary - all fixed and verified.
- **`scripts/quick_test_visualize.py`:** ad-hoc single-image visual
  testing tool, reuses `run.py`'s actual inference path (no parallel
  reimplementation), tested on 3 real cases including the non-`.npy`
  format-conversion path.

---

## D. Known limitations (disclosed, not hidden)

- Validation is an **unsupervised-clustering OOD proxy**, not a confirmed
  source-based split - real OOD performance is only confirmed against
  KLA's actual test set.
- **Scale generalization is verified at 256→512 (2×) only** - real
  reconstruction quality at a true higher resolution remains unverified
  (no real 512×512+ KLA data exists to test against; see Section B).
- Inputs **≤8px per side** fall back to the classical path (a real
  architectural constraint, safely handled, not expected to matter at
  KLA's real 128×128 resolution).
- **SAR training content is domain-different from KLA's** (terrain vs.
  semiconductor) - included for shared noise physics, not content
  similarity; a small rough check showed no degradation but n=10 is not
  proof.
- DIV2K/Flickr2K license terms are academic-research-only/disputed -
  standard practice in SR research, framed for this project's
  non-commercial context, not verified clean for commercial use.

---

## E. Open items (not blocking)

- **Task 4 - loss-ablation integration:** three short Kaggle training runs
  (isolating Sobel edge loss and range-consistency loss's individual
  contributions) were launched via `reports/loss_ablation_handoff.md` and
  have not yet returned. This is supplementary analysis explaining *why*
  the composite loss works, not a requirement for the model to work or
  for the submission to be valid - Stage B ships and performs as
  documented in Section B regardless of this result. Will be integrated
  into `reports/ppt_metrics_table.md` when it lands.
