# run.py compliance checklist — KLA PS01 final requirements

Verified against `submission/Phoenix/run.py`, checkpoint = Stage A
(`models/checkpoint.pt`), on 5 real never-seen `Test_NoisyLR` samples
(indices 0, 5, 123, 250, 399 of 400) plus a genuinely nonexistent output
directory. Full test script: the compliance run this table is built from.

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 1 | Entry script named `run.py`, callable as `python run.py <input-dir> <output-dir>` | **PASS** | File renamed via `git mv`; positional args confirmed in that exact order in `argparse` setup |
| 2 | Reads ALL `.npy` files from input dir | **PASS** | `sorted(args.input_dir.glob("*.npy"))` — 5/5 input files found and processed |
| 3 | Creates output dir if missing | **PASS** | Ran against `outputs_DOES_NOT_EXIST_YET`, confirmed nonexistent beforehand (`ls` → No such file or directory), directory + 5 files present after |
| 4 | Exactly one output per input, exact filename match | **PASS** | 5 inputs → 5 outputs; `[f.name for f in in_files] == [f.name for f in out_files]` → `True` |
| 5 | Output shape `(H,W)` or `(H,W,1)` — confirmed which one | **PASS, (H,W)** | Raw model output is `(1,1,256,256)` NCHW (confirmed via direct execution), squeezed to plain `(256,256)` before saving — matches KLA's own GT/NoisyLR `.npy` convention exactly. All 5 outputs: `ndim==2` → `True` |
| 6 | Output values strictly in `[0,1]`, no NaN/Inf | **PASS** | `sanitize_output()` is a universal final gate before every `np.save`, applied regardless of code path. All 5: `range_ok=True`, `finite_ok=True` (min/max printed per file, all within `[0.009, 1.000]`) |
| 7 | Correct target resolution (128→256) | **PASS** | All 5: input `(128,128)` → output `(256,256)`, `resolution_ok=True` |
| 8 | Zero internet access at runtime | **PASS** | Import chain audited (grep for `lpips\|torchvision\|huggingface\|torch.hub\|download\|http` — zero matches); proven by monkey-patching `socket.connect`/`socket.getaddrinfo` to hard-raise on any call, running the full pipeline end-to-end: 0 network calls, correct output |
| 9 | Model weights bundled locally under `models/`, loaded from disk only | **PASS** | `models/checkpoint.pt` present in submission folder; `torch.load` reads local path only, no URL |
| 10 | Exact folder structure `{run.py, requirements.txt, README.md, models/}` | **PASS** | `run.py` is fully self-contained (model architecture inlined, no `src/` dependency) — no code folder needed beyond the required four items |
| 11 | `requirements.txt` = exact pinned versions | **PASS** | Built from a clean venv containing only `run.py`'s actual dependencies, frozen exactly; verified end-to-end against a completely fresh venv installing strictly from that file |
| 12 | README lets a stranger set up and run with zero questions | **PASS** (fresh-eyes pass done) | Added: Python version requirement, venv steps, concrete example command, NVIDIA driver prerequisite note, clarified metrics are on our own val split not KLA's official test set, documented the now-fixed cwd-independence |

## One real bug found and fixed during this pass (not previously caught)

`--checkpoint`'s default was a relative path (`Path("models/checkpoint.pt")`),
which resolves against the *process's current working directory*, not
`run.py`'s own location. If invoked as `python /some/path/run.py <in> <out>`
from a different cwd — plausible for an automated grading harness — this
would have failed to find the checkpoint. Fixed by resolving the default via
`Path(__file__).resolve().parent`. Verified: the exact failing scenario
(wrong cwd + absolute script path) now works, and output is bit-identical to
the normal-invocation case (`np.array_equal` → `True` on all samples), so
the fix changed nothing about correctness, only robustness to how the
script gets invoked.
