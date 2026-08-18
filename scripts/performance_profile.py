"""
Memory footprint and latency breakdown for run.py's actual inference path.

Honesty note on "batch of 16": run.py's real code processes images ONE AT A
TIME in a loop (no batching) - that's what's actually measured as "single
image" and "16 images, run.py's real sequential path" below. A genuinely
batched (batch_size=16 in one forward call) number is reported separately
and labeled as a hypothetical/upper-bound headroom figure, since it's not
what run.py does today - reporting it as if it were "run.py processing a
batch" would misrepresent the actual code path.

Latency breakdown uses local hardware (not the H100) to decompose the
76.4ms/image H100 figure into components - relative proportions (load vs
inference vs I/O) are expected to transfer directionally even though the
absolute numbers differ from the H100's, which remains the real end-to-end
figure for the feasibility slide.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import importlib.util
spec = importlib.util.spec_from_file_location("run_module", ROOT / "run.py")
run_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_module)


def make_synthetic_input(shape=(128, 128), seed=0):
    rng = np.random.default_rng(seed)
    arr = rng.normal(loc=0.55, scale=0.2, size=shape).astype(np.float32)
    return np.clip(arr, 0.0, 1.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=Path("models/checkpoint.pt"))
    ap.add_argument("--n-timing-images", type=int, default=50)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type != "cuda":
        print("WARNING: no CUDA device available locally - memory figures below are N/A on CPU, "
              "timing figures are still valid as relative-proportion evidence.")

    # ---------- Task 3: memory footprint ----------
    print("\n=== Memory footprint ===")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    t_load_start = time.perf_counter()
    model, upscale = run_module.load_model(args.checkpoint, device)
    t_load_end = time.perf_counter()
    load_time_s = t_load_end - t_load_start

    if device.type == "cuda":
        mem_after_load = torch.cuda.max_memory_allocated() / 1e6
        print(f"Peak VRAM after model load: {mem_after_load:.1f} MB")

    # single image, run.py's actual per-image path
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    x1 = make_synthetic_input()
    with torch.no_grad():
        xt = torch.from_numpy(x1).unsqueeze(0).unsqueeze(0).to(device)
        y = model(xt)
        y = run_module.suppress_checkerboard(y)
        y = y.clamp(0.0, 1.0)
        _ = y.squeeze(0).squeeze(0).cpu().numpy()
    if device.type == "cuda":
        torch.cuda.synchronize()
        mem_single = torch.cuda.max_memory_allocated() / 1e6
        print(f"Peak VRAM, single image (run.py's actual per-image path): {mem_single:.1f} MB")

    # 16 images, run.py's ACTUAL sequential path (no batching in real code)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(16):
            xi = make_synthetic_input()
            xt = torch.from_numpy(xi).unsqueeze(0).unsqueeze(0).to(device)
            y = model(xt)
            y = run_module.suppress_checkerboard(y)
            y = y.clamp(0.0, 1.0)
            _ = y.squeeze(0).squeeze(0).cpu().numpy()
    if device.type == "cuda":
        torch.cuda.synchronize()
        mem_16_sequential = torch.cuda.max_memory_allocated() / 1e6
        print(f"Peak VRAM, 16 images via run.py's REAL sequential (one-at-a-time) path: {mem_16_sequential:.1f} MB")

    # hypothetical: genuinely batched forward pass, batch_size=16 - NOT what
    # run.py does today, reported separately as headroom/feasibility context
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        batch = np.stack([make_synthetic_input(seed=i) for i in range(16)])
        xt = torch.from_numpy(batch).unsqueeze(1).to(device)  # (16, 1, 128, 128)
        y = model(xt)
        y = run_module.suppress_checkerboard(y)
        y = y.clamp(0.0, 1.0)
        _ = y.cpu().numpy()
    if device.type == "cuda":
        torch.cuda.synchronize()
        mem_16_batched = torch.cuda.max_memory_allocated() / 1e6
        print(f"Peak VRAM, HYPOTHETICAL genuinely-batched forward pass (batch_size=16, "
              f"NOT what run.py's real code does): {mem_16_batched:.1f} MB")

    # ---------- Task 4: latency breakdown ----------
    print("\n=== Latency breakdown (local hardware, component analysis) ===")
    print(f"Model/checkpoint load time (one-time cost): {load_time_s * 1000:.1f} ms")

    # warmup
    with torch.no_grad():
        for _ in range(5):
            xt = torch.from_numpy(make_synthetic_input()).unsqueeze(0).unsqueeze(0).to(device)
            _ = model(xt)
    if device.type == "cuda":
        torch.cuda.synchronize()

    n = args.n_timing_images
    inputs = [make_synthetic_input(seed=i) for i in range(n)]

    # pure forward-pass time (no I/O, no checkerboard suppression/sanitize)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for arr in inputs:
            xt = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)
            _ = model(xt)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    pure_inference_ms_per_image = (t1 - t0) / n * 1000

    # full run.py per-image path (forward + checkerboard + clamp + sanitize), no disk I/O
    t0 = time.perf_counter()
    with torch.no_grad():
        for arr in inputs:
            xt = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)
            y = model(xt)
            y = run_module.suppress_checkerboard(y)
            y = y.clamp(0.0, 1.0)
            out = y.squeeze(0).squeeze(0).cpu().numpy()
            out = run_module.sanitize_output(out)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    full_path_ms_per_image = (t1 - t0) / n * 1000

    # disk I/O: write inputs to temp npy, then time read+write for the same n images
    tmp_in = Path("reports/_perf_tmp_in")
    tmp_out = Path("reports/_perf_tmp_out")
    tmp_in.mkdir(parents=True, exist_ok=True)
    tmp_out.mkdir(parents=True, exist_ok=True)
    for i, arr in enumerate(inputs):
        np.save(tmp_in / f"{i:04d}.npy", arr)

    t0 = time.perf_counter()
    for i in range(n):
        arr = np.load(tmp_in / f"{i:04d}.npy")
        np.save(tmp_out / f"{i:04d}.npy", arr)  # same array, isolates I/O cost only
    t1 = time.perf_counter()
    disk_io_ms_per_image = (t1 - t0) / n * 1000

    for f in tmp_in.glob("*.npy"):
        f.unlink()
    for f in tmp_out.glob("*.npy"):
        f.unlink()
    tmp_in.rmdir()
    tmp_out.rmdir()

    print(f"Pure forward-pass inference: {pure_inference_ms_per_image:.2f} ms/image")
    print(f"Full run.py per-image path (forward + checkerboard suppress + clamp + sanitize): "
          f"{full_path_ms_per_image:.2f} ms/image")
    print(f"Disk I/O (read input .npy + write output .npy): {disk_io_ms_per_image:.2f} ms/image")
    print(f"Sum (load amortized over {n} images + full path + I/O): "
          f"{load_time_s * 1000 / n + full_path_ms_per_image + disk_io_ms_per_image:.2f} ms/image")

    print("\n=== Summary for report ===")
    print(f"n_timing_images={n}")
    print(f"load_time_ms_total={load_time_s * 1000:.1f}")
    print(f"pure_inference_ms_per_image={pure_inference_ms_per_image:.2f}")
    print(f"full_path_ms_per_image={full_path_ms_per_image:.2f}")
    print(f"disk_io_ms_per_image={disk_io_ms_per_image:.2f}")
    if device.type == "cuda":
        print(f"peak_vram_single_image_mb={mem_single:.1f}")
        print(f"peak_vram_16_sequential_mb={mem_16_sequential:.1f}")
        print(f"peak_vram_16_batched_hypothetical_mb={mem_16_batched:.1f}")


if __name__ == "__main__":
    main()
