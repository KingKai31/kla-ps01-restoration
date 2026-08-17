"""
Standalone inference script for KLA PS01 submission/benchmarking.

Usage (exact required form):
    python run.py <input_dir> <output_dir>

Reads every .npy in input_dir (float32 grayscale, any HxW), restores it, and
writes the result to output_dir under the same filename. Output arrays are
(H, W) - no channel dimension - matching KLA's own GT/NoisyLR .npy
convention exactly (confirmed by inspecting all provided data; the model's
raw NCHW tensor output is squeezed down to this before saving).

No network access required at any point: the model is built from raw
torch.nn primitives (no pretrained backbone), and weights are loaded from a
local checkpoint file only. No lpips/torchvision-pretrained/huggingface
import anywhere in this script's import chain - verified by explicit audit,
not assumed.

Self-contained by design: the model architecture (NAFNetSR and its building
blocks, normally developed in src/models/nafnet.py in the training repo) is
inlined directly below rather than imported, so this single file has no
dependency on any other project code - only pip-installed packages and the
local models/checkpoint.pt.

Guardrails (never let one bad input crash the whole batch):
  - A cheap, low-blend fixed blur suppresses pixel-shuffle checkerboard
    periodicity without materially softening real detail.
  - If the model errors on a given image, falls back to classical bicubic
    upsampling + non-local-means denoising, logging the trigger.
  - If even loading the input file fails, writes a neutral placeholder
    instead of crashing the run - logged as a hard failure, not silent.
  - sanitize_output() is the final gate before every single np.save call,
    regardless of which path produced the array (model success, classical
    fallback, or a placeholder): re-clips to [0,1], replaces any NaN/Inf,
    and asserts the result is actually finite, in-range, and (H,W) before
    it's allowed to be written. This does not trust the upstream clamp -
    KLA's spec makes NaN/Inf/out-of-range an explicit fail condition, so
    this is checked directly at the point of writing, not inferred.
  - A final summary reports fallback/failure counts so this is never
    silently masked as "it worked".
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.transform import resize as sk_resize
from skimage.restoration import denoise_nl_means, estimate_sigma

DEFAULT_SHAPE = (256, 256)  # best-effort placeholder shape when input can't even be read


# --- model architecture (inlined from src/models/nafnet.py - see that file
# in the training repo for design rationale) --------------------------------

class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = x.var(1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimplifiedChannelAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        return x * self.conv(self.pool(x))


class NAFBlock(nn.Module):
    def __init__(self, c: int, dw_expand: int = 2, ffn_expand: int = 2):
        super().__init__()
        dw_channel = c * dw_expand
        self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, dw_channel, kernel_size=1)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, kernel_size=3, padding=1, groups=dw_channel)
        self.sg1 = SimpleGate()
        self.sca = SimplifiedChannelAttention(dw_channel // 2)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, kernel_size=1)

        ffn_channel = c * ffn_expand
        self.norm2 = LayerNorm2d(c)
        self.conv4 = nn.Conv2d(c, ffn_channel, kernel_size=1)
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, kernel_size=1)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)))

    def forward(self, x):
        y = self.conv1(self.norm1(x))
        y = self.conv2(y)
        y = self.sg1(y)
        y = self.sca(y)
        y = self.conv3(y)
        x = x + y * self.beta

        y = self.conv4(self.norm2(x))
        y = self.sg2(y)
        y = self.conv5(y)
        x = x + y * self.gamma
        return x


class NAFNetSR(nn.Module):
    def __init__(self, img_channel: int = 1, width: int = 32,
                 enc_blk_nums=(1, 1, 1, 2), middle_blk_num: int = 2,
                 dec_blk_nums=(1, 1, 1, 1), upscale: int = 2):
        super().__init__()
        self.upscale = upscale
        self.intro = nn.Conv2d(img_channel, width, kernel_size=3, padding=1)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, chan * 2, kernel_size=2, stride=2))
            chan *= 2

        self.middle = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(
                nn.Conv2d(chan, chan * 2, kernel_size=1, bias=False),
                nn.PixelShuffle(2),
            ))
            chan //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        # fused pixel-shuffle upsampling head: single extra upscale at the very
        # end, after the U-Net has restored back to input resolution
        self.up_head = nn.Sequential(
            nn.Conv2d(chan, img_channel * (upscale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale),
        )

        self.padder_size = 2 ** len(enc_blk_nums)

    def _pad_to_multiple(self, x):
        _, _, h, w = x.shape
        mod = self.padder_size
        pad_h = (mod - h % mod) % mod
        pad_w = (mod - w % mod) % mod
        return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        _, _, h, w = inp.shape
        x = self._pad_to_multiple(inp)

        x = self.intro(x)
        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            skips.append(x)
            x = down(x)

        x = self.middle(x)

        for decoder, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            x = up(x)
            x = x + skip
            x = decoder(x)

        out = self.up_head(x)

        base = F.interpolate(inp, scale_factor=self.upscale, mode="bilinear", align_corners=False)
        out = out[:, :, : h * self.upscale, : w * self.upscale] + base
        return out


def load_model(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    upscale = ckpt.get("upscale", 2)
    model = NAFNetSR(img_channel=1, width=ckpt.get("width", 32), upscale=upscale)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, upscale


def suppress_checkerboard(y: torch.Tensor, blend: float = 0.15) -> torch.Tensor:
    kernel = torch.ones(1, 1, 3, 3, device=y.device, dtype=y.dtype) / 9.0
    blurred = F.conv2d(y, kernel, padding=1)
    return (1 - blend) * y + blend * blurred


def classical_fallback(noisy: np.ndarray, scale: int) -> np.ndarray:
    h, w = noisy.shape
    up = sk_resize(noisy, (h * scale, w * scale), order=3, mode="reflect", anti_aliasing=True)
    up = np.clip(up, 0.0, 1.0)
    try:
        sigma_est = float(np.mean(estimate_sigma(up)))
        denoised = denoise_nl_means(up, h=1.15 * sigma_est, fast_mode=True, patch_size=5, patch_distance=6)
    except Exception:
        denoised = up  # bicubic-only if NLM itself errors (e.g. degenerate constant image)
    return np.clip(denoised, 0.0, 1.0).astype(np.float32)


def sanitize_output(arr: np.ndarray) -> np.ndarray:
    """Final gate before every np.save call, regardless of which code path
    produced `arr`. Does not trust any upstream clip/clamp - KLA's spec
    makes NaN/Inf/out-of-[0,1] values and wrong shape explicit fail
    conditions, so this re-clips, replaces any NaN/Inf directly, and
    asserts the result is finite, in-range, and (H, W) before returning.
    If any assertion here ever fires, that's a real upstream bug to fix,
    not something to paper over."""
    arr = np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=0.5, posinf=1.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 1.0)
    assert np.all(np.isfinite(arr)), "sanitize_output: non-finite values survived nan_to_num"
    assert arr.min() >= 0.0 and arr.max() <= 1.0, \
        f"sanitize_output: values out of [0,1] survived clip - min={arr.min()}, max={arr.max()}"
    assert arr.ndim == 2, f"sanitize_output: expected (H, W), got shape {arr.shape}"
    return arr


def main():
    # Default checkpoint path resolves relative to this script's own
    # location, not the process's current working directory. A relative
    # default (plain Path("models/checkpoint.pt")) would break if this is
    # invoked as `python /some/path/run.py <in> <out>` from a different
    # cwd - plausible for an automated grading harness, not just a human
    # typing `cd` first. This must work with zero manual configuration
    # regardless of invocation style.
    default_checkpoint = Path(__file__).resolve().parent / "models" / "checkpoint.pt"

    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir", type=Path)
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--checkpoint", type=Path, default=default_checkpoint)
    ap.add_argument("--checkerboard-blend", type=float, default=0.15)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, upscale = load_model(args.checkpoint, device)

    files = sorted(args.input_dir.glob("*.npy"))
    print(f"Found {len(files)} input files. Device: {device}")

    load_failures, model_fallbacks, fallback_failures = [], [], []

    with torch.no_grad():
        for f in files:
            try:
                arr = np.load(f).astype(np.float32)
                if arr.ndim != 2:
                    raise ValueError(f"expected 2D grayscale array, got shape {arr.shape}")
            except Exception as e:
                print(f"ERROR: could not load {f.name} ({e}); writing placeholder, skipping", file=sys.stderr)
                load_failures.append(f.name)
                np.save(args.output_dir / f.name, sanitize_output(np.full(DEFAULT_SHAPE, 0.5, dtype=np.float32)))
                continue

            try:
                arr_clean = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
                x = torch.from_numpy(arr_clean).unsqueeze(0).unsqueeze(0).to(device)
                y = model(x)
                y = suppress_checkerboard(y, blend=args.checkerboard_blend)
                y = y.clamp(0.0, 1.0)
                out = y.squeeze(0).squeeze(0).cpu().numpy()
                if not np.all(np.isfinite(out)):
                    raise RuntimeError("model produced non-finite output")
            except Exception as e:
                model_fallbacks.append(f.name)
                print(f"WARNING: model failed on {f.name} ({e}); falling back to bicubic+NLM", file=sys.stderr)
                try:
                    out = classical_fallback(np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0), scale=upscale)
                except Exception as e2:
                    fallback_failures.append(f.name)
                    print(f"ERROR: fallback also failed on {f.name} ({e2}); writing placeholder", file=sys.stderr)
                    h, w = arr.shape
                    out = np.full((h * upscale, w * upscale), 0.5, dtype=np.float32)

            np.save(args.output_dir / f.name, sanitize_output(out))

    print(f"Wrote {len(files)} restored images to {args.output_dir}")
    print(f"Load failures: {len(load_failures)} {load_failures}")
    print(f"Classical-fallback triggers: {len(model_fallbacks)} {model_fallbacks}")
    print(f"Fallback-also-failed: {len(fallback_failures)} {fallback_failures}")
    if not load_failures and not model_fallbacks:
        print("Model succeeded on every image, no fallback triggered.")


if __name__ == "__main__":
    main()
