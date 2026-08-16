import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.models.nafnet import NAFNetSR
from src.losses.charbonnier_msssim import CharbonnierMSSSIMLoss

torch.manual_seed(0)

model = NAFNetSR(img_channel=1, width=32, upscale=2)
n_params = sum(p.numel() for p in model.parameters())
print(f"params: {n_params/1e6:.2f}M")

x = torch.rand(2, 1, 128, 128)
y = model(x)
print("input", x.shape, "-> output", y.shape)
assert y.shape == (2, 1, 256, 256), f"unexpected output shape {y.shape}"

# forward-compatibility check: also accept 256->512 without code changes
x2 = torch.rand(1, 1, 256, 256)
y2 = model(x2)
print("input", x2.shape, "-> output", y2.shape)
assert y2.shape == (1, 1, 512, 512)

# odd input size not divisible by padder_size, check padding path
x3 = torch.rand(1, 1, 130, 130)
y3 = model(x3)
print("input", x3.shape, "-> output", y3.shape)
assert y3.shape == (1, 1, 260, 260)

gt = torch.rand(2, 1, 256, 256)
crit = CharbonnierMSSSIMLoss()
loss, parts = crit(y, gt)
print("loss", loss.item(), parts)

loss.backward()
print("backward OK, grad on intro.weight:", model.intro.weight.grad is not None)

print("SMOKE TEST PASSED")
