import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.models.nafnet import NAFNetSR
from src.losses.stageB_composite import StageBCompositeLoss

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

model = NAFNetSR(img_channel=1, width=32, upscale=2).to(device)
crit = StageBCompositeLoss().to(device)

x = torch.rand(2, 1, 128, 128, device=device)
gt = torch.rand(2, 1, 256, 256, device=device)

pred = model(x)
print("pred shape", pred.shape, "range", pred.min().item(), pred.max().item())

loss, parts = crit(pred, gt)
print("total loss:", loss.item())
print("parts:", parts)

loss.backward()
print("backward OK, grad on intro.weight:", model.intro.weight.grad is not None)
print("SMOKE TEST PASSED")
