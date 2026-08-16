"""
Stage B composite loss: Charbonnier + MS-SSIM + light LPIPS + Sobel edge
loss + range-consistency penalty.

Kept as a separate module from Stage A's CharbonnierMSSSIMLoss (which stays
as-is, documented and reproducible) rather than modifying it in place.

- LPIPS weight starts conservative (default 0.075, within the requested
  0.05-0.1 range) to avoid hallucinated texture - this is a starting point
  to tune from validation curves, not a frozen final value.
- Sobel edge loss: targets the soft-edge/blur-like failure mode by directly
  penalizing gradient-magnitude mismatch between prediction and GT.
- Range-consistency penalty: penalizes output pixels outside a plausible
  clean-image range (GT is bounded in [0, 1]) - targets the speckle
  overshoot behavior KLA's docs call out, applied to the RAW (unclamped)
  network output so it can actually learn to stop overshooting, not just
  penalize a value that eval-time clamping would hide anyway.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import MS_SSIM
import lpips


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps2 = eps ** 2

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps2))


class SobelEdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)
        self.charbonnier = CharbonnierLoss()

    def _gradient_magnitude(self, x):
        gx = F.conv2d(x, self.kx, padding=1)
        gy = F.conv2d(x, self.ky, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-6)

    def forward(self, pred, target):
        pred_edges = self._gradient_magnitude(pred)
        target_edges = self._gradient_magnitude(target)
        return self.charbonnier(pred_edges, target_edges)


class RangeConsistencyPenalty(nn.Module):
    """Penalizes raw (pre-clamp) output pixels outside [low, high]."""

    def __init__(self, low: float = 0.0, high: float = 1.0):
        super().__init__()
        self.low = low
        self.high = high

    def forward(self, raw_pred):
        over = torch.clamp(raw_pred - self.high, min=0.0)
        under = torch.clamp(self.low - raw_pred, min=0.0)
        return torch.mean(over * over + under * under)


class StageBCompositeLoss(nn.Module):
    def __init__(self, charbonnier_weight: float = 1.0, msssim_weight: float = 0.2,
                 lpips_weight: float = 0.075, sobel_weight: float = 0.1,
                 range_weight: float = 0.05, data_range: float = 1.0):
        super().__init__()
        self.charbonnier = CharbonnierLoss()
        self.msssim = MS_SSIM(data_range=data_range, size_average=True, channel=1)
        self.lpips_fn = lpips.LPIPS(net="alex")
        for p in self.lpips_fn.parameters():
            p.requires_grad_(False)
        self.sobel = SobelEdgeLoss()
        self.range_penalty = RangeConsistencyPenalty(0.0, data_range)

        self.w_char = charbonnier_weight
        self.w_msssim = msssim_weight
        self.w_lpips = lpips_weight
        self.w_sobel = sobel_weight
        self.w_range = range_weight

    def forward(self, raw_pred, target):
        pred_c = torch.clamp(raw_pred, 0.0, 1.0)
        target_c = torch.clamp(target, 0.0, 1.0)

        char_loss = self.charbonnier(raw_pred, target)
        msssim_loss = 1.0 - self.msssim(pred_c, target_c)

        pred_lp = pred_c.repeat(1, 3, 1, 1) * 2 - 1
        target_lp = target_c.repeat(1, 3, 1, 1) * 2 - 1
        lpips_loss = self.lpips_fn(pred_lp, target_lp).mean()

        sobel_loss = self.sobel(pred_c, target_c)
        range_loss = self.range_penalty(raw_pred)

        total = (self.w_char * char_loss + self.w_msssim * msssim_loss
                 + self.w_lpips * lpips_loss + self.w_sobel * sobel_loss
                 + self.w_range * range_loss)

        parts = {
            "charbonnier": char_loss.item(),
            "ms_ssim_loss": msssim_loss.item(),
            "lpips": lpips_loss.item(),
            "sobel": sobel_loss.item(),
            "range_penalty": range_loss.item(),
        }
        return total, parts
