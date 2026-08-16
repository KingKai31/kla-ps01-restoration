"""Stage A loss: Charbonnier + MS-SSIM only, per the KLA hackathon plan."""

import torch
import torch.nn as nn
from pytorch_msssim import MS_SSIM


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps2 = eps ** 2

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps2))


class CharbonnierMSSSIMLoss(nn.Module):
    def __init__(self, charbonnier_weight: float = 1.0, msssim_weight: float = 0.2,
                 data_range: float = 1.0):
        super().__init__()
        self.charbonnier = CharbonnierLoss()
        self.msssim = MS_SSIM(data_range=data_range, size_average=True, channel=1)
        self.charbonnier_weight = charbonnier_weight
        self.msssim_weight = msssim_weight

    def forward(self, pred, target):
        pred_c = torch.clamp(pred, 0.0, 1.0)
        target_c = torch.clamp(target, 0.0, 1.0)
        char_loss = self.charbonnier(pred, target)
        ms_ssim_val = self.msssim(pred_c, target_c)
        msssim_loss = 1.0 - ms_ssim_val
        total = self.charbonnier_weight * char_loss + self.msssim_weight * msssim_loss
        return total, {"charbonnier": char_loss.item(), "ms_ssim_loss": msssim_loss.item()}
