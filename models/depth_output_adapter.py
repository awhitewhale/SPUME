"""Small residual depth adapter with a capacity-matched context channel."""

from __future__ import annotations

import torch
from torch import nn


class DepthOutputAdapter(nn.Module):
    """Predict a bounded log-depth residual; zero initialization is exact identity."""

    def __init__(self, hidden: int = 16, max_log_residual: float = 0.7):
        super().__init__()
        self.max_log_residual = float(max_log_residual)
        self.net = nn.Sequential(
            nn.Conv2d(6, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, rgb, depth, confidence, context):
        log_depth = depth.clamp_min(1e-4).log()
        log_conf = confidence.clamp_min(1e-4).log()
        features = torch.cat([rgb, log_depth[:, None], log_conf[:, None], context[:, None]], dim=1)
        residual = self.max_log_residual * torch.tanh(self.net(features)[:, 0])
        return (log_depth + residual).exp(), residual
