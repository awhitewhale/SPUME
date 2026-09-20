"""Identity-initialized controller for token-level residual writes."""

from __future__ import annotations

import torch
from torch import nn


class SurfaceWriteController(nn.Module):
    """Map frozen evidence channels to a surface-write multiplier in (0, 1)."""

    def __init__(self, channels: int = 5, initial_gate: float = 0.99):
        super().__init__()
        self.linear = nn.Linear(channels, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.constant_(self.linear.bias, torch.logit(torch.tensor(initial_gate)).item())

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(evidence).squeeze(-1))


def register_residual_write_gate(block: nn.Module, gate: torch.Tensor):
    """Gate only the residual update of one global block; special tokens stay unchanged."""

    def hook(_module, args, kwargs, output):
        tokens = args[0]
        sequence, patch_h, patch_w = gate.shape
        token_count = tokens.shape[1] // sequence
        patch_count = patch_h * patch_w
        special = token_count - patch_count
        if special < 0:
            raise ValueError("Gate has more patch entries than the token sequence")
        multiplier = torch.ones(
            (1, sequence, token_count, 1), device=tokens.device, dtype=tokens.dtype
        )
        multiplier[:, :, special:, 0] = gate.reshape(sequence, -1).to(tokens.dtype)
        shaped_input = tokens.view(1, sequence, token_count, -1)
        shaped_output = output.view_as(shaped_input)
        return (shaped_input + multiplier * (shaped_output - shaped_input)).reshape_as(output)

    return block.register_forward_hook(hook, with_kwargs=True)
