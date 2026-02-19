from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from torch import nn


@dataclass
class DecodeOutput:
    logits: torch.Tensor
    logits_trace: List[torch.Tensor]
    refinement_steps: int


class IterativeRefinementDecoder(nn.Module):
    """Diffusion-inspired iterative correction in latent space."""

    def __init__(self, dim: int = 256, num_colors: int = 10, refine_steps: int = 4) -> None:
        super().__init__()
        self.refine_steps = refine_steps
        self.transition = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.readout = nn.Linear(dim, num_colors)

    def forward(self, latent: torch.Tensor) -> DecodeOutput:
        z = latent
        trace: List[torch.Tensor] = []
        for _ in range(self.refine_steps):
            delta = self.transition(z)
            z = z + delta
            trace.append(self.readout(z))

        return DecodeOutput(logits=trace[-1], logits_trace=trace, refinement_steps=self.refine_steps)
