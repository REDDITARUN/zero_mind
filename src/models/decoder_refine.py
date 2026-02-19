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


class IterativeDecoder(nn.Module):
    """Dead-simple iterative MLP refinement on the same tokens.

    Each reasoner token maps to one output pixel. Shared transition
    weights across refinement steps (diffusion-style). No attention,
    no cross-connections — just refine each position independently.
    The reasoner already handled spatial reasoning via 2D RoPE attention.
    """

    def __init__(
        self,
        dim: int = 256,
        num_colors: int = 10,
        refine_steps: int = 6,
        **_kwargs,
    ) -> None:
        super().__init__()
        self.refine_steps = refine_steps
        self.transition = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.readout = nn.Linear(dim, num_colors)

    def forward(self, latent: torch.Tensor, h: int, w: int) -> DecodeOutput:
        z = latent
        trace: List[torch.Tensor] = []
        for _ in range(self.refine_steps):
            z = z + self.transition(z)
            trace.append(self.readout(z))

        return DecodeOutput(
            logits=trace[-1],
            logits_trace=trace,
            refinement_steps=self.refine_steps,
        )
