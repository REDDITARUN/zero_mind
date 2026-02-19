from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from torch import nn
import torch.nn.functional as F

from src.models.ijepa_rope import RopeAttentionBlock


@dataclass
class DecodeOutput:
    logits: torch.Tensor
    draft_logits: torch.Tensor
    logits_trace: List[torch.Tensor]
    refinement_steps: int


class DraftCorrectDecoder(nn.Module):
    """Two-stage decoder inspired by System 1 / System 2 cognition.

    Stage 1 — Draft (System 1):
      Fast MLP refinement on the SAME tokens from the reasoner.
      Each token directly maps to a spatial position in the output grid.
      Strong, easy gradients; shared transition weights across steps.

    Stage 2 — Correct (System 2):
      Soft-embed the draft prediction back into latent space,
      fuse with the refined reasoning state, then apply
      self-attention with 2D RoPE to detect and fix spatial errors.
      Shared weights across correction steps (diffusion-style).

    The draft gives an easy learning signal (direct gradient path).
    The corrector learns to find and fix specific mistakes.
    Final output = corrected logits (residual on top of draft).
    """

    def __init__(
        self,
        dim: int = 256,
        num_colors: int = 10,
        draft_steps: int = 3,
        correct_steps: int = 3,
        heads: int = 8,
    ) -> None:
        super().__init__()
        self.draft_steps = draft_steps
        self.correct_steps = correct_steps

        self.draft_transition = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.draft_readout = nn.Linear(dim, num_colors)

        self.draft_embed = nn.Embedding(num_colors, dim)
        self.inject_norm = nn.LayerNorm(dim * 2)
        self.inject_proj = nn.Linear(dim * 2, dim)
        self.correct_attn = RopeAttentionBlock(dim=dim, heads=heads)
        self.correct_transition = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.correct_readout = nn.Linear(dim, num_colors)

    def forward(self, latent: torch.Tensor, h: int, w: int) -> DecodeOutput:
        z = latent
        trace: List[torch.Tensor] = []

        for _ in range(self.draft_steps):
            z = z + self.draft_transition(z)
            trace.append(self.draft_readout(z))
        draft_logits = trace[-1]

        draft_probs = F.softmax(draft_logits / 0.5, dim=-1)
        draft_latent = draft_probs @ self.draft_embed.weight
        z2 = self.inject_proj(self.inject_norm(torch.cat([z, draft_latent], dim=-1)))

        for _ in range(self.correct_steps):
            z2 = self.correct_attn(z2, h=h, w=w)
            z2 = z2 + self.correct_transition(z2)
            trace.append(self.correct_readout(z2))

        corrected_logits = trace[-1]

        return DecodeOutput(
            logits=corrected_logits,
            draft_logits=draft_logits,
            logits_trace=trace,
            refinement_steps=self.draft_steps + self.correct_steps,
        )
