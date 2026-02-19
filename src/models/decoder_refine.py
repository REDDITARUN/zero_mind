from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from torch import nn

from src.models.ijepa_rope import RopeAttentionBlock


@dataclass
class DecodeOutput:
    logits: torch.Tensor
    logits_trace: List[torch.Tensor]
    refinement_steps: int


class ContentDecoder(nn.Module):
    """Cross-attention content generator with diffusion-style iterative refinement.

    Creates a canvas of output-size tokens (with learned 2D positional encoding),
    then iteratively refines them by:
      1. Cross-attending to the reasoned source latent (pull information)
      2. Self-attending with 2D RoPE (enforce spatial consistency)
      3. MLP transition (non-linear refinement)

    All weights are shared across refinement steps (true diffusion-style).
    Supports arbitrary output sizes up to max_grid × max_grid.
    """

    def __init__(
        self,
        dim: int = 256,
        num_colors: int = 10,
        refine_steps: int = 6,
        heads: int = 8,
        max_grid: int = 30,
    ) -> None:
        super().__init__()
        self.refine_steps = refine_steps

        self.canvas_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.row_embed = nn.Embedding(max_grid, dim)
        self.col_embed = nn.Embedding(max_grid, dim)

        self.cross_norm_q = nn.LayerNorm(dim)
        self.cross_norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)

        self.self_attn = RopeAttentionBlock(dim=dim, heads=heads)

        self.transition = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

        self.readout = nn.Linear(dim, num_colors)

    def forward(self, source_latent: torch.Tensor, h_out: int, w_out: int) -> DecodeOutput:
        b = source_latent.shape[0]
        t_out = h_out * w_out
        device = source_latent.device

        rows = torch.arange(h_out, device=device).repeat_interleave(w_out)
        cols = torch.arange(w_out, device=device).repeat(h_out)
        pos = self.row_embed(rows) + self.col_embed(cols)
        z = self.canvas_token.expand(b, t_out, -1) + pos.unsqueeze(0)

        trace: List[torch.Tensor] = []
        for _ in range(self.refine_steps):
            q = self.cross_norm_q(z)
            kv = self.cross_norm_kv(source_latent)
            attn_out, _ = self.cross_attn(q, kv, kv)
            z = z + attn_out

            z = self.self_attn(z, h=h_out, w=w_out)
            z = z + self.transition(z)

            trace.append(self.readout(z))

        return DecodeOutput(
            logits=trace[-1],
            logits_trace=trace,
            refinement_steps=self.refine_steps,
        )
