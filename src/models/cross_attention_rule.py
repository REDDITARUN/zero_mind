from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class RuleExtractionOutput:
    test_conditioned_latent: torch.Tensor
    context_latent: torch.Tensor


class CrossAttentionRuleExtractor(nn.Module):
    """
    Builds a rule context from train input/output latents and conditions test latent on it.
    """

    def __init__(self, dim: int = 256, heads: int = 8) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.fuse = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(
        self,
        test_latent: torch.Tensor,
        train_input_latents: List[torch.Tensor],
        train_output_latents: List[torch.Tensor],
    ) -> RuleExtractionOutput:
        # test_latent: [1, T, D], train latents list of [1, T_i, D]
        if len(train_input_latents) != len(train_output_latents):
            raise ValueError("Mismatched train latent lists")

        ctx_chunks = []
        for x_in, x_out in zip(train_input_latents, train_output_latents):
            # Difference carries transformation signature.
            ctx_chunks.append(x_in)
            ctx_chunks.append(x_out)
            if x_in.shape[1] != x_out.shape[1]:
                # ARC pairs can have different HxW across examples.
                # Align token lengths via adaptive pooling before diffing.
                tgt_t = min(x_in.shape[1], x_out.shape[1])
                x_in_aligned = F.adaptive_avg_pool1d(x_in.transpose(1, 2), output_size=tgt_t).transpose(1, 2)
                x_out_aligned = F.adaptive_avg_pool1d(x_out.transpose(1, 2), output_size=tgt_t).transpose(1, 2)
                delta = x_out_aligned - x_in_aligned
            else:
                delta = x_out - x_in
            ctx_chunks.append(delta)
        context = torch.cat(ctx_chunks, dim=1)

        q = self.norm_q(test_latent)
        kv = self.norm_kv(context)
        attn_out, _ = self.cross_attn(q, kv, kv, need_weights=False)
        fused = self.fuse(torch.cat([test_latent, attn_out], dim=-1))
        return RuleExtractionOutput(test_conditioned_latent=fused, context_latent=context)
