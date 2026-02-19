from __future__ import annotations

from typing import Tuple
import math

import torch
from torch import nn
import torch.nn.functional as F

from src.utils.grids import flatten_hw


def _build_2d_frequencies(rotary_dim: int, device: torch.device) -> torch.Tensor:
    half = rotary_dim // 2
    if half <= 0:
        raise ValueError("rotary_dim must be >= 2")
    inv_freq = 1.0 / (10000 ** (torch.arange(0, rotary_dim, 2, device=device).float() / rotary_dim))
    return inv_freq


def _apply_rotary(x: torch.Tensor, positions: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    sinusoid = torch.einsum("t,f->tf", positions.float(), inv_freq)
    sin = torch.sin(sinusoid)[None, None, :, :]
    cos = torch.cos(sinusoid)[None, None, :, :]

    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    rotated_even = x1 * cos - x2 * sin
    rotated_odd = x1 * sin + x2 * cos
    return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)


def apply_2d_rope(q: torch.Tensor, k: torch.Tensor, h: int, w: int) -> Tuple[torch.Tensor, torch.Tensor]:
    device = q.device
    d = q.shape[-1]
    half = d // 2
    q_row, q_col = q[..., :half], q[..., half:]
    k_row, k_col = k[..., :half], k[..., half:]

    row_pos = torch.arange(h, device=device).repeat_interleave(w)
    col_pos = torch.arange(w, device=device).repeat(h)
    inv_freq = _build_2d_frequencies(half, device=device)

    q_row = _apply_rotary(q_row, row_pos, inv_freq)
    k_row = _apply_rotary(k_row, row_pos, inv_freq)
    q_col = _apply_rotary(q_col, col_pos, inv_freq)
    k_col = _apply_rotary(k_col, col_pos, inv_freq)
    return torch.cat([q_row, q_col], dim=-1), torch.cat([k_row, k_col], dim=-1)


class RopeAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, t, d = x.shape
        y = self.norm1(x)
        qkv = self.qkv(y).view(b, t, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = apply_2d_rope(q, k, h=h, w=w)
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        attn = attn.transpose(1, 2).reshape(b, t, d)
        x = x + self.proj(attn)
        x = x + self.mlp(self.norm2(x))
        return x


class IJepaEncoder2DRoPE(nn.Module):
    """Pure encoder -- produces latent tokens from grid. No self-supervised loss."""

    def __init__(
        self,
        num_colors: int = 11,
        dim: int = 256,
        depth: int = 6,
        heads: int = 8,
    ) -> None:
        super().__init__()
        self.color_embed = nn.Embedding(num_colors, dim)
        self.blocks = nn.ModuleList([RopeAttentionBlock(dim=dim, heads=heads) for _ in range(depth)])

    def forward(self, grid_tokens: torch.Tensor) -> torch.Tensor:
        """grid_tokens: [B, H, W] -> latent_tokens: [B, H*W, D]"""
        b, h, w = grid_tokens.shape
        x = self.color_embed(grid_tokens)
        x = flatten_hw(x)
        for blk in self.blocks:
            x = blk(x, h=h, w=w)
        return x


class IJepaPredictor(nn.Module):
    """Lightweight predictor for I-JEPA self-supervised loss.

    Takes context latents (with mask tokens at target positions),
    runs through a small transformer, and outputs predicted latents
    at the target positions.
    """

    def __init__(self, dim: int = 256, depth: int = 2, heads: int = 8) -> None:
        super().__init__()
        self.proj_in = nn.Linear(dim, dim)
        self.mask_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.blocks = nn.ModuleList([RopeAttentionBlock(dim=dim, heads=heads) for _ in range(depth)])
        self.proj_out = nn.Linear(dim, dim)

    def forward(
        self,
        context_latent: torch.Tensor,
        target_indices: torch.Tensor,
        h: int,
        w: int,
    ) -> torch.Tensor:
        x = self.proj_in(context_latent)
        b, t, d = x.shape
        mask_tokens = self.mask_token.expand(b, target_indices.shape[0], d)
        x = x.clone()
        x[:, target_indices] = mask_tokens
        for blk in self.blocks:
            x = blk(x, h=h, w=w)
        return self.proj_out(x[:, target_indices])
