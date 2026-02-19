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
    """Pure encoder -- produces latent tokens from grid.

    When use_mamba=True, alternates attention blocks (global/spatial via 2D RoPE)
    with Mamba blocks (sequential/selective scanning).
    """

    def __init__(
        self,
        num_colors: int = 11,
        dim: int = 256,
        depth: int = 6,
        heads: int = 8,
        use_mamba: bool = False,
    ) -> None:
        super().__init__()
        self.color_embed = nn.Embedding(num_colors, dim)
        self.use_mamba = use_mamba

        if use_mamba:
            from src.models.mamba_block import BiMambaBlock

            blocks: list[nn.Module] = []
            for i in range(depth):
                if i % 2 == 0:
                    blocks.append(BiMambaBlock(dim=dim))
                else:
                    blocks.append(RopeAttentionBlock(dim=dim, heads=heads))
            self.blocks = nn.ModuleList(blocks)
            self._block_types = ["mamba" if i % 2 == 0 else "attn" for i in range(depth)]
        else:
            self.blocks = nn.ModuleList([RopeAttentionBlock(dim=dim, heads=heads) for _ in range(depth)])
            self._block_types = ["attn"] * depth

    def forward(self, grid_tokens: torch.Tensor) -> torch.Tensor:
        """grid_tokens: [B, H, W] -> latent_tokens: [B, H*W, D]"""
        b, h, w = grid_tokens.shape
        x = self.color_embed(grid_tokens)
        x = flatten_hw(x)
        for blk, btype in zip(self.blocks, self._block_types):
            if btype == "mamba":
                x = blk(x)
            else:
                x = blk(x, h=h, w=w)
        return x


class DINOHead(nn.Module):
    """DINO projection head for self-distillation.

    Projects pooled latent to a prototypical space where
    teacher-student distillation produces stable representations.
    """

    def __init__(self, dim: int = 256, hidden_dim: int = 512, out_dim: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


def dino_loss(
    student_out: torch.Tensor,
    teacher_out: torch.Tensor,
    center: torch.Tensor,
    student_temp: float = 0.1,
    teacher_temp: float = 0.07,
) -> torch.Tensor:
    """DINO cross-entropy loss with centering + sharpening."""
    teacher_probs = F.softmax((teacher_out - center) / teacher_temp, dim=-1)
    student_log_probs = F.log_softmax(student_out / student_temp, dim=-1)
    return -torch.sum(teacher_probs * student_log_probs, dim=-1).mean()
