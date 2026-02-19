"""Bidirectional Mamba-style selective state space block.

Pure PyTorch implementation with JIT-compiled scan kernels.
For ARC grids (max 30x30 = 900 tokens), this is efficient enough.

Why Mamba for ARC?
- Attention asks: "which cells relate to which?" (pairwise, global)
- Mamba asks: "how do patterns flow across the grid?" (sequential, selective)
Together they give the model two complementary ways to understand spatial structure.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


@torch.jit.script
def _scan_fwd(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
) -> torch.Tensor:
    b, t, d = x.shape
    h = torch.zeros(b, d, A.shape[1], device=x.device, dtype=x.dtype)
    ys = torch.zeros_like(x)
    for i in range(t):
        dt_i = dt[:, i, :, None]
        h = h * torch.exp(dt_i * A) + x[:, i, :, None] * (dt_i * B[:, i, None, :])
        ys[:, i] = (h * C[:, i, None, :]).sum(-1) + x[:, i] * D
    return ys


@torch.jit.script
def _scan_rev(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
) -> torch.Tensor:
    b, t, d = x.shape
    h = torch.zeros(b, d, A.shape[1], device=x.device, dtype=x.dtype)
    ys = torch.zeros_like(x)
    for i in range(t - 1, -1, -1):
        dt_i = dt[:, i, :, None]
        h = h * torch.exp(dt_i * A) + x[:, i, :, None] * (dt_i * B[:, i, None, :])
        ys[:, i] = (h * C[:, i, None, :]).sum(-1) + x[:, i] * D
    return ys


class BiMambaBlock(nn.Module):
    """Bidirectional selective state space block.

    Scans the sequence in both directions with input-dependent (selective)
    state transitions. The bidirectional scan captures patterns flowing
    in both directions across the flattened grid.
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        self.d_inner = dim * expand
        self.d_state = d_state

        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, self.d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
        )

        self.x_proj = nn.Linear(self.d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

        A = torch.arange(1, d_state + 1).float().repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        residual = x
        x = self.norm(x)

        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)

        x_conv = self.conv1d(x_inner.transpose(1, 2))[:, :, :t].transpose(1, 2)
        x_conv = F.silu(x_conv)

        dt = F.softplus(self.dt_proj(x_conv))
        bc = self.x_proj(x_conv)
        B, C = bc.chunk(2, dim=-1)
        A = -torch.exp(self.A_log)

        y = _scan_fwd(x_conv, dt, A, B, C, self.D) + _scan_rev(x_conv, dt, A, B, C, self.D)
        y = y * F.silu(z)

        return residual + self.out_proj(y)
