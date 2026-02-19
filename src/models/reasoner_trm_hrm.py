from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from src.models.ijepa_rope import apply_2d_rope


# ---------------------------------------------------------------------------
# HRM-faithful building blocks
# ---------------------------------------------------------------------------

def _rms_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps).to(x.dtype)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, expansion: float = 4.0) -> None:
        super().__init__()
        inter = ((int(round(expansion * dim * 2 / 3)) + 63) // 64) * 64
        self.gate_up = nn.Linear(dim, inter * 2, bias=False)
        self.down = nn.Linear(inter, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class HrmBlock(nn.Module):
    """HRM-style block: post-norm RMS + SwiGLU + 2D RoPE attention.

    Matches the reference HierarchicalReasoningModel_ACTV1Block:
      x = rms_norm(x + self_attn(x))
      x = rms_norm(x + mlp(x))
    but using 2D RoPE (better for grids) instead of 1D.
    """

    def __init__(self, dim: int, heads: int, expansion: float = 4.0, rms_eps: float = 1e-5) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.rms_eps = rms_eps

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.mlp = SwiGLU(dim, expansion)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, t, d = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = apply_2d_rope(q, k, h=h, w=w)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(b, t, d)

        x = _rms_norm(x + self.proj(attn), self.rms_eps)
        x = _rms_norm(x + self.mlp(x), self.rms_eps)
        return x


class HrmLevel(nn.Module):
    """One level (H or L) of the HRM hierarchy.

    Matches HierarchicalReasoningModel_ACTV1ReasoningModule:
      hidden = hidden + injection, then through layers.
    """

    def __init__(self, dim: int, num_layers: int, heads: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([HrmBlock(dim=dim, heads=heads) for _ in range(num_layers)])

    def forward(self, hidden: torch.Tensor, injection: torch.Tensor, h: int, w: int) -> torch.Tensor:
        hidden = hidden + injection
        for block in self.blocks:
            hidden = block(hidden, h=h, w=w)
        return hidden


# ---------------------------------------------------------------------------
# Main reasoner
# ---------------------------------------------------------------------------

@dataclass
class ReasonerOutput:
    latent: torch.Tensor
    steps_used: int
    active_experts: int


class TrmHrmReasoner(nn.Module):
    """Hierarchical Reasoning Model — faithful to reference HRM.

    Key design choices matching the reference (hrm_act_v1.py):
      1. Separate H_level and L_level with independent parameters
      2. 1-step gradient trick: all iterations except the final L+H run
         in torch.no_grad(), only the last step backpropagates
      3. Post-norm RMS + SwiGLU blocks (matching reference block design)
      4. Fixed initial states (buffers, not parameters)

    Our improvements over reference:
      - 2D RoPE instead of 1D (better for ARC grids)
      - Dynamic expert growth/pruning on top of final H state
    """

    def __init__(
        self,
        dim: int = 256,
        h_layers: int = 3,
        l_layers: int = 3,
        h_cycles: int = 3,
        l_cycles: int = 2,
        heads: int = 8,
        max_experts: int = 8,
        init_active_experts: int = 2,
        # Legacy compat — accepted, used as fallback
        depth: int | None = None,
    ) -> None:
        super().__init__()
        if depth is not None:
            h_layers = depth
            l_layers = depth

        self.h_cycles = h_cycles
        self.l_cycles = l_cycles
        self.max_experts = max_experts
        self.active_experts = max(1, min(init_active_experts, max_experts))
        self.prune_cooldown_remaining = 0
        self.register_buffer("expert_utility_ema", torch.zeros(max_experts))

        self.H_level = HrmLevel(dim=dim, num_layers=h_layers, heads=heads)
        self.L_level = HrmLevel(dim=dim, num_layers=l_layers, heads=heads)

        self.register_buffer("H_init", torch.randn(dim) * 0.02)
        self.register_buffer("L_init", torch.randn(dim) * 0.02)

        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, dim),
                    nn.GELU(),
                    nn.Linear(dim, dim),
                )
                for _ in range(max_experts)
            ]
        )

    def forward(self, x: torch.Tensor, h: int, w: int) -> ReasonerOutput:
        b, t, d = x.shape

        z_H = self.H_init.view(1, 1, d).expand(b, t, d)
        z_L = self.L_init.view(1, 1, d).expand(b, t, d)

        last_h = self.h_cycles - 1
        last_l = self.l_cycles - 1

        # 1-step gradient trick (matching reference HRM):
        # All iterations except the final L+H step run without grad.
        # Only the last L_level + H_level call records activations for backprop.
        with torch.no_grad():
            for h_step in range(self.h_cycles):
                for l_step in range(self.l_cycles):
                    if not (h_step == last_h and l_step == last_l):
                        z_L = self.L_level(z_L, z_H + x, h, w)
                if h_step != last_h:
                    z_H = self.H_level(z_H, z_L, h, w)

        # Final iteration — this is the only step that gets gradients.
        # Gradients flow: loss → z_H → H_level → z_L → L_level → x → encoder
        z_L = self.L_level(z_L, z_H + x, h, w)
        z_H = self.H_level(z_H, z_L, h, w)

        if self.active_experts > 0:
            exp_out = torch.zeros_like(z_H)
            for i in range(self.active_experts):
                exp_out = exp_out + self.experts[i](z_H)
            z_H = z_H + exp_out / float(self.active_experts)

        total_steps = self.h_cycles * self.l_cycles + self.h_cycles

        return ReasonerOutput(
            latent=z_H,
            steps_used=total_steps,
            active_experts=self.active_experts,
        )

    @torch.no_grad()
    def adapt_structure(
        self,
        reward: float,
        novelty: float,
        uncertainty: float,
        hard_cap: int | None = None,
        target_cap: int | None = None,
        grow_threshold: float = 0.55,
        prune_threshold: float = 0.15,
        ema_momentum: float = 0.9,
        prune_cooldown_steps: int = 0,
    ) -> tuple[bool, bool]:
        pressure = 0.45 * novelty + 0.35 * uncertainty + 0.20 * max(0.0, 0.5 - reward)
        stability = 0.6 * reward + 0.2 * (1.0 - novelty) + 0.2 * (1.0 - uncertainty)

        grown = False
        pruned = False
        utility_signal = max(0.0, min(1.0, stability))
        self.expert_utility_ema[: self.active_experts] = (
            ema_momentum * self.expert_utility_ema[: self.active_experts]
            + (1.0 - ema_momentum) * utility_signal
        )

        max_allowed = self.max_experts if hard_cap is None else max(1, min(self.max_experts, hard_cap))
        if pressure > grow_threshold and self.active_experts < max_allowed:
            self.active_experts += 1
            grown = True
            self.prune_cooldown_remaining = max(self.prune_cooldown_remaining, max(0, prune_cooldown_steps))

        if target_cap is not None and self.active_experts > target_cap:
            pressure = min(pressure, prune_threshold - 1e-3)

        if self.prune_cooldown_remaining > 0:
            self.prune_cooldown_remaining -= 1

        if (
            self.prune_cooldown_remaining <= 0
            and pressure < prune_threshold
            and stability > 0.75
            and self.active_experts > 1
            and float(self.expert_utility_ema[self.active_experts - 1]) < 0.4
        ):
            self.active_experts -= 1
            pruned = True

        return grown, pruned
