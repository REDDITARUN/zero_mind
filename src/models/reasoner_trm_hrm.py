from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from torch import nn

from src.models.ijepa_rope import RopeAttentionBlock


@dataclass
class ReasonerOutput:
    latent: torch.Tensor
    q_logits: torch.Tensor
    steps_used: int
    active_experts: int


class ReasoningModule(nn.Module):
    """TRM/HRM-style reasoning module with input injection.

    Each call: hidden = hidden + injection, then through transformer blocks.
    Uses 2D RoPE so the reasoner is spatially aware.
    """

    def __init__(self, dim: int = 256, depth: int = 2, heads: int = 8) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([RopeAttentionBlock(dim=dim, heads=heads) for _ in range(depth)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_injection: torch.Tensor,
        h: int,
        w: int,
    ) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for block in self.blocks:
            hidden_states = block(hidden_states, h=h, w=w)
        return hidden_states


class TrmHrmReasoner(nn.Module):
    """TRM-inspired iterative reasoner with H/L hierarchy.

    Architecture follows the TRM reference: a shared ReasoningModule is used
    for both L-level (detail) and H-level (abstract) processing.  The nested
    H/L cycle with input re-injection and 1-step gradient trick matches the
    reference design.

    Our additions beyond the reference:
    - Dynamic expert growth/pruning for adaptive capacity
    - Q-head for self-assessment of correctness
    - Route-control modulation of effective H-cycles
    """

    def __init__(
        self,
        dim: int = 256,
        depth: int = 2,
        h_cycles: int = 3,
        l_cycles: int = 2,
        heads: int = 8,
        max_experts: int = 8,
        init_active_experts: int = 2,
    ) -> None:
        super().__init__()
        self.h_cycles = h_cycles
        self.l_cycles = l_cycles
        self.max_experts = max_experts
        self.active_experts = max(1, min(init_active_experts, max_experts))
        self.prune_cooldown_remaining = 0
        self.register_buffer("expert_utility_ema", torch.zeros(max_experts))

        # TRM-style: shared reasoning module for both L and H levels
        self.reasoning = ReasoningModule(dim=dim, depth=depth, heads=heads)

        # Learned initial states (broadcast to [B, T, D])
        self.h_init = nn.Parameter(torch.randn(dim) * 0.02)
        self.l_init = nn.Parameter(torch.randn(dim) * 0.02)

        # Q-head: self-assessment -- predicts whether current output is correct
        self.q_head = nn.Linear(dim, 1)

        # Dynamic expert modules (our innovation beyond references)
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

    def forward(self, x: torch.Tensor, route_control: torch.Tensor, h: int, w: int) -> ReasonerOutput:
        """
        x: [B, T, D] -- input embeddings (from cross-attention rule extractor)
        route_control: [B, 2] -- router alpha_sft/alpha_rl
        h, w: spatial dims of the test grid
        """
        b, t, d = x.shape

        z_H = self.h_init.view(1, 1, d).expand(b, t, d)
        z_L = self.l_init.view(1, 1, d).expand(b, t, d)

        rl_strength = route_control[:, 1].mean().item()
        effective_h = max(2, int(self.h_cycles * (0.5 + 0.5 * rl_strength)))

        # 1-step gradient trick (from HRM/TRM reference):
        # All iterations except the final H-cycle run without gradients.
        # This is much more memory-efficient and matches the reference design.
        if effective_h > 1:
            with torch.no_grad():
                for _h_step in range(effective_h - 1):
                    for _l_step in range(self.l_cycles):
                        z_L = self.reasoning(z_L, z_H + x, h, w)
                    z_H = self.reasoning(z_H, z_L, h, w)

        # Final H-cycle WITH gradients
        for _l_step in range(self.l_cycles):
            z_L = self.reasoning(z_L, z_H + x, h, w)
        z_H = self.reasoning(z_H, z_L, h, w)

        # Expert enrichment (our innovation)
        if self.active_experts > 0:
            exp_out = torch.zeros_like(z_H)
            for i in range(self.active_experts):
                exp_out = exp_out + self.experts[i](z_H)
            z_H = z_H + exp_out / float(self.active_experts)

        # Q-head: predict whether the model will get this task correct
        pooled = z_H.mean(dim=1)
        q_logits = self.q_head(pooled).squeeze(-1)

        total_steps = effective_h * self.l_cycles + self.l_cycles

        return ReasonerOutput(
            latent=z_H,
            q_logits=q_logits,
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
