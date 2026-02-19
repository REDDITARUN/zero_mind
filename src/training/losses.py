from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.models.unified_model import UnifiedForwardOutput


@dataclass
class LossBreakdown:
    total: torch.Tensor
    l_sft: torch.Tensor
    l_qhalt: torch.Tensor
    l_ijepa: torch.Tensor
    l_eff: torch.Tensor
    l_route: torch.Tensor
    l_consistency: torch.Tensor
    reward: torch.Tensor
    exact_match: torch.Tensor
    eff_sft_weight: torch.Tensor
    eff_rl_weight: torch.Tensor
    explore_strength: torch.Tensor


def _exact_match(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = logits.argmax(dim=1).squeeze(0)
    return (pred == target).all().float()


def compute_unified_loss(
    out: UnifiedForwardOutput,
    target_grid: torch.Tensor,
    step: int,
    total_steps: int,
    sft_warmup_steps: int = 200,
    min_sft_weight: float = 0.25,
    sft_refresh_interval: int = 0,
    sft_refresh_span: int = 0,
    sft_refresh_min: float = 0.6,
    routing_explore_steps: int = 300,
    routing_explore_floor: float = 0.15,
    routing_entropy_bonus: float = 0.02,
    w_ijepa: float = 0.1,
    w_qhalt: float = 0.5,
    w_eff: float = 0.1,
    expert_budget_coeff: float = 0.25,
    w_route: float = 0.05,
    w_consistency: float = 0.05,
    compute_penalty_coeff: float = 0.01,
    reward_exact_coeff: float = 1.0,
    # Legacy compat -- kept for adapt_structure but not used in loss
    reward_pixel_coeff: float = 0.2,
) -> LossBreakdown:
    # ---- SFT term ----
    l_sft = F.cross_entropy(out.test_logits, target_grid.unsqueeze(0))

    # ---- Reward (for adapt_structure and metrics, not directly in loss) ----
    exact = _exact_match(out.test_logits, target_grid)
    pixel_acc = (out.test_logits.argmax(dim=1).squeeze(0) == target_grid).float().mean()
    compute_cost = compute_penalty_coeff * (out.reasoner.steps_used + out.decoder.refinement_steps)
    reward = reward_exact_coeff * exact + reward_pixel_coeff * pixel_acc - compute_cost

    # ---- Q-halt loss: self-assessment (from HRM/TRM reference) ----
    # The Q-head predicts whether the model got this task correct.
    # This is the RL component: the model learns to assess its own confidence.
    is_correct = exact.detach()
    l_qhalt = F.binary_cross_entropy_with_logits(
        out.reasoner.q_logits,
        is_correct.expand_as(out.reasoner.q_logits),
    )

    # ---- Efficiency and consistency ----
    l_eff = torch.tensor(
        float(out.reasoner.steps_used + out.decoder.refinement_steps + expert_budget_coeff * out.reasoner.active_experts),
        device=target_grid.device,
    )
    if len(out.decoder.logits_trace) > 1:
        diffs = []
        for a, b in zip(out.decoder.logits_trace[:-1], out.decoder.logits_trace[1:]):
            diffs.append((b - a).abs().mean())
        l_consistency = torch.stack(diffs).mean()
    else:
        l_consistency = torch.zeros((), device=target_grid.device)

    # ---- Router regularization ----
    l_route = out.router.routing_loss
    alpha_sft = out.router.alpha_sft.mean()
    alpha_rl = out.router.alpha_rl.mean()

    # Router exploration schedule
    explore_strength_val = max(0.0, 1.0 - (step / max(1, routing_explore_steps)))
    explore_strength = torch.tensor(explore_strength_val, device=target_grid.device)
    random_sft = torch.rand((), device=target_grid.device)
    explored_sft = (1.0 - explore_strength) * alpha_sft + explore_strength * random_sft
    explored_sft = torch.clamp(explored_sft, min=routing_explore_floor, max=1.0 - routing_explore_floor)

    # SFT warmup + explored router mix
    warm = max(0.0, 1.0 - (step / max(1, sft_warmup_steps)))
    eff_sft = torch.clamp(
        torch.tensor(warm, device=target_grid.device) + (1.0 - warm) * explored_sft,
        min=min_sft_weight,
        max=0.98,
    )
    # Periodic SFT refresh windows
    if sft_refresh_interval > 0 and sft_refresh_span > 0:
        phase = (step - 1) % sft_refresh_interval
        if phase < sft_refresh_span:
            eff_sft = torch.clamp(eff_sft, min=sft_refresh_min, max=0.98)
    eff_rl = 1.0 - eff_sft

    # Router entropy bonus
    router_entropy = -(alpha_sft * torch.log(alpha_sft + 1e-9) + alpha_rl * torch.log(alpha_rl + 1e-9))
    l_route = l_route - routing_entropy_bonus * explore_strength * router_entropy

    # ---- Total loss ----
    # When eff_sft is high (SFT mode): strong supervised learning
    # When eff_sft is low (RL mode): weaker supervision, more self-assessment
    total = (
        eff_sft * l_sft
        + w_qhalt * l_qhalt
        + w_ijepa * out.ijepa_loss
        + w_eff * l_eff
        + w_route * l_route
        + w_consistency * l_consistency
    )
    return LossBreakdown(
        total=total,
        l_sft=l_sft,
        l_qhalt=l_qhalt,
        l_ijepa=out.ijepa_loss,
        l_eff=l_eff,
        l_route=l_route,
        l_consistency=l_consistency,
        reward=reward.detach(),
        exact_match=exact.detach(),
        eff_sft_weight=eff_sft.detach(),
        eff_rl_weight=eff_rl.detach(),
        explore_strength=explore_strength.detach(),
    )
