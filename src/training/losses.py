from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.models.unified_model import UnifiedForwardOutput


@dataclass
class LossBreakdown:
    total: torch.Tensor
    l_sft: torch.Tensor
    l_aux: torch.Tensor
    l_dino: torch.Tensor
    l_consistency: torch.Tensor
    reward: torch.Tensor
    exact_match: torch.Tensor


def _exact_match(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = logits.argmax(dim=1).squeeze(0)
    return (pred == target).all().float()


def compute_unified_loss(
    out: UnifiedForwardOutput,
    target_grid: torch.Tensor,
    step: int,
    total_steps: int,
    lr_warmup_steps: int = 1000,
    w_aux: float = 0.3,
    w_dino: float = 0.5,
    w_consistency: float = 0.02,
    compute_penalty_coeff: float = 0.01,
    expert_budget_coeff: float = 0.25,
    # Legacy compat -- these are accepted but ignored
    sft_warmup_steps: int = 0,
    min_sft_weight: float = 0.0,
    sft_refresh_interval: int = 0,
    sft_refresh_span: int = 0,
    sft_refresh_min: float = 0.0,
    routing_explore_steps: int = 0,
    routing_explore_floor: float = 0.0,
    routing_entropy_bonus: float = 0.0,
    reward_pixel_coeff: float = 0.0,
    reward_exact_coeff: float = 1.0,
    w_qhalt: float = 0.0,
) -> LossBreakdown:
    # Main SFT loss
    l_sft = F.cross_entropy(out.test_logits, target_grid.unsqueeze(0))

    # Auxiliary direct readout loss (short gradient path for encoder)
    l_aux = F.cross_entropy(out.aux_logits, target_grid.unsqueeze(0))

    # DINO self-distillation (clamped to prevent dominating early training)
    l_dino = out.dino_loss.clamp(max=5.0)

    # Decoder consistency: refinement should converge
    if len(out.decoder.logits_trace) > 1:
        diffs = []
        for a, b in zip(out.decoder.logits_trace[:-1], out.decoder.logits_trace[1:]):
            diffs.append((b - a).abs().mean())
        l_consistency = torch.stack(diffs).mean()
    else:
        l_consistency = torch.zeros((), device=target_grid.device)

    # Metrics (for structure adaptation, not in gradient path of total loss)
    exact = _exact_match(out.test_logits, target_grid)
    pixel_acc = (out.test_logits.argmax(dim=1).squeeze(0) == target_grid).float().mean()
    compute_cost = compute_penalty_coeff * (out.reasoner.steps_used + out.decoder.refinement_steps)
    reward = reward_exact_coeff * exact + reward_pixel_coeff * pixel_acc - compute_cost

    # Ramp auxiliary losses gradually so early training focuses on SFT
    ramp = min(1.0, step / max(1, lr_warmup_steps))

    total = l_sft + ramp * w_aux * l_aux + ramp * w_dino * l_dino + w_consistency * l_consistency

    return LossBreakdown(
        total=total,
        l_sft=l_sft,
        l_aux=l_aux,
        l_dino=l_dino,
        l_consistency=l_consistency,
        reward=reward.detach(),
        exact_match=exact.detach(),
    )
