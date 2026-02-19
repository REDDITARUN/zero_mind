from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.models.unified_model import UnifiedForwardOutput


@dataclass
class LossBreakdown:
    total: torch.Tensor
    l_content: torch.Tensor
    l_shape: torch.Tensor
    l_aux: torch.Tensor
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
    w_shape: float = 0.3,
    w_aux: float = 0.3,
    w_consistency: float = 0.02,
    compute_penalty_coeff: float = 0.01,
    expert_budget_coeff: float = 0.25,
    **_kwargs,
) -> LossBreakdown:
    device = target_grid.device

    l_content = F.cross_entropy(out.test_logits, target_grid.unsqueeze(0))

    true_h, true_w = target_grid.shape
    l_shape_h = F.cross_entropy(out.shape_h_logits, torch.tensor([true_h - 1], device=device))
    l_shape_w = F.cross_entropy(out.shape_w_logits, torch.tensor([true_w - 1], device=device))
    l_shape = (l_shape_h + l_shape_w) * 0.5

    l_aux = F.cross_entropy(out.aux_logits, target_grid.unsqueeze(0))

    if len(out.decoder.logits_trace) > 1:
        diffs = []
        for a, b in zip(out.decoder.logits_trace[:-1], out.decoder.logits_trace[1:]):
            diffs.append((b - a).abs().mean())
        l_consistency = torch.stack(diffs).mean()
    else:
        l_consistency = torch.zeros((), device=device)

    exact = _exact_match(out.test_logits, target_grid)
    pixel_acc = (out.test_logits.argmax(dim=1).squeeze(0) == target_grid).float().mean()
    compute_cost = compute_penalty_coeff * (out.reasoner.steps_used + out.decoder.refinement_steps)
    reward = exact + 0.5 * pixel_acc - compute_cost

    ramp = min(1.0, step / max(1, lr_warmup_steps))

    total = l_content + ramp * w_shape * l_shape + ramp * w_aux * l_aux + w_consistency * l_consistency

    return LossBreakdown(
        total=total,
        l_content=l_content,
        l_shape=l_shape,
        l_aux=l_aux,
        l_consistency=l_consistency,
        reward=reward.detach(),
        exact_match=exact.detach(),
    )
