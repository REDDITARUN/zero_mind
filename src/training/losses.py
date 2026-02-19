from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.models.unified_model import UnifiedForwardOutput


@dataclass
class LossBreakdown:
    total: torch.Tensor
    l_correct: torch.Tensor
    l_draft: torch.Tensor
    l_aux: torch.Tensor
    l_focus: torch.Tensor
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
    lr_warmup_steps: int = 500,
    w_draft: float = 0.5,
    w_aux: float = 0.3,
    w_focus: float = 1.0,
    w_consistency: float = 0.02,
    label_smoothing: float = 0.05,
    loss_clamp: float = 4.0,
    compute_penalty_coeff: float = 0.01,
    expert_budget_coeff: float = 0.25,
    **_kwargs,
) -> LossBreakdown:
    device = target_grid.device
    target = target_grid.unsqueeze(0)

    l_correct = F.cross_entropy(out.test_logits, target, label_smoothing=label_smoothing)
    l_draft = F.cross_entropy(out.draft_logits, target, label_smoothing=label_smoothing)
    l_aux = F.cross_entropy(out.aux_logits, target)

    draft_pred = out.draft_logits.argmax(dim=1)
    draft_wrong = (draft_pred != target).float()
    n_wrong = draft_wrong.sum()
    if n_wrong > 0:
        per_pixel_ce = F.cross_entropy(out.test_logits, target, reduction="none")
        l_focus = (per_pixel_ce * draft_wrong).sum() / n_wrong
    else:
        l_focus = torch.zeros((), device=device)

    if len(out.decoder.logits_trace) > 1:
        diffs = []
        for a, b in zip(out.decoder.logits_trace[:-1], out.decoder.logits_trace[1:]):
            diffs.append((b - a).abs().mean())
        l_consistency = torch.stack(diffs).mean()
    else:
        l_consistency = torch.zeros((), device=device)

    exact = _exact_match(out.test_logits, target_grid)
    pixel_acc = (out.test_logits.argmax(dim=1).squeeze(0) == target_grid).float().mean()
    compute_cost = compute_penalty_coeff * out.decoder.refinement_steps
    reward = exact + 0.5 * pixel_acc - compute_cost

    ramp = min(1.0, step / max(1, lr_warmup_steps))

    total = (
        l_correct
        + ramp * w_draft * l_draft
        + ramp * w_aux * l_aux
        + ramp * w_focus * l_focus
        + w_consistency * l_consistency
    )

    if loss_clamp > 0:
        total = torch.clamp(total, max=loss_clamp)

    return LossBreakdown(
        total=total,
        l_correct=l_correct,
        l_draft=l_draft,
        l_aux=l_aux,
        l_focus=l_focus,
        l_consistency=l_consistency,
        reward=reward.detach(),
        exact_match=exact.detach(),
    )
